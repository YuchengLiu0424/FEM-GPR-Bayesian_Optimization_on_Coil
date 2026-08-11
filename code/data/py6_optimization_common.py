"""Shared design-space, GP, acquisition, FEM, and persistence utilities.

This module is intentionally not a full experiment runner.  It is used by
py7--py10 so every optimization method uses the same bounds, feasibility
decision, GP settings, and COMSOL single-point interface.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import qmc

import py1_generate_parameter_table as py1
from py3_train_gp import fit_gp_from_observations, predict_gp_physical

SCRIPT_DIR = Path(__file__).resolve().parent
PY1_DIR = SCRIPT_DIR / "py1_output"
PY2_DIR = SCRIPT_DIR / "py2_output"
PY6_DIR = SCRIPT_DIR / "py6_output"
FEATURE_COLUMNS = list(py1.NORMALIZED_COLUMNS)
TARGET_COLUMN = "target"
ID_COLUMN = "sample_id"
SEEDS = (42, 43, 44, 45, 46)


def load_fem_dataset() -> pd.DataFrame:
    """Load the finalized 250-point dataset without modifying it."""
    path = PY2_DIR / "output.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing high-fidelity dataset: {path}")
    data = pd.read_csv(path)
    required = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"FEM dataset is missing required columns: {missing}")
    if not np.isfinite(data[FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy(float)).all():
        raise ValueError("FEM dataset contains non-finite values.")
    py1.validate_fem_dataset_contract(data, PY2_DIR / "metadata.json")
    return data.loc[:, required].copy()


def candidate_record(u: np.ndarray, sample_id: str = "candidate") -> dict[str, Any]:
    """Return the canonical py1 geometry/mass/feasibility record."""
    return py1.evaluate_unit_candidate(np.asarray(u, dtype=float), sample_id=sample_id).to_dict()


def is_feasible(u: np.ndarray) -> bool:
    return py1.is_unit_candidate_feasible(np.asarray(u, dtype=float))


def feasible_lhs(n: int, seed: int, *, max_batches: int = 40) -> np.ndarray:
    """Generate exactly *n* feasible unit-cube candidates using deterministic LHS."""
    if n < 1:
        raise ValueError("n must be positive")
    accepted: list[np.ndarray] = []
    dimension = len(FEATURE_COLUMNS)
    for batch in range(max_batches):
        sampler = qmc.LatinHypercube(d=dimension, seed=seed + batch)
        raw = sampler.random(max(2 * n, 64))
        evaluated = py1.evaluate_unit_candidates(raw)
        accepted.extend(raw[evaluated["feasible"].to_numpy(dtype=bool)])
        if len(accepted) >= n:
            return np.asarray(accepted[:n], dtype=float)
    raise RuntimeError("Could not generate enough feasible LHS candidates.")


def remove_near_observed(candidates: np.ndarray, observations: np.ndarray, min_distance: float = 1e-3) -> np.ndarray:
    """Exclude candidates within a unit-cube Euclidean distance of observations."""
    pool = np.asarray(candidates, dtype=float)
    seen = np.asarray(observations, dtype=float)
    if pool.ndim != 2 or pool.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("candidates must have shape (n, 7).")
    if seen.size == 0:
        return pool.copy()
    if seen.ndim != 2 or seen.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("observations must have shape (m, 7).")
    distances = np.linalg.norm(pool[:, None, :] - seen[None, :, :], axis=2)
    filtered = pool[np.min(distances, axis=1) > float(min_distance)]
    if len(filtered) == 0:
        raise RuntimeError("All acquisition candidates are too close to existing observations.")
    return filtered


def fit_surrogate(observations: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Fit exactly the py3 GP/scaler implementation to current observations."""
    if len(observations) < 2:
        raise ValueError("At least two observations are required to fit the GP.")
    bundle = fit_gp_from_observations(
        observations[FEATURE_COLUMNS].to_numpy(float),
        observations[TARGET_COLUMN].to_numpy(float),
        feature_columns=FEATURE_COLUMNS,
        random_state=seed,
    )
    bundle["target_col"] = TARGET_COLUMN
    return bundle


def predict_surrogate(bundle: dict[str, Any], candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return GP mean and latent-function standard deviation for acquisition."""
    return predict_gp_physical(bundle, candidates, uncertainty="latent")


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best_observed: float, xi: float = 0.0
) -> np.ndarray:
    """Expected improvement for maximization, evaluated in physical target units."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    improvement = mean - float(best_observed) - float(xi)
    safe_std = np.maximum(std, 1e-15)
    z = improvement / safe_std
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    value = improvement * ndtr(z) + safe_std * normal_pdf
    return np.where(std > 1e-15, value, np.maximum(improvement, 0.0))


def select_acquisition_candidate(
    bundle: dict[str, Any], candidates: np.ndarray, best_observed: float, *, kind: str = "ei", xi: float = 0.0
) -> tuple[np.ndarray, float, float, float]:
    """Choose a feasible point from a supplied candidate pool."""
    pool = np.asarray(candidates, dtype=float)
    if pool.ndim != 2 or pool.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("Candidate pool must have shape (n, 7).")
    mean, std = predict_surrogate(bundle, pool)
    if kind == "ei":
        score = expected_improvement(mean, std, best_observed, xi)
    elif kind == "mean":
        score = mean
    else:
        raise ValueError("kind must be 'ei' or 'mean'.")
    index = int(np.argmax(score))
    return pool[index].copy(), float(mean[index]), float(std[index]), float(score[index])


def evaluate_online(evaluator: Any, u: np.ndarray, sample_id: str) -> dict[str, Any]:
    """Evaluate one already-feasible unit point through COMSOL and enrich its record."""
    record = candidate_record(u, sample_id)
    if not bool(record["feasible"]):
        raise ValueError("Refusing to send an infeasible point to COMSOL.")
    sample = pd.Series(record)
    start = time.perf_counter()
    fem = evaluator.evaluate(sample)
    elapsed = time.perf_counter() - start
    return {**record, **fem, "evaluation_time_seconds": elapsed}


def append_checkpoint(directory: Path, observations: pd.DataFrame, history: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Write restartable plain-text state after each expensive FEM call."""
    directory.mkdir(parents=True, exist_ok=True)
    observations.to_csv(directory / "observations.csv", index=False)
    history.to_csv(directory / "iteration_history.csv", index=False)
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def save_run_summary(
    directory: Path,
    observations: pd.DataFrame,
    history: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Save the best design with complete inputs and a compact timing summary."""
    if observations.empty:
        raise ValueError("Cannot summarize an empty observation table.")
    best = observations.loc[observations[TARGET_COLUMN].idxmax()]
    design = candidate_record(
        best[FEATURE_COLUMNS].to_numpy(dtype=float),
        sample_id=str(best[ID_COLUMN]),
    )
    design.update({
        "method": metadata["method"],
        "mode": metadata["mode"],
        "seed": int(metadata["seed"]),
        TARGET_COLUMN: float(best[TARGET_COLUMN]),
    })
    pd.DataFrame([design]).to_csv(directory / "best_design.csv", index=False)

    initial_rows = history.loc[history["iteration"] == 0, "best_so_far"]
    if initial_rows.empty:
        raise ValueError("Run history is missing the iteration-0 initial design.")
    initial_best = float(initial_rows.iloc[0])
    final_best = float(best[TARGET_COLUMN])
    wall_seconds = float(history.get("cumulative_wall_seconds", pd.Series([0.0])).max())
    fem_seconds = float(history.get("fem_evaluation_seconds", pd.Series([0.0])).sum())
    gp_seconds = float(history.get("gp_fit_seconds", pd.Series([0.0])).sum())
    acquisition_seconds = float(history.get("acquisition_seconds", pd.Series([0.0])).sum())
    improvement = final_best - initial_best
    summary = {
        **metadata,
        "initial_best_response": initial_best,
        "final_best_response": final_best,
        "improvement": improvement,
        "optimization_wall_seconds": wall_seconds,
        "total_fem_seconds": fem_seconds,
        "total_gp_fit_seconds": gp_seconds,
        "total_acquisition_seconds": acquisition_seconds,
        "improvement_per_hour": improvement * 3600.0 / max(wall_seconds, 1e-12),
    }
    with (directory / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def existing_pool_candidates(data: pd.DataFrame, excluded_ids: Iterable[str]) -> pd.DataFrame:
    excluded = set(map(str, excluded_ids))
    return data.loc[~data[ID_COLUMN].astype(str).isin(excluded)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test shared optimization utilities.")
    parser.add_argument("--self-test", action="store_true", help="Write py6_output/self_test.json without calling COMSOL.")
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return
    data = load_fem_dataset()
    bundle = fit_surrogate(data.head(25), seed=42)
    candidate, mean, std, ei = select_acquisition_candidate(bundle, data.iloc[25:][FEATURE_COLUMNS], data.head(25)[TARGET_COLUMN].max())
    PY6_DIR.mkdir(parents=True, exist_ok=True)
    with (PY6_DIR / "self_test.json").open("w", encoding="utf-8") as handle:
        json.dump({"n_dataset": len(data), "candidate": candidate.tolist(), "mean": mean, "std": std, "ei": ei}, handle, indent=2)
    print(f"Saved self-test output to: {PY6_DIR / 'self_test.json'}")


if __name__ == "__main__":
    main()
