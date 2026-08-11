r"""Time the first 50 FEM observations of one completed py7 BO run.

Only the 50 individual FEM evaluation durations are used. COMSOL startup,
model-loading, cleanup, and shutdown time are deliberately excluded.

The estimated cost assigned to the 25 precomputed initial samples is:

    estimated_initial_25_fem_seconds = total_50_fem_seconds / 2

Example::

    D:\Tools\Python\python.exe py11_time_initial_fem_samples.py --seed 42

Use ``--resume`` after a stopped COMSOL/MPh session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import py1_generate_parameter_table as py1
import py6_optimization_common as common
from py2_high_fidelity_evaluation_engine import ComsolEvaluator, STUDY_TAG


SCRIPT_DIR = Path(__file__).resolve().parent
PY7_DIR = SCRIPT_DIR / "py7_output"
OUTPUT_DIR = SCRIPT_DIR / "py11_output"
EXPECTED_FEM_OBSERVATIONS = 50
INITIAL_FEM_OBSERVATIONS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate and time all 50 FEM observations from one completed py7 run."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-rtol", type=float, default=1e-3)
    parser.add_argument("--target-atol", type=float, default=1e-9)
    parser.add_argument(
        "--allow-target-drift",
        action="store_true",
        help=(
            "Continue timing if the current MPH response differs from the historical "
            "py7 target. The discrepancy remains recorded per sample."
        ),
    )
    return parser.parse_args()


def load_fifty_observations(seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = PY7_DIR / f"seed_{seed}"
    metadata_path = run_dir / "metadata.json"
    observations_path = run_dir / "observations.csv"
    if not metadata_path.exists() or not observations_path.exists():
        raise FileNotFoundError(f"Missing completed py7 run for seed {seed}: {run_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("design_space_fingerprint") != py1.design_space_fingerprint():
        raise ValueError(
            "Refusing to time a py7 run created under an obsolete "
            f"design-space contract: {metadata_path}"
        )
    if metadata.get("completed") is not True:
        raise ValueError(f"Refusing to time an incomplete py7 run: {metadata_path}")
    if int(metadata.get("actual_fem_calls", 0)) < EXPECTED_FEM_OBSERVATIONS:
        raise ValueError(
            f"py7 seed {seed} has actual_fem_calls="
            f"{metadata.get('actual_fem_calls')!r}; expected at least "
            f"{EXPECTED_FEM_OBSERVATIONS}."
        )
    if int(metadata.get("total_budget", 0)) < EXPECTED_FEM_OBSERVATIONS:
        raise ValueError(
            f"py7 seed {seed} total_budget must be at least "
            f"{EXPECTED_FEM_OBSERVATIONS}."
        )
    if metadata.get("initial_size") != INITIAL_FEM_OBSERVATIONS:
        raise ValueError(
            f"py7 seed {seed} initial_size must be {INITIAL_FEM_OBSERVATIONS}."
        )

    observations = pd.read_csv(observations_path)
    required = [common.ID_COLUMN, *common.FEATURE_COLUMNS, common.TARGET_COLUMN]
    missing = [column for column in required if column not in observations]
    if missing:
        raise ValueError(f"Missing required py7 observation columns: {missing}")
    if len(observations) < EXPECTED_FEM_OBSERVATIONS:
        raise ValueError(
            f"Expected at least {EXPECTED_FEM_OBSERVATIONS} observation rows, "
            f"found {len(observations)}."
        )

    selected = (
        observations.iloc[:EXPECTED_FEM_OBSERVATIONS][required]
        .copy()
        .reset_index(drop=True)
    )
    if selected[common.ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("The 50 py7 observation sample IDs must be unique.")
    values = selected[[*common.FEATURE_COLUMNS, common.TARGET_COLUMN]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("The 50 py7 observations contain non-finite inputs or targets.")
    return selected, metadata


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint(
    output: Path,
    results: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    atomic_write_csv(results, output / "fem50_timing.csv")
    atomic_write_json(metadata, output / "metadata.json")


def configure_stationary_point_evaluation(evaluator: ComsolEvaluator) -> dict[str, Any]:
    """Disable the COMSOL Optimization step in memory and retain Stationary."""
    if evaluator.model is None:
        raise RuntimeError("COMSOL evaluator is not open.")
    study = evaluator.model.java.study(STUDY_TAG)
    feature_types: dict[str, str] = {}
    disabled: list[str] = []
    stationary: list[str] = []
    for tag in list(study.feature().tags()):
        feature = study.feature(tag)
        feature_type = str(feature.getType())
        feature_types[str(tag)] = feature_type
        if feature_type.lower() == "optimization":
            feature.active(False)
            disabled.append(str(tag))
        elif feature_type.lower() == "stationary":
            feature.active(True)
            stationary.append(str(tag))
    if not stationary:
        raise RuntimeError(
            f"Study {STUDY_TAG!r} contains no Stationary step: {feature_types}"
        )
    return {
        "study_tag": STUDY_TAG,
        "study_feature_types": feature_types,
        "disabled_optimization_features": disabled,
        "active_stationary_features": stationary,
    }


def run(
    seed: int,
    resume: bool,
    target_rtol: float,
    target_atol: float,
    allow_target_drift: bool,
) -> None:
    observations, py7_metadata = load_fifty_observations(seed)
    output = OUTPUT_DIR / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    timing_path = output / "fem50_timing.csv"
    metadata_path = output / "metadata.json"

    if timing_path.exists():
        if not resume:
            raise FileExistsError(
                f"{timing_path} already exists. Use --resume or clear this seed output."
            )
        results = pd.read_csv(timing_path)
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored.get("seed") != seed:
            raise ValueError("Stored py11 seed does not match the requested seed.")
        if stored.get("expected_fem_observations") != EXPECTED_FEM_OBSERVATIONS:
            raise ValueError("Stored py11 output does not follow the 50-FEM contract.")
    else:
        results = pd.DataFrame()

    if not results.empty:
        discrepancy = results["target_absolute_difference"].astype(float)
        scale = results["original_target"].astype(float).abs()
        inconsistent = discrepancy > target_atol + target_rtol * scale
        results["target_consistent"] = ~inconsistent
        if inconsistent.any() and not allow_target_drift:
            raise ValueError(
                "Existing checkpoint contains target drift. Resume with "
                "--allow-target-drift for timing-only completion."
            )

    completed_ids = (
        set(results["sample_id"].astype(str)) if not results.empty else set()
    )
    remaining = observations.loc[
        ~observations[common.ID_COLUMN].astype(str).isin(completed_ids)
    ].copy()
    metadata: dict[str, Any] = {
        "seed": seed,
        "source": str(PY7_DIR / f"seed_{seed}" / "observations.csv"),
        "source_py7_method": py7_metadata.get("method"),
        "source_py7_initial_size": INITIAL_FEM_OBSERVATIONS,
        "source_py7_total_budget": int(py7_metadata["total_budget"]),
        "expected_fem_observations": EXPECTED_FEM_OBSERVATIONS,
        "initial_cost_definition": "sum of 50 pure FEM evaluations divided by 2",
        "startup_and_shutdown_time_included": False,
        "allow_target_drift": bool(allow_target_drift),
        "completed_samples": int(len(results)),
        "completed": False,
    }
    checkpoint(output, results, metadata)

    if remaining.empty:
        if len(results) != EXPECTED_FEM_OBSERVATIONS:
            raise ValueError("Checkpoint sample count is inconsistent with the contract.")
        print(f"FEM timing already contains 50 samples: {output}")
    else:
        evaluator = ComsolEvaluator()
        evaluator.open()
        metadata["stationary_point_evaluation"] = (
            configure_stationary_point_evaluation(evaluator)
        )
        checkpoint(output, results, metadata)
        failure: BaseException | None = None
        try:
            for _, source_row in remaining.iterrows():
                sample_id = str(source_row[common.ID_COLUMN])
                u = source_row[common.FEATURE_COLUMNS].to_numpy(float)
                original_target = float(source_row[common.TARGET_COLUMN])
                record = common.evaluate_online(evaluator, u, sample_id)
                rerun_target = float(record[common.TARGET_COLUMN])
                consistent = bool(np.isclose(
                    rerun_target,
                    original_target,
                    rtol=target_rtol,
                    atol=target_atol,
                ))
                record.update({
                    "original_target": original_target,
                    "rerun_target": rerun_target,
                    "target_absolute_difference": abs(rerun_target - original_target),
                    "target_relative_difference": (
                        abs(rerun_target - original_target)
                        / max(abs(original_target), 1e-15)
                    ),
                    "target_consistent": consistent,
                    "timing_order": int(len(results) + 1),
                })
                results = pd.concat(
                    [results, pd.DataFrame([record])],
                    ignore_index=True,
                )
                metadata["completed_samples"] = int(len(results))
                checkpoint(output, results, metadata)
                print(
                    f"[{len(results)}/{EXPECTED_FEM_OBSERVATIONS}] {sample_id}: "
                    f"{record['evaluation_time_seconds']:.3f} s"
                )
                if not consistent and not allow_target_drift:
                    raise RuntimeError(
                        f"{sample_id}: rerun target {rerun_target:.12g} T differs "
                        f"from original {original_target:.12g} T."
                    )
        except BaseException as exc:
            failure = exc
        finally:
            try:
                evaluator.close()
            except Exception as exc:
                print(f"Warning: COMSOL cleanup failed: {exc}")

        metadata["completed_samples"] = int(len(results))
        metadata["completed"] = len(results) == EXPECTED_FEM_OBSERVATIONS
        checkpoint(output, results, metadata)
        if failure is not None:
            print(
                f"Saved checkpoint at {len(results)}/{EXPECTED_FEM_OBSERVATIONS}. "
                "Restart with --resume."
            )
            raise failure

    if len(results) != EXPECTED_FEM_OBSERVATIONS:
        raise RuntimeError("Cannot summarize before all 50 FEM evaluations finish.")
    fem_seconds = results["evaluation_time_seconds"].astype(float)
    if not np.isfinite(fem_seconds).all() or (fem_seconds <= 0.0).any():
        raise ValueError("FEM timing values must be finite and positive.")
    if "target_consistent" not in results:
        results["target_consistent"] = np.isclose(
            results["rerun_target"].astype(float),
            results["original_target"].astype(float),
            rtol=target_rtol,
            atol=target_atol,
        )
        checkpoint(output, results, metadata)
    consistent = results["target_consistent"].astype(bool)
    total_50 = float(fem_seconds.sum())
    summary = {
        **metadata,
        "completed": True,
        "n_samples": EXPECTED_FEM_OBSERVATIONS,
        "total_50_fem_seconds": total_50,
        "estimated_initial_25_fem_seconds": total_50 / 2.0,
        "fem_seconds_mean": float(fem_seconds.mean()),
        "fem_seconds_std": float(fem_seconds.std(ddof=1)),
        "fem_seconds_min": float(fem_seconds.min()),
        "fem_seconds_max": float(fem_seconds.max()),
        "startup_and_shutdown_time_included": False,
        "target_consistent_samples": int(consistent.sum()),
        "target_inconsistent_samples": int((~consistent).sum()),
        "valid_for_fem_timing": True,
        "valid_for_historical_target_reproduction": bool(consistent.all()),
    }
    atomic_write_json(summary, output / "timing_summary.json")
    print(f"Saved completed 50-FEM timing benchmark: {output}")
    print(f"Pure 50-FEM time: {total_50:.3f} s")
    print(f"Assigned initial-25 FEM time: {total_50 / 2.0:.3f} s")


def main() -> None:
    args = parse_args()
    run(
        args.seed,
        args.resume,
        args.target_rtol,
        args.target_atol,
        args.allow_target_drift,
    )


if __name__ == "__main__":
    main()
