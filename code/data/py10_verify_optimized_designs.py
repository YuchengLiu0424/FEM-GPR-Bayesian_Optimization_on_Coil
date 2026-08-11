"""Independently verify final optimization candidates with COMSOL when requested."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import py1_generate_parameter_table as py1
import py6_optimization_common as common

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "py10_output"


def candidates(mode: str) -> list[tuple[str, int | None, pd.Series, Path]]:
    """Collect completed py7 and shared-seed py8 optimization runs."""
    found = []
    roots = [("expected_improvement_bo", SCRIPT_DIR / "py7_output")]
    py8_root = SCRIPT_DIR / "py8_output"
    if py8_root.exists():
        roots.extend(
            (path.name, path)
            for path in sorted(py8_root.iterdir())
            if path.is_dir()
        )
    for method, root in roots:
        if not root.exists():
            continue
        files = list(root.glob("seed_*/observations.csv"))
        for file in sorted(set(files)):
            metadata_path = file.parent / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata for {file}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("design_space_fingerprint") != py1.design_space_fingerprint():
                raise ValueError(
                    "Refusing an optimization run created under an obsolete "
                    f"design-space contract: {metadata_path}"
                )
            if metadata.get("mode") != mode:
                continue
            if metadata.get("completed") is not True:
                raise ValueError(f"Refusing incomplete run: {metadata_path}")
            actual_calls = metadata.get("actual_fem_calls")
            budget_calls = metadata.get(
                "budget_accounted_fem_calls", actual_calls
            )
            total_budget = metadata.get("total_budget")
            if (
                not isinstance(actual_calls, int)
                or not isinstance(budget_calls, int)
                or not isinstance(total_budget, int)
                or budget_calls > total_budget
            ):
                raise ValueError(f"Invalid FEM budget accounting: {metadata_path}")
            if (
                method == "expected_improvement_bo"
                and budget_calls != total_budget
            ):
                raise ValueError(f"Refusing unfinished py7 FEM budget: {metadata_path}")
            frame = pd.read_csv(file)
            target = pd.to_numeric(frame[common.TARGET_COLUMN], errors="coerce")
            eligible = frame.loc[np.isfinite(target)].copy()
            if "feasible" in eligible.columns:
                feasible = eligible["feasible"].map(
                    lambda value: str(value).strip().lower() in {"true", "1"}
                )
                eligible = eligible.loc[feasible]
            if eligible.empty:
                raise ValueError(f"No feasible finite observations found in {file}")
            best = eligible.loc[
                pd.to_numeric(eligible[common.TARGET_COLUMN], errors="coerce").idxmax()
            ]
            seed_value = metadata.get("seed")
            seed = int(seed_value) if seed_value is not None else None
            found.append((method, seed, best, file.parent))
    if not found: raise FileNotFoundError("No py7/py8 observations.csv files found.")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or perform independent final COMSOL verification.")
    parser.add_argument("--run-comsol", action="store_true", help="Actually launch COMSOL and re-evaluate final candidates.")
    parser.add_argument("--mode", choices=("replay", "online"), default="replay")
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--min-distance", type=float, default=1e-3)
    args = parser.parse_args(); output_dir = OUTPUT_DIR; output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    evaluator = None
    try:
        if args.run_comsol:
            from py2_high_fidelity_evaluation_engine import ComsolEvaluator
            evaluator = ComsolEvaluator()
            evaluator.open()
        for method, seed, best, run_dir in candidates(args.mode):
            row = {"method": method, "seed": seed, "candidate_type": "best_observed", "sample_id": str(best[common.ID_COLUMN]), "optimization_observed_target": float(best[common.TARGET_COLUMN]),
                   "predicted_mean_before": float(best.get("gp_mean_before", np.nan)), "predicted_std_before": float(best.get("gp_std_before", np.nan))}
            row.update({name: float(best[name]) for name in common.FEATURE_COLUMNS})
            candidate_label = seed if seed is not None else run_dir.name
            physical = common.candidate_record(np.array([row[name] for name in common.FEATURE_COLUMNS]), f"verify_{method}_{candidate_label}")
            row.update({name: physical[name] for name in ("c1_mm", "r1_mm", "t1_mm", "l1_mm", "l2_mm", "w1_mm", "h1_mm", "Mcore_g", "Mcoil_g", "Mtotal_g", "feasible")})
            if evaluator is not None:
                checked = common.evaluate_online(evaluator, np.array([row[name] for name in common.FEATURE_COLUMNS]), f"verify_{method}_{candidate_label}")
                row["verified_fem_target"] = float(checked[common.TARGET_COLUMN]); row["verification_error"] = row["verified_fem_target"] - row["optimization_observed_target"]
            rows.append(row)
            if method not in {"expected_improvement_bo", "gp_mean"}:
                continue
            observations = pd.read_csv(run_dir / "observations.csv")
            if seed is None:
                raise ValueError(f"GP-based method is missing a seed: {run_dir}")
            model_path = (
                run_dir / "models" / "gp_final.joblib"
                if method == "expected_improvement_bo"
                else run_dir / "gp_final.joblib"
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Missing final GP model: {model_path}")
            bundle = joblib.load(model_path)
            pool = common.feasible_lhs(args.candidate_pool_size, seed + 99001)
            pool = common.remove_near_observed(pool, observations[common.FEATURE_COLUMNS].to_numpy(float), args.min_distance)
            u, mean, std, _ = common.select_acquisition_candidate(bundle, pool, float(observations[common.TARGET_COLUMN].max()), kind="mean")
            recommended = common.candidate_record(u, f"gp_recommendation_{method}_{seed}")
            gp_row = {"method": method, "seed": seed, "candidate_type": "final_gp_recommendation", "sample_id": recommended[common.ID_COLUMN],
                      "optimization_observed_target": np.nan, "predicted_mean_before": mean, "predicted_std_before": std}
            gp_row.update({name: float(recommended[name]) for name in common.FEATURE_COLUMNS})
            gp_row.update({name: recommended[name] for name in ("c1_mm", "r1_mm", "t1_mm", "l1_mm", "l2_mm", "w1_mm", "h1_mm", "Mcore_g", "Mcoil_g", "Mtotal_g", "feasible")})
            if evaluator is not None:
                checked = common.evaluate_online(evaluator, u, str(gp_row["sample_id"]))
                gp_row["verified_fem_target"] = float(checked[common.TARGET_COLUMN])
                gp_row["verification_error"] = gp_row["verified_fem_target"] - gp_row["predicted_mean_before"]
                gp_row["standardized_verification_residual"] = gp_row["verification_error"] / max(gp_row["predicted_std_before"], 1e-15)
            rows.append(gp_row)
    finally:
        if evaluator is not None: evaluator.close()
    pd.DataFrame(rows).to_csv(output_dir / "verification_results.csv", index=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f: json.dump(
        {
            "mode": args.mode,
            "comsol_rerun": bool(args.run_comsol),
            "n_candidates": len(rows),
            "design_space_definition": py1.design_space_definition(),
            "design_space_fingerprint": py1.design_space_fingerprint(),
        },
        f,
        indent=2,
    )
    print(f"Saved verification results to: {output_dir}")


if __name__ == "__main__": main()
