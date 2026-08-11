"""Sequential expected-improvement Bayesian optimization.

Replay mode consumes only the existing FEM table and is intended for logic
validation.  Online mode evaluates newly proposed feasible points in COMSOL.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import py1_generate_parameter_table as py1
import py6_optimization_common as common
OUTPUT_DIR = Path(__file__).resolve().parent / "py7_output"
VALIDATION_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential EI Bayesian optimization.")
    parser.add_argument("--mode", choices=("replay", "online"), default="replay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--initial-size", type=int, default=25)
    parser.add_argument(
        "--total-budget",
        type=int,
        default=75,
        help=(
            "Total FEM budget: 25 shared initial observations plus "
            "50 sequential BO evaluations by default."
        ),
    )
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--min-distance", type=float, default=1e-3, help="Minimum unit-cube distance from an observed point in online mode.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run with the same mode, seed, and budget.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deliberately replace an existing run for the same seed.",
    )
    return parser.parse_args()


def initial_and_validation(
    data: pd.DataFrame,
    seed: int,
    initial_size: int,
    validation_seed: int = VALIDATION_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool, validation = train_test_split(
        data, test_size=50, random_state=validation_seed, shuffle=True
    )
    if initial_size > len(train_pool):
        raise ValueError(f"initial-size must not exceed {len(train_pool)}.")
    order = np.random.RandomState(seed).permutation(train_pool.index)
    initial = train_pool.loc[order[:initial_size]].copy().reset_index(drop=True)
    available = train_pool.drop(index=order[:initial_size]).copy().reset_index(drop=True)
    return initial, validation.reset_index(drop=True), available


def run_seed(args: argparse.Namespace, seed: int) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.total_budget < args.initial_size:
        raise ValueError("total-budget must be at least initial-size.")
    data = common.load_fem_dataset()
    observations, validation, replay_pool = initial_and_validation(data, seed, args.initial_size)
    output = OUTPUT_DIR / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"method": "expected_improvement_bo", "mode": args.mode, "seed": seed, "initial_size": args.initial_size,
                 "total_budget": args.total_budget, "held_out_validation_size": len(validation), "xi": args.xi,
                 "validation_seed": VALIDATION_SEED,
                 "acquisition_optimization": "candidate-pool maximization of expected improvement", "candidate_pool_size": args.candidate_pool_size,
                 "min_distance": args.min_distance,
                 "design_space_definition": py1.design_space_definition(),
                 "design_space_fingerprint": py1.design_space_fingerprint()}
    if not args.resume and (output / "metadata.json").exists():
        existing = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if not args.overwrite:
            raise FileExistsError(
                f"{output} already contains {existing.get('mode')!r} results. "
                "Use --resume or --overwrite explicitly."
            )
    stored: dict = {}
    if args.resume and (output / "observations.csv").exists():
        stored = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        for key in (
            "method",
            "mode",
            "seed",
            "initial_size",
            "design_space_fingerprint",
        ):
            if stored.get(key) != metadata[key]:
                raise ValueError(f"Cannot resume: metadata mismatch for {key}.")
        stored_budget = int(stored.get("total_budget", 0))
        if stored_budget > args.total_budget:
            raise ValueError(
                "Cannot resume with a smaller total budget: "
                f"stored={stored_budget}, requested={args.total_budget}."
            )
        observations = pd.read_csv(output / "observations.csv")
        history = pd.read_csv(output / "iteration_history.csv").to_dict("records")
    else:
        initial_best = float(observations[common.TARGET_COLUMN].max())
        history = [{"iteration": 0, "phase": "initial_design", "cumulative_fem_calls": len(observations),
                    "best_before": initial_best, "observed_target": np.nan, "best_so_far": initial_best,
                    "gp_mean_before": np.nan, "gp_std_before": np.nan, "acquisition": np.nan,
                    "gp_fit_seconds": 0.0, "acquisition_seconds": 0.0,
                    "fem_evaluation_seconds": 0.0, "iteration_wall_seconds": 0.0,
                    "cumulative_wall_seconds": 0.0}]
    if len(observations) > args.total_budget:
        raise ValueError("Stored observations exceed total-budget.")
    if args.resume and len(observations) == args.total_budget and stored.get("completed") is True:
        print(f"Run already complete, skipping: {output}")
        return
    replay_pool = common.existing_pool_candidates(replay_pool, observations[common.ID_COLUMN])
    evaluator = None
    elapsed_before_resume = float(history[-1].get("cumulative_wall_seconds", 0.0))
    session_start = time.perf_counter()
    try:
        if args.mode == "online":
            from py2_high_fidelity_evaluation_engine import ComsolEvaluator
            evaluator = ComsolEvaluator()
            evaluator.open()
        for iteration in range(len(observations), args.total_budget):
            iteration_start = time.perf_counter()
            fit_start = time.perf_counter()
            bundle = common.fit_surrogate(observations, seed)
            fit_seconds = time.perf_counter() - fit_start
            best_before = float(observations[common.TARGET_COLUMN].max())
            acquisition_start = time.perf_counter()
            if args.mode == "replay":
                if replay_pool.empty:
                    raise RuntimeError("Replay candidate pool exhausted before budget.")
                u_pool = replay_pool[common.FEATURE_COLUMNS].to_numpy(float)
                u, mean, std, acquisition = common.select_acquisition_candidate(bundle, u_pool, best_before, kind="ei", xi=args.xi)
                distances = np.max(np.abs(u_pool - u), axis=1)
                choice = replay_pool.iloc[int(np.argmin(distances))].copy()
                evaluation_seconds = 0.0
                replay_pool = replay_pool.drop(index=choice.name).reset_index(drop=True)
                record = choice.to_dict()
            else:
                pool = common.feasible_lhs(args.candidate_pool_size, seed + iteration * 1009)
                pool = common.remove_near_observed(pool, observations[common.FEATURE_COLUMNS].to_numpy(float), args.min_distance)
                u, mean, std, acquisition = common.select_acquisition_candidate(bundle, pool, best_before, kind="ei", xi=args.xi)
                record = common.evaluate_online(evaluator, u, f"bo_{seed}_{iteration:04d}")
                evaluation_seconds = float(record["evaluation_time_seconds"])
            acquisition_seconds = time.perf_counter() - acquisition_start - evaluation_seconds
            acquisition_seconds = max(acquisition_seconds, 0.0)
            record.update({"gp_mean_before": mean, "gp_std_before": std, "acquisition_before": acquisition, "source": args.mode})
            observations = pd.concat([observations, pd.DataFrame([record])], ignore_index=True)
            history.append({"iteration": iteration - args.initial_size + 1, "cumulative_fem_calls": len(observations),
                            "best_before": best_before, "observed_target": float(record[common.TARGET_COLUMN]),
                            "best_so_far": float(observations[common.TARGET_COLUMN].max()), "gp_mean_before": mean,
                            "gp_std_before": std, "acquisition": acquisition, "gp_fit_seconds": fit_seconds,
                            "acquisition_seconds": acquisition_seconds,
                            "fem_evaluation_seconds": evaluation_seconds,
                            "iteration_wall_seconds": time.perf_counter() - iteration_start,
                            "cumulative_wall_seconds": elapsed_before_resume + time.perf_counter() - session_start})
            common.append_checkpoint(output, observations, pd.DataFrame(history), metadata)
            models_dir = output / "models"
            models_dir.mkdir(exist_ok=True)
            joblib.dump(bundle, models_dir / f"gp_iteration_{iteration:04d}.joblib")
        # fit/save final GP even when no sequential calls were requested.
        final_bundle = common.fit_surrogate(observations, seed)
        (output / "validation_set.csv").write_text(validation.to_csv(index=False), encoding="utf-8")
        (output / "models").mkdir(exist_ok=True)
        joblib.dump(final_bundle, output / "models" / "gp_final.joblib")
        metadata["completed"] = True
        metadata["actual_fem_calls"] = int(len(observations))
        common.append_checkpoint(output, observations, pd.DataFrame(history), metadata)
        common.save_run_summary(output, observations, pd.DataFrame(history), metadata)
        print(f"Saved BO run: {output}")
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as exc:
                print(f"Warning: COMSOL cleanup failed after saved results: {exc}")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in (common.SEEDS if args.all_seeds else (args.seed,)):
        run_seed(args, seed)


if __name__ == "__main__":
    main()
