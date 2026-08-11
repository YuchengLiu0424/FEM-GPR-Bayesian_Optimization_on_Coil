"""Recompute manuscript numerical claims from the current archived outputs.

This audit is read-only with respect to all experiment outputs and the COMSOL
model.  It writes only a machine-readable audit report under ``py18_output``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PAPER = PROJECT / "paper" / "conference_101719.tex"
OUTPUT = ROOT / "py18_output"
FINGERPRINT = "7c19f05da8d0f8f526ca172bae6f48d0a05d79b894317764f0e29e8c8797cfce"
MODEL_HASH = "1c261222bc6281419b1b1d2b24e510c30acc6dbff83d4ceff3f10949d66cfcd3"


records: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(claim: str, observed: Any, expected: Any, source: str, passed: bool, note: str = "") -> None:
    records.append(
        {
            "claim": claim,
            "observed": observed,
            "expected_in_manuscript": expected,
            "source": source,
            "status": "PASS" if passed else "FAIL",
            "note": note,
        }
    )


def close(claim: str, observed: float, expected: float, source: str, atol: float, note: str = "") -> None:
    record(claim, float(observed), float(expected), source, bool(np.isclose(observed, expected, rtol=0.0, atol=atol)), note)


def exact(claim: str, observed: Any, expected: Any, source: str, note: str = "") -> None:
    record(claim, observed, expected, source, observed == expected, note)


def audit_contract() -> None:
    design = pd.read_csv(ROOT / "py1_output" / "design_points_samples.csv")
    fem = pd.read_csv(ROOT / "py2_output" / "output.csv")
    exact("FEM table row count", len(fem), 250, "py2_output/output.csv")
    exact("FEM sample IDs unique", bool(fem.sample_id.astype(str).is_unique), True, "py2_output/output.csv")
    exact("FEM targets finite", bool(np.isfinite(fem.target).all()), True, "py2_output/output.csv")
    exact("Design-table row count", len(design), 250, "py1_output/design_points_samples.csv")
    exact(
        "Design-table SHA-256",
        sha256(ROOT / "py1_output" / "design_points_samples.csv"),
        json.loads((ROOT / "py2_output" / "metadata.json").read_text(encoding="utf-8"))["design_file_sha256"],
        "py2_output/metadata.json",
    )
    exact(
        "Source MPH SHA-256",
        sha256(PROJECT / "surrogate_simulation" / "ML surrogate.mph"),
        MODEL_HASH,
        "source MPH and manuscript",
    )

    bad_fingerprints = []
    bad_model_hashes = []
    for path in ROOT.glob("py*_output/**/metadata.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "design_space_fingerprint" in data and data["design_space_fingerprint"] != FINGERPRINT:
            bad_fingerprints.append(str(path.relative_to(ROOT)))
        if "model_sha256" in data and str(data["model_sha256"]).lower() != MODEL_HASH:
            bad_model_hashes.append(str(path.relative_to(ROOT)))
    exact("All metadata design-space fingerprints current", bad_fingerprints, [], "all py*_output/**/metadata.json")
    exact("All stored model hashes current", bad_model_hashes, [], "all py*_output/**/metadata.json")

    max_coil_g = math.pi * 8940.0 * ((0.025**2) - (0.020**2)) * 0.005 * 1000.0
    close("Analytic maximum coil mass (g)", max_coil_g, 31.6, "py1 coil-mass formula and box bounds", 0.05)
    low_mu = 0.12 / (4 * math.pi * 1e-7 * 795.77)
    close("First-interval secant relative permeability", low_mu, 120.0, "py16_output/bh_curve.csv", 0.1)


def audit_surrogate() -> None:
    frames = []
    for index in range(1, 6):
        frame = pd.read_csv(ROOT / "py5_output" / f"5.{index}" / "learning_curve_results.csv")
        frame["repeat"] = index
        frames.append(frame)
    learning = pd.concat(frames, ignore_index=True)
    aggregate = learning.groupby("train_size").agg(
        r2_mean=("validation_r2", "mean"), r2_sd=("validation_r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
        c1_mean=("coverage_1sigma", "mean"), c1_sd=("coverage_1sigma", "std"),
        c2_mean=("coverage_2sigma", "mean"), c2_sd=("coverage_2sigma", "std"),
    )
    manuscript = {
        25: (0.9872, 0.0041, 20.32, 2.92, 14.07, 1.63, .516, .097, .776, .085),
        50: (0.9973, 0.0013, 9.14, 2.17, 6.83, 1.62, .580, .110, .856, .071),
        75: (0.9985, 0.0008, 6.77, 1.86, 4.77, 1.14, .572, .117, .868, .061),
        100: (0.9991, 0.0004, 5.33, 1.12, 3.61, .53, .668, .046, .920, .035),
        125: (0.9992, 0.0004, 4.96, 1.30, 3.18, .75, .720, .075, .908, .030),
        150: (0.9994, 0.0003, 4.38, 1.39, 2.78, .77, .712, .083, .908, .058),
        175: (0.9995, 0.0003, 3.69, 1.14, 2.30, .41, .732, .046, .896, .043),
        200: (0.9996, 0.0002, 3.29, .84, 2.09, .29, .736, .046, .908, .023),
    }
    columns = ["r2_mean", "r2_sd", "rmse_mean", "rmse_sd", "mae_mean", "mae_sd", "c1_mean", "c1_sd", "c2_mean", "c2_sd"]
    tolerances = [5e-5, 5e-5, .005, .005, .005, .005, .0005, .0005, .0005, .0005]
    for size, expected in manuscript.items():
        row = aggregate.loc[size].copy()
        row[["rmse_mean", "rmse_sd", "mae_mean", "mae_sd"]] *= 1e6
        for column, value, tolerance in zip(columns, expected, tolerances):
            close(f"Learning curve N={size} {column}", row[column], value, "py5_output/5.1--5.5", tolerance)

    metrics = json.loads((ROOT / "py4_output" / "metrics.json").read_text(encoding="utf-8"))
    calibration = pd.read_csv(ROOT / "py12_output" / "calibration_metrics_by_split.csv").set_index("seed")
    close("Seed-42 R2", metrics["r2"], .9993, "py4_output/metrics.json", 5e-5)
    close("Seed-42 RMSE (microtesla)", metrics["rmse"] * 1e6, 4.680, "py4_output/metrics.json", .0005)
    close("Seed-42 MAE (microtesla)", metrics["mae"] * 1e6, 2.235, "py4_output/metrics.json", .0005)
    close("Seed-42 1-sigma coverage", metrics["within_1sigma"], .76, "py4_output/metrics.json", 5e-9)
    close("Seed-42 2-sigma coverage", metrics["within_2sigma"], .90, "py4_output/metrics.json", 5e-9)
    close("Seed-42 residual mean", calibration.loc[42, "standardized_residual_mean"], -.449, "py12 calibration", .0005)
    close("Seed-42 residual sample SD", calibration.loc[42, "standardized_residual_sd"], 1.668, "py12 calibration", .0005)
    close("Seed-42 mean NLPD T", calibration.loc[42, "mean_nlpd_T"], -10.708, "py12 calibration", .0005)
    # Explicit prose claim audit: 0.76 is above 68.27%, while 0.90 is below 95.45%.
    record(
        "Prose: seed-42 C1 above and C2 below Gaussian references",
        {"C1": metrics["within_1sigma"], "C2": metrics["within_2sigma"]},
        "C1 above 0.6827 and C2 below 0.9545",
        "py4_output/metrics.json",
        metrics["within_1sigma"] > .6827 and metrics["within_2sigma"] < .9545,
        "",
    )

    summary = json.loads((ROOT / "py12_output" / "summary.json").read_text(encoding="utf-8"))
    close("Pooled residual mean", summary["pooled_standardized_residual_mean"], -.122, "py12 summary", .0005)
    close("Pooled residual sample SD", summary["pooled_standardized_residual_sd"], 1.315, "py12 summary", .0005)
    close("Mean split NLPD T", summary["mean_split_nlpd_T"], -11.178, "py12 summary", .0005)
    close("Mean split NLPD standardized", summary["mean_split_nlpd_standardized_target"], -2.523, "py12 summary", .0005)

    ard = pd.read_csv(ROOT / "py12_output" / "ard_length_scale_summary.csv")
    expected_ard = {
        "u_h1": (21.10, 3.36, 1), "u_w1": (22.56, 4.14, 2),
        "u_t1": (28.74, 3.35, 3), "u_l1": (28.84, 3.18, 4),
        "u_l2": (35.29, 6.24, 5), "u_r1": (36.58, 3.82, 6),
        "u_c1": (591.41, 176.53, 7),
    }
    for variable, (mean, sd, rank) in expected_ard.items():
        row = ard.loc[ard.variable == variable].iloc[0]
        close(f"ARD {variable} mean", row.length_scale_mean, mean, "py12 ARD summary", .005)
        close(f"ARD {variable} SD", row.length_scale_sd, sd, "py12 ARD summary", .005)
        exact(f"ARD {variable} rank", int(row.surrogate_sensitivity_rank), rank, "py12 ARD summary")
    by_split = pd.read_csv(ROOT / "py12_output" / "ard_length_scales_by_split.csv")
    close("Signal variance minimum", by_split.signal_variance.min(), 70.04, "py12 ARD by split", .005)
    close("Signal variance maximum", by_split.signal_variance.max(), 231.72, "py12 ARD by split", .005)
    close("LML minimum", by_split.log_marginal_likelihood.min(), 397.01, "py12 ARD by split", .005)
    close("LML maximum", by_split.log_marginal_likelihood.max(), 419.71, "py12 ARD by split", .005)
    close("White-noise minimum", by_split.white_noise_level.min(), 1e-10, "py12 ARD by split", 1e-14)
    close("White-noise maximum", by_split.white_noise_level.max(), 3.04e-5, "py12 ARD by split", 5e-8)


def audit_optimization() -> None:
    summary = pd.read_csv(ROOT / "py9_output" / "optimization_summary.csv").set_index("method")
    expected_terminal = {
        "bobyqa": (.943, .027, .910, .974), "cobyla": (.940, .021, .918, .965),
        "expected_improvement_bo": (.909, .012, .890, .920),
        "nelder_mead": (.743, .062, .636, .780), "ego": (.724, .033, .665, .740),
    }
    protocol = pd.read_csv(ROOT / "py9_output" / "optimization_efficiency.csv")
    for method, (mean, sd, minimum, maximum) in expected_terminal.items():
        row = summary.loc[method]
        close(f"{method} terminal mean mT", row.final_response_mean * 1e3, mean, "py9 optimization summary", .0005)
        close(f"{method} terminal SD mT", row.final_response_std * 1e3, sd, "py9 optimization summary", .0005)
        values = protocol.loc[protocol.method == method, "final_best_response"] * 1e3
        close(f"{method} terminal minimum mT", values.min(), minimum, "py9 run protocol", .0005)
        close(f"{method} terminal maximum mT", values.max(), maximum, "py9 run protocol", .0005)

    history = pd.read_csv(ROOT / "py9_output" / "optimization_history.csv")
    checkpoints = {
        "expected_improvement_bo": {25: .675, 30: .883, 40: .890, 50: .893, 75: .909},
        "bobyqa": {25: .675, 30: .675, 40: .685, 50: .830, 75: .943},
        "cobyla": {25: .675, 30: .938}, "ego": {25: .675, 30: .675, 40: .680, 50: .680},
        "nelder_mead": {25: .675, 30: .680, 40: .696, 50: .713},
    }
    for method, values in checkpoints.items():
        for count, expected in values.items():
            subset = history.loc[(history.method == method) & (history.cumulative_fem_calls == count), "best_so_far"]
            close(f"{method} checkpoint n={count} mean mT", subset.mean() * 1e3, expected, "py9 optimization history", .0005)
    # Reported uncertainty values in prose.
    for method, count, mean, sd in [
        ("expected_improvement_bo", 30, .883, .019), ("bobyqa", 75, .943, .027),
        ("expected_improvement_bo", 75, .909, .012), ("cobyla", 30, .938, .018),
    ]:
        subset = history.loc[(history.method == method) & (history.cumulative_fem_calls == count), "best_so_far"] * 1e3
        close(f"{method} n={count} prose mean", subset.mean(), mean, "py9 optimization history", .0005)
        close(f"{method} n={count} prose sample SD", subset.std(ddof=1), sd, "py9 optimization history", .0005)

    cob = protocol.loc[protocol.method == "cobyla"].sort_values("seed")
    exact("COBYLA cumulative counts", cob.budget_accounted_fem_calls.astype(int).tolist(), [34, 37, 34, 34, 41], "py9 run protocol")
    expected_cob = [.9652, .9513, .9183, .9182, .9494]
    for seed, value, expected in zip(cob.seed.astype(int), cob.final_best_response * 1e3, expected_cob):
        close(f"COBYLA seed {seed} terminal mT", value, expected, "py9 run protocol", .00005)
    exact("Only one COBYLA run reaches n=40", int((cob.budget_accounted_fem_calls >= 40).sum()), 1, "py9 run protocol")

    stats = pd.read_csv(ROOT / "py9_output" / "statistical_comparisons.csv")
    def stat(a: str, b: str) -> pd.Series:
        direct = stats.loc[(stats.method_a == a) & (stats.method_b == b)]
        if not direct.empty:
            return direct.iloc[0]
        reverse = stats.loc[(stats.method_a == b) & (stats.method_b == a)]
        if reverse.empty:
            raise KeyError(f"Missing statistical comparison: {a}, {b}")
        row = reverse.iloc[0].copy()
        row["mean_final_difference_a_minus_b"] *= -1
        row["test_statistic"] *= -1
        row["effect_size"] *= -1
        return row
    for a, b, delta, delta_tolerance, dz, p in [
        ("expected_improvement_bo", "ego", .186, .0005, 5.05, 3.51e-4),
        ("expected_improvement_bo", "nelder_mead", .166, .0005, 2.42, 5.64e-3),
        ("bobyqa", "expected_improvement_bo", .03385, .000005, 2.02, .0106),
        ("cobyla", "expected_improvement_bo", .0313, .00005, 1.09, .0714),
    ]:
        row = stat(a, b)
        close(f"{a}-{b} mean terminal delta mT", row.mean_final_difference_a_minus_b * 1e3, delta, "py9 statistical comparisons", delta_tolerance)
        close(f"{a}-{b} Cohen dz", row.effect_size, dz, "py9 statistical comparisons", .005)
        close(f"{a}-{b} paired p", row.p_value, p, "py9 statistical comparisons", max(5e-7, abs(p) * .001))

    timing_expected = {
        "expected_improvement_bo": (75, 178.3, 28.3), "bobyqa": (75, 116.4, 1.8),
        "cobyla": (36, 91.6, 27.7), "ego": (75, 109.6, 3.8), "nelder_mead": (75, 122.0, 5.1),
    }
    for method, (calls, mean, sd) in timing_expected.items():
        row = summary.loc[method]
        close(f"{method} accounted calls mean", row.budget_accounted_fem_calls_mean, calls, "py9 optimization summary", .05)
        close(f"{method} end-to-end mean s", row.end_to_end_wall_time_mean_seconds, mean, "py9 optimization summary", .05)
        close(f"{method} end-to-end SD s", row.end_to_end_wall_time_std_seconds, sd, "py9 optimization summary", .05)
    close("EGO raw calls mean", summary.loc["ego", "actual_fem_calls_mean"], 75.2, "py9 optimization summary", .01)
    ego43 = protocol.loc[(protocol.method == "ego") & (protocol.seed == 43)].iloc[0]
    exact("EGO seed43 raw calls", int(ego43.actual_fem_calls), 76, "py9 run protocol")
    exact("EGO seed43 accounted calls", int(ego43.budget_accounted_fem_calls), 75, "py9 run protocol")


def audit_policy_and_physics() -> None:
    kernel = pd.read_csv(ROOT / "py13_output" / "kernel_screening_summary.csv").set_index("model")
    close("Selected BoTorch RMSE microtesla", kernel.loc["botorch_matern52_ls5", "rmse_T_mean"] * 1e6, 3.162, "py13 kernel summary", .0005)
    close("Production sklearn RMSE microtesla", kernel.loc["sklearn_matern52_current", "rmse_T_mean"] * 1e6, 3.289, "py13 kernel summary", .0005)
    close("Selected BoTorch 95% coverage", kernel.loc["botorch_matern52_ls5", "coverage_95_mean"], .996, "py13 kernel summary", .0005)
    aggregate = pd.read_csv(ROOT / "py13_output" / "paired_summary_aggregate.csv").set_index("method")
    expected = {
        "random": (.713, .055, .039, .053), "sklearn_mean": (.738, 0, .063, .097),
        "sklearn_ei": (.738, 0, .063, .097), "botorch_mean": (.738, 0, .063, .097),
        "botorch_logei": (.738, 0, .063, .097),
    }
    for method, values in expected.items():
        row = aggregate.loc[method]
        actual = [row.final_best_T_mean * 1e3, row.final_best_T_std * 1e3, row.improvement_T_mean * 1e3, row.improvement_T_std * 1e3]
        for label, observed, target in zip(("endpoint mean", "endpoint SD", "improvement mean", "improvement SD"), actual, values):
            close(f"Policy {method} {label} mT", observed, target, "py13 paired summary", .0005)
    by_seed = pd.read_csv(ROOT / "py13_output" / "paired_summary_by_seed.csv")
    pool_max = pd.read_csv(ROOT / "py2_output" / "output.csv").target.max()
    initial = by_seed.loc[by_seed.method == "sklearn_ei", ["seed", "initial_best_T"]]
    exact("Seeds with pool maximum in initial 25", initial.loc[np.isclose(initial.initial_best_T, pool_max), "seed"].astype(int).tolist(), [43, 45, 46], "py13 paired summary and py2 pool")

    verify = pd.read_csv(ROOT / "py10_output" / "verification_results.csv")
    selected = {}
    for method in ("expected_improvement_bo", "bobyqa", "cobyla"):
        subset = verify.loc[(verify.method == method) & (verify.candidate_type == "best_observed")]
        selected[method] = subset.loc[subset.verified_fem_target.idxmax()]
    manuscript_designs = {
        "expected_improvement_bo": ([1.2371, 11.8847, 9.3159, 7.7991, 14.8635, 4.9086, 4.9370], 170.38, 19.52, .8621, .9195),
        "bobyqa": ([.5, 10.0921, 10, 7.0412, 19.7206, 5, 5], 190.70, 17.68, .0412, .9737),
        "cobyla": ([3.0313, 10, 10, 7, 18.0930, 5, 5], 169.48, 17.55, 0, .9652),
    }
    names = ["c1_mm", "r1_mm", "t1_mm", "l1_mm", "l2_mm", "w1_mm", "h1_mm"]
    for method, (dims, core, coil, margin, response) in manuscript_designs.items():
        row = selected[method]
        for name, target in zip(names, dims):
            close(f"Selected {method} {name}", row[name], target, "py10 verification results", .00005)
        close(f"Selected {method} core mass g", row.Mcore_g, core, "py10 verification results", .005)
        close(f"Selected {method} coil mass g", row.Mcoil_g, coil, "py10 verification results", .005)
        close(f"Selected {method} axial margin mm", row.l1_mm - row.h1_mm - 2, margin, "py10 verification results", .00005)
        # Table uses optimization-stage observed response, not independent-rerun response.
        close(f"Selected {method} optimization response mT", row.optimization_observed_target * 1e3, response, "py10 verification results", .00005)

    physics = pd.read_csv(ROOT / "py16_output" / "design_summary.csv").set_index("label")
    expected_physics = {
        "EI-BO": (24.234, .91952, .91952, 2.858, 18.981, 3),
        "BOBYQA": (25.000, .97361, .94378, 2.615, 63.424, 1),
        "COBYLA": (25.000, .96515, .93558, 2.816, 19.444, 2),
    }
    for label, values in expected_physics.items():
        row = physics.loc[label]
        actual = [row.original_current_A, row.original_target_T * 1e3, row.fixed_current_target_T * 1e3, row.core_B_mean_T * 1e3, row.core_B_max_T * 1e3, int(row.fixed_current_rank)]
        tolerances = [.0005, .000005, .000005, .0005, .0005, 0]
        for name, observed, target, tolerance in zip(("I", "production ROI", "common-I ROI", "core mean B", "core max B", "rank"), actual, values, tolerances):
            if name == "rank": exact(f"Physics {label} {name}", observed, target, "py16 design summary")
            else: close(f"Physics {label} {name}", observed, target, "py16 design summary", tolerance)
    close("Common-current reduction of BOBYQA percent", -physics.loc["BOBYQA", "fixed_vs_original_relative"] * 100, 3.064, "py16 design summary", .0005)
    close("Original 25A vs 24.234A current difference percent", (25 / physics.loc["EI-BO", "original_current_A"] - 1) * 100, 3.16, "py16 design summary", .005)

    bh = pd.read_csv(ROOT / "py16_output" / "bh_curve.csv")
    exact("B-H table row count", len(bh), 63, "py16 B-H table")
    close("B-H first nonzero H", bh.iloc[1].H_Am, 795.77, "py16 B-H table", .005)
    close("B-H first nonzero B", bh.iloc[1].B_T, .12, "py16 B-H table", 1e-9)
    exact("All selected core maxima below first nonzero B", bool((physics.core_B_max_T < .12).all()), True, "py16 design summary")

    sens14 = pd.read_csv(ROOT / "py14_output" / "sensitivity_results.csv").set_index("case")
    expected14 = {
        "l1_baseline": (.00091952, 0), "l1_feasible_1": (.00093006, 1.147),
        "l1_feasible_2": (.00092518, .616), "l1_feasible_3": (.00092584, .688),
        "l1_feasible_4": (.00091372, -.631),
    }
    baseline = sens14.loc["l1_baseline", "target_T"]
    for case, (target, percent) in expected14.items():
        close(f"EI sensitivity {case} target", sens14.loc[case, "target_T"], target, "py14 sensitivity", 5e-9)
        signed = (sens14.loc[case, "target_T"] / baseline - 1) * 100
        close(f"EI sensitivity {case} signed percent", signed, percent, "py14 sensitivity", .0005)
    exact("EI mesh element counts", sens14.loc[["mesh_hauto_3", "mesh_hauto_2", "mesh_hauto_1"], "mesh_element_count"].astype(int).tolist(), [2574, 7029, 26386], "py14 sensitivity")
    close("EI production vs finest mesh percent", (sens14.loc["mesh_hauto_3", "target_T"] / sens14.loc["mesh_hauto_1", "target_T"] - 1) * 100, -.069, "py14 sensitivity", .0005)
    close("EI 200 relative to 400 domain percent", (sens14.loc["domain_200_mm", "target_T"] / sens14.loc["domain_400_mm", "target_T"] - 1) * 100, -.227, "py14 sensitivity", .0005)
    close("Prose: EI increase when enlarging 200 to 400 mm percent", (sens14.loc["domain_400_mm", "target_T"] / sens14.loc["domain_200_mm", "target_T"] - 1) * 100, .228, "py14 sensitivity", .0005)

    sens17 = pd.read_csv(ROOT / "py17_output" / "sensitivity_results.csv")
    for label, mesh_pct, domain_pct in [("BOBYQA", .004, .239), ("COBYLA", .548, .179)]:
        sub = sens17.loc[sens17.label == label]
        prod = sub.loc[(sub.mesh_hauto == 3) & (sub.outer_domain_size_mm == 200), "target_T"].iloc[0]
        fine = sub.loc[(sub.mesh_hauto == 1) & (sub.outer_domain_size_mm == 200), "target_T"].iloc[0]
        large = sub.loc[(sub.mesh_hauto == 3) & (sub.outer_domain_size_mm == 400), "target_T"].iloc[0]
        close(f"{label} production-vs-fine absolute percent", abs(prod / fine - 1) * 100, mesh_pct, "py17 sensitivity", .0005)
        close(f"{label} production-vs-400 absolute percent", abs(prod / large - 1) * 100, domain_pct, "py17 sensitivity", .0005)
    # Ranking at production, finest, and largest domain.
    for setting, mesh, domain in [("production", 3, 200), ("finest", 1, 200), ("largest", 3, 400)]:
        values = sens17.loc[(sens17.mesh_hauto == mesh) & (sens17.outer_domain_size_mm == domain)].set_index("label").target_T
        # EI values come from py14 at the corresponding setting.
        ei_case = "mesh_hauto_1" if setting == "finest" else ("domain_400_mm" if setting == "largest" else "mesh_hauto_3")
        order = sorted({"BOBYQA": values["BOBYQA"], "COBYLA": values["COBYLA"], "EI-BO": sens14.loc[ei_case, "target_T"]}, key=lambda key: {"BOBYQA": values["BOBYQA"], "COBYLA": values["COBYLA"], "EI-BO": sens14.loc[ei_case, "target_T"]}[key], reverse=True)
        exact(f"Winner ordering at {setting}", order, ["BOBYQA", "COBYLA", "EI-BO"], "py14 and py17 sensitivity")


def audit_figures_and_text() -> None:
    text = PAPER.read_text(encoding="utf-8")
    import re
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    missing = [name for name in figures if not (PAPER.parent / name).exists()]
    exact("All manuscript graphics exist", missing, [], "conference_101719.tex")
    expected_sources = {
        "learning_curve_r2.png": ROOT / "py5_output" / "5.1" / "learning_curve_r2.png",
        "learning_curve_rmse.png": ROOT / "py5_output" / "5.1" / "learning_curve_rmse.png",
        "learning_curve_mae.png": ROOT / "py5_output" / "5.1" / "learning_curve_mae.png",
        "learning_curve_cov1.png": ROOT / "py5_output" / "5.1" / "learning_curve_cov1.png",
        "learning_curve_cov2.png": ROOT / "py5_output" / "5.1" / "learning_curve_cov2.png",
        "parity_plot.png": ROOT / "py4_output" / "parity_plot.png",
        "absolute_error_vs_predictive_std.png": ROOT / "py12_output" / "absolute_error_vs_predictive_std.png",
        "reliability_curve.png": ROOT / "py12_output" / "reliability_curve.png",
        "standardized_residual_qq.png": ROOT / "py12_output" / "standardized_residual_qq.png",
        "best_fem_response_vs_evaluations.png": ROOT / "py9_output" / "best_fem_response_vs_evaluations.png",
    }
    for name, source in expected_sources.items():
        exact(f"Figure {name} matches current output", sha256(PAPER.parent / "figure" / name), sha256(source), str(source.relative_to(PROJECT)))


def main() -> None:
    audit_contract()
    audit_surrogate()
    audit_optimization()
    audit_policy_and_physics()
    audit_figures_and_text()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT / "claim_audit.csv", index=False)
    failures = frame.loc[frame.status == "FAIL"].to_dict("records")
    summary = {
        "checks": int(len(frame)),
        "passed": int((frame.status == "PASS").sum()),
        "failed": int((frame.status == "FAIL").sum()),
        "failures": failures,
        "audit_scope": "current manuscript quantitative claims checked against current archived outputs; no COMSOL solve in this script",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
