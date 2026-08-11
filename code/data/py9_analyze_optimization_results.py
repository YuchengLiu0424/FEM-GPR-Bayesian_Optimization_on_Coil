"""Aggregate repeated shared-seed Python and COMSOL optimization runs.

Every included run must be explicitly complete and exhaust its own declared
FEM budget.  Methods may have different numbers of runs; inferential statistics
are emitted only when the available replication supports them.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import py1_generate_parameter_table as py1

SCRIPT_DIR = Path(__file__).resolve().parent
PY7_DIR = SCRIPT_DIR / "py7_output"
PY8_DIR = SCRIPT_DIR / "py8_output"
PY11_DIR = SCRIPT_DIR / "py11_output"
OUTPUT_DIR = SCRIPT_DIR / "py9_output"

METHOD_COLORS = {
    "bobyqa": "tab:blue",
    "cobyla": "tab:orange",
    "ego": "tab:green",
    "expected_improvement_bo": "tab:red",
    "nelder_mead": "tab:purple",
    "gp_mean": "tab:brown",
    "random_search": "tab:pink",
}

METHOD_DISPLAY_NAMES = {
    "bobyqa": "BOBYQA (COMSOL embedded)",
    "cobyla": "COBYLA (COMSOL embedded)",
    "ego": "EGO (COMSOL embedded)",
    "expected_improvement_bo": "EI-BO",
    "nelder_mead": "Nelder-Mead (COMSOL embedded)",
    "gp_mean": "GP mean",
    "random_search": "Random search",
}

METHOD_ORDER = [
    "bobyqa",
    "cobyla",
    "ego",
    "nelder_mead",
    "expected_improvement_bo",
    "gp_mean",
    "random_search",
]

RUN_LEVEL_COMSOL_COLORS = {
    "bobyqa": "C0",
    "cobyla": "C1",
    "ego": "C2",
    "nelder_mead": "C8",
}

BO_SEED_COLORS = {
    42: "C3",
    43: "C4",
    44: "C5",
    45: "C6",
    46: "C7",
}


def method_color(method: str) -> str:
    """Return one stable color for a method across every py9 figure."""
    return METHOD_COLORS.get(method, "tab:gray")


def method_display_name(method: str) -> str:
    """Return the publication-facing method label used in py9 figures."""
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", " "))


def ordered_methods(values: pd.Series) -> list[str]:
    """Keep COMSOL-embedded methods together before Python methods."""
    available = set(values.astype(str))
    return [
        *[method for method in METHOD_ORDER if method in available],
        *sorted(available - set(METHOD_ORDER)),
    ]


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9, "axes.grid": True, "grid.alpha": .28})


def collect(
    mode: str,
    required_methods: list[str],
    required_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    designs = []
    roots = [("expected_improvement_bo", PY7_DIR)]
    if PY8_DIR.exists():
        roots.extend(
            (path.name, path)
            for path in sorted(PY8_DIR.iterdir())
            if path.is_dir()
        )
    protocol_rows = []
    initial_ids: dict[tuple[str, str], tuple[str, ...]] = {}
    for method, root in roots:
        if method not in required_methods:
            continue
        if not root.exists():
            continue
        paths = list(root.glob("seed_*/iteration_history.csv"))
        for path in sorted(set(paths)):
            run_id = path.parent.name
            metadata_path = path.parent / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata for {path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("design_space_fingerprint") != py1.design_space_fingerprint():
                raise ValueError(
                    "Refusing a run created under an obsolete design-space "
                    f"contract: {metadata_path}"
                )
            if metadata.get("mode") != mode:
                raise ValueError(f"Mode mismatch in {metadata_path}")
            if metadata.get("completed") is not True:
                raise ValueError(f"Refusing to analyze incomplete run: {metadata_path}")
            actual_calls = metadata.get("actual_fem_calls")
            budget_calls = metadata.get(
                "budget_accounted_fem_calls", actual_calls
            )
            total_budget = metadata.get("total_budget")
            if not isinstance(actual_calls, int):
                raise ValueError(f"Missing actual_fem_calls in {metadata_path}")
            if not isinstance(total_budget, int):
                raise ValueError(f"Missing total_budget in {metadata_path}")
            if not isinstance(budget_calls, int):
                raise ValueError(
                    f"Missing budget-accounted FEM calls in {metadata_path}"
                )
            if budget_calls > total_budget:
                raise ValueError(
                    f"Run exceeds its declared budget in {metadata_path}: "
                    f"budget_accounted_fem_calls={budget_calls}, "
                    f"raw_actual_fem_calls={actual_calls}, "
                    f"total_budget={total_budget}"
                )
            if (
                method == "expected_improvement_bo"
                and budget_calls != total_budget
            ):
                raise ValueError(
                    f"Refusing an unfinished py7 budget in {metadata_path}: "
                    f"budget_accounted_fem_calls={budget_calls}, "
                    f"total_budget={total_budget}"
                )
            seed = metadata.get("seed")
            protocol_rows.append((
                method, run_id, seed, metadata.get("initial_size"), total_budget,
                metadata.get("held_out_validation_size"), metadata.get("validation_seed"),
                metadata.get("candidate_pool_size"), metadata.get("min_distance"),
                budget_calls,
                actual_calls,
            ))
            observations = pd.read_csv(path.parent / "observations.csv")
            initial_size = int(metadata.get("initial_size") or 0)
            if initial_size > 0 and "sample_id" in observations:
                initial_ids[(method, run_id)] = tuple(
                    observations["sample_id"].iloc[:initial_size].astype(str)
                )
            summary_path = path.parent / "run_summary.json"
            design_path = path.parent / "best_design.csv"
            if not summary_path.exists() or not design_path.exists():
                raise FileNotFoundError(f"Missing run_summary.json or best_design.csv in {path.parent}")
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_summary.update(
                {
                    "method": method,
                    "run_id": run_id,
                    "seed": seed,
                    "budget_accounted_fem_calls": budget_calls,
                }
            )
            summaries.append(run_summary)
            design = pd.read_csv(design_path)
            design["method"] = method
            design["run_id"] = run_id
            design["seed"] = seed
            designs.append(design)
            frame = pd.read_csv(path)
            history_required = [
                "iteration",
                "cumulative_fem_calls",
                "best_so_far",
                "cumulative_wall_seconds",
            ]
            missing_history = [
                column for column in history_required if column not in frame
            ]
            if missing_history:
                raise ValueError(
                    f"Missing iteration-history columns in {path}: {missing_history}"
                )
            row_identity = ["iteration", "cumulative_fem_calls", "best_so_far"]
            # Old checkpoint files may contain trailing timing-summary rows
            # without an iteration identity. They are not optimization steps.
            fully_blank = frame[row_identity].isna().all(axis=1)
            frame = frame.loc[~fully_blank].copy()
            if frame[history_required].isna().any().any():
                raise ValueError(f"Partially missing iteration-history row in {path}")
            if (
                frame.empty
                or int(frame["cumulative_fem_calls"].max()) != budget_calls
            ):
                raise ValueError(
                    f"History in {path} does not end at "
                    f"budget_accounted_fem_calls={budget_calls}."
                )
            if not frame.empty:
                frame["method"] = method
                frame["run_id"] = run_id
                frame["seed"] = seed
                rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No completed {mode} py7/py8 iteration_history.csv files found.")
    protocol = pd.DataFrame(protocol_rows, columns=[
        "method", "run_id", "seed", "initial_size", "total_budget",
        "validation_size", "validation_seed", "candidate_pool_size",
        "min_distance", "budget_accounted_fem_calls", "actual_fem_calls",
    ])
    present = set(protocol["method"])
    missing = sorted(set(required_methods) - present)
    if missing:
        raise FileNotFoundError(
            "Required optimization methods are missing from completed outputs: "
            + ", ".join(missing)
        )
    expected_seed_set = set(required_seeds)
    for method in required_methods:
        method_seeds = {
            int(value)
            for value in protocol.loc[
                protocol["method"] == method, "seed"
            ].dropna()
        }
        if method_seeds != expected_seed_set:
            raise ValueError(
                f"Method {method!r} has seeds {sorted(method_seeds)}; "
                f"expected {sorted(expected_seed_set)}."
            )

    # Fair paired inference is allowed only for shared seeds whose initial
    # samples are actually identical.
    for seed, seed_block in protocol.dropna(subset=["seed"]).groupby("seed"):
        if seed_block["method"].nunique() < 2:
            continue
        sets = {
            initial_ids[(row.method, row.run_id)]
            for row in seed_block.itertuples()
            if (row.method, row.run_id) in initial_ids
        }
        if len(sets) > 1:
            raise ValueError(
                f"Methods sharing seed {seed!r} do not share identical initial samples."
            )
    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(summaries),
        pd.concat(designs, ignore_index=True),
        protocol,
    )


def load_initial_fem_timings() -> dict[int, dict[str, float]]:
    """Load completed py11 pure-FEM timing summaries, keyed by BO seed."""
    timings: dict[int, dict[str, float]] = {}
    if not PY11_DIR.exists():
        return timings
    for path in sorted(PY11_DIR.glob("seed_*/timing_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("completed") is not True:
            raise ValueError(f"Refusing incomplete py11 timing summary: {path}")
        if summary.get("valid_for_fem_timing") is not True:
            raise ValueError(f"py11 summary is not valid for FEM timing: {path}")
        seed = summary.get("seed")
        n_samples = summary.get("n_samples")
        fem_seconds = summary.get("estimated_initial_25_fem_seconds")
        if not isinstance(seed, int) or not isinstance(n_samples, int):
            raise ValueError(f"Invalid seed or n_samples in {path}")
        if not isinstance(fem_seconds, (int, float)) or not np.isfinite(fem_seconds):
            raise ValueError(f"Invalid estimated_initial_25_fem_seconds in {path}")
        timings[seed] = {
            "n_samples": int(n_samples),
            "estimated_initial_samples": 25,
            "pure_fem_seconds": float(fem_seconds),
            "source": str(path),
        }
    return timings


def add_initial_fem_time(
    history: pd.DataFrame,
    runs: pd.DataFrame,
    reference_seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the shared initial-design FEM cost to every seeded method."""
    timing = load_initial_fem_timings()
    if (runs["initial_size"].fillna(0).astype(int) > 0).any() and not timing:
        raise FileNotFoundError(
            f"No completed py11 timing summaries were found in {PY11_DIR}."
        )
    if reference_seed is not None and reference_seed not in timing:
        raise FileNotFoundError(
            f"Requested py11 reference seed {reference_seed} is unavailable."
        )

    enriched = runs.copy()
    initial_costs: list[float] = []
    sources: list[str] = []
    exact_flags: list[bool] = []
    for row in enriched.itertuples():
        initial_size = int(row.initial_size or 0)
        if initial_size == 0:
            initial_costs.append(0.0)
            sources.append("no_shared_initial_design")
            exact_flags.append(True)
            continue

        run_seed = int(row.seed) if pd.notna(row.seed) else None
        selected_seed: int | None = None
        exact = False
        if run_seed is not None and run_seed in timing:
            selected_seed = run_seed
            exact = True
        elif reference_seed is not None:
            selected_seed = reference_seed
        if selected_seed is None:
            raise FileNotFoundError(
                f"No py11 initial-FEM timing exists for {row.method}/{row.run_id}. "
                "Run py11 for that seed or pass --initial-fem-reference-seed."
            )

        selected = timing[selected_seed]
        if (
            selected["n_samples"] != 50
            or selected["estimated_initial_samples"] != initial_size
        ):
            raise ValueError(
                f"py11 seed {selected_seed} must time 50 FEM observations and "
                f"estimate initial_size={initial_size} by division."
            )
        initial_costs.append(float(selected["pure_fem_seconds"]))
        sources.append(str(selected["source"]))
        exact_flags.append(exact)

    enriched["online_optimization_wall_seconds"] = enriched[
        "optimization_wall_seconds"
    ].astype(float)
    enriched["initial_fem_seconds"] = initial_costs
    enriched["initial_fem_timing_source"] = sources
    enriched["initial_fem_timing_exact_for_seed"] = exact_flags
    enriched["end_to_end_wall_seconds"] = (
        enriched["online_optimization_wall_seconds"]
        + enriched["initial_fem_seconds"]
    )

    cost_lookup = enriched.set_index(["method", "run_id"])[
        "initial_fem_seconds"
    ]
    history_enriched = history.copy()
    history_enriched["initial_fem_seconds"] = [
        float(cost_lookup.loc[(method, run_id)])
        for method, run_id in zip(
            history_enriched["method"], history_enriched["run_id"]
        )
    ]
    history_enriched["comparison_cumulative_wall_seconds"] = (
        history_enriched["cumulative_wall_seconds"].astype(float)
        + history_enriched["initial_fem_seconds"]
    )
    return history_enriched, enriched


def curve(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for method in ordered_methods(data["method"]):
        group = data.loc[data["method"] == method]
        stat = group.groupby("cumulative_fem_calls")["best_so_far"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        stat["se"] = stat["std"] / np.sqrt(stat["count"])
        stat["ci95"] = stats.t.ppf(0.975, stat["count"] - 1) * stat["se"]
        color = method_color(method)
        label = method_display_name(method)
        if method == "expected_improvement_bo":
            n_runs = group["run_id"].nunique()
            label = f"EI-BO mean ± 95% CI ({n_runs} seeds)"
        ax.plot(
            stat["cumulative_fem_calls"],
            stat["mean"],
            marker="o",
            markersize=1.8,
            markeredgewidth=0,
            markevery=max(1, len(stat) // 14),
            linewidth=1.25,
            color=color,
            label=label,
        )
        valid = stat["count"] >= 2
        if valid.any():
            ax.fill_between(
                stat.loc[valid, "cumulative_fem_calls"],
                stat.loc[valid, "mean"] - stat.loc[valid, "ci95"],
                stat.loc[valid, "mean"] + stat.loc[valid, "ci95"],
                color=color,
                alpha=.16,
            )
    ax.set_xlabel("Cumulative accounted FEM observations")
    ax.set_ylabel("Best observed FEM response (T)")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def final_boxplot(data: pd.DataFrame, output: Path) -> None:
    final = data.sort_values("cumulative_fem_calls").groupby(
        ["method", "run_id"]
    ).tail(1)
    methods = ordered_methods(final["method"])
    values = [final.loc[final["method"] == method, "best_so_far"].to_numpy() for method in methods]
    labels = []
    for method in methods:
        label = method_display_name(method)
        if method == "expected_improvement_bo":
            n_runs = final.loc[final["method"] == method, "run_id"].nunique()
            label = f"EI-BO ({n_runs} seeds)"
        labels.append(label)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
    )
    ax.set_ylabel("Final best observed FEM response (T)")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def improvement_plot(data: pd.DataFrame, output: Path) -> None:
    valid_runs = []
    for _, run in data.groupby(["method", "run_id"]):
        if (run["iteration"] == 0).any():
            valid_runs.append(run)
    if not valid_runs:
        return
    frame = pd.concat(valid_runs, ignore_index=True)
    initial_best = frame["best_so_far"].where(frame["iteration"] == 0).groupby(
        [frame["method"], frame["run_id"]]
    ).transform("first")
    if initial_best.isna().any():
        raise ValueError("A non-endpoint run is missing its iteration-0 initial-design row.")
    frame["improvement_over_initial"] = frame["best_so_far"] - initial_best
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for method in ordered_methods(frame["method"]):
        group = frame.loc[frame["method"] == method]
        stat = group.groupby("cumulative_fem_calls")["improvement_over_initial"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        stat["se"] = stat["std"] / np.sqrt(stat["count"])
        stat["ci95"] = stats.t.ppf(0.975, stat["count"] - 1) * stat["se"]
        color = method_color(method)
        label = method_display_name(method)
        if method == "expected_improvement_bo":
            n_runs = group["run_id"].nunique()
            label = f"EI-BO mean ± 95% CI ({n_runs} seeds)"
        ax.plot(
            stat["cumulative_fem_calls"],
            stat["mean"],
            marker="o",
            markersize=1.8,
            markeredgewidth=0,
            markevery=max(1, len(stat) // 14),
            linewidth=1.25,
            color=color,
            label=label,
        )
        valid = stat["count"] >= 2
        if valid.any():
            ax.fill_between(
                stat.loc[valid, "cumulative_fem_calls"],
                stat.loc[valid, "mean"] - stat.loc[valid, "ci95"],
                stat.loc[valid, "mean"] + stat.loc[valid, "ci95"],
                color=color,
                alpha=.16,
            )
    ax.set_xlabel("Cumulative accounted FEM observations"); ax.set_ylabel("Improvement over initial best (T)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def response_time_plot(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    # Preserve the original Matplotlib color-cycle assignment for this
    # run-level plot: three external methods, five EI-BO seeds, then
    # Nelder-Mead.  Unlike aggregate figures, every BO seed is identified.
    for method in ordered_methods(data["method"]):
        group = data.loc[data["method"] == method]
        if group.empty:
            continue
        for run_id, run in group.groupby("run_id"):
            run = run.sort_values("comparison_cumulative_wall_seconds")
            if method == "expected_improvement_bo":
                seed_values = run["seed"].dropna().unique()
                if len(seed_values) != 1:
                    raise ValueError(
                        f"EI-BO run {run_id} must contain exactly one seed."
                    )
                seed = int(seed_values[0])
                color = BO_SEED_COLORS.get(seed, "C9")
                label = f"EI-BO seed {seed}"
            else:
                color = RUN_LEVEL_COMSOL_COLORS.get(
                    method, method_color(method)
                )
                label = method_display_name(method)
            endpoint_only = bool(
                run.get("endpoint_only", pd.Series(False, index=run.index))
                .fillna(False)
                .astype(bool)
                .all()
            )
            if endpoint_only or len(run) == 1:
                ax.scatter(
                    run["comparison_cumulative_wall_seconds"],
                    run["best_so_far"],
                    color=color,
                    s=18,
                    alpha=0.9,
                    label=label,
                    zorder=3,
                )
            else:
                ax.step(
                    run["comparison_cumulative_wall_seconds"],
                    run["best_so_far"],
                    where="post",
                    color=color,
                    alpha=0.32,
                    linewidth=1.5,
                    label=label,
                )
    ax.set_xlabel("Comparison cumulative wall time (s)")
    ax.set_ylabel("Best observed FEM response (T)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def final_time_scatter(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    for method in ordered_methods(summary["method"]):
        group = summary.loc[summary["method"] == method]
        if method == "expected_improvement_bo":
            for row in group.sort_values("seed").itertuples():
                if pd.isna(row.seed):
                    raise ValueError(
                        f"EI-BO run {row.run_id} is missing its seed."
                    )
                seed = int(row.seed)
                ax.scatter(
                    row.end_to_end_wall_seconds,
                    row.final_best_response,
                    color=BO_SEED_COLORS.get(seed, "C9"),
                    s=18,
                    label=f"EI-BO seed {seed}",
                )
        else:
            ax.scatter(
                group["end_to_end_wall_seconds"],
                group["final_best_response"],
                color=RUN_LEVEL_COMSOL_COLORS.get(
                    method, method_color(method)
                ),
                s=18,
                label=method_display_name(method),
            )
    ax.set_xlabel("Comparison end-to-end wall time (s)")
    ax.set_ylabel("Final best FEM response (T)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def add_efficiency_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    # This quantity is defined within each shared-seed run.
    result["relative_improvement"] = (
        result["final_best_response"] - result["initial_best_response"]
    ) / result["initial_best_response"].abs().clip(lower=1e-15)
    result["relative_improvement_per_hour"] = (
        result["relative_improvement"] * 3600.0
        / result["end_to_end_wall_seconds"].clip(lower=1e-12)
    )
    return result


def add_history_metrics(
    summary: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Add fixed-budget, threshold-free trajectory metrics for each run."""
    records = []
    for (method, run_id), run in history.groupby(["method", "run_id"]):
        run = run.sort_values("cumulative_fem_calls")
        x = run["cumulative_fem_calls"].to_numpy(float)
        y = run["best_so_far"].to_numpy(float)
        endpoint_only = bool(
            run.get("endpoint_only", pd.Series(False, index=run.index))
            .fillna(False)
            .astype(bool)
            .all()
        )
        if endpoint_only:
            records.append({
                "method": method,
                "run_id": run_id,
                "auc_best_response": np.nan,
                "auc_improvement": np.nan,
                "mean_best_response_over_budget": np.nan,
                "mean_improvement_over_budget": np.nan,
            })
            continue
        initial = float(y[0])
        span = float(x[-1] - x[0])
        auc_response = float(np.trapezoid(y, x)) if len(x) > 1 else 0.0
        auc_improvement = (
            float(np.trapezoid(y - initial, x)) if len(x) > 1 else 0.0
        )
        records.append({
            "method": method,
            "run_id": run_id,
            "auc_best_response": auc_response,
            "auc_improvement": auc_improvement,
            "mean_best_response_over_budget": (
                auc_response / span if span > 0 else float(y[-1])
            ),
            "mean_improvement_over_budget": (
                auc_improvement / span if span > 0 else 0.0
            ),
        })
    return summary.merge(pd.DataFrame(records), on=["method", "run_id"], how="left")


def _mean_ci95(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return np.nan, np.nan
    half = float(
        stats.t.ppf(0.975, len(clean) - 1)
        * clean.std(ddof=1)
        / np.sqrt(len(clean))
    )
    return float(clean.mean() - half), float(clean.mean() + half)


def method_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method, group in runs.groupby("method"):
        response_ci = _mean_ci95(group["final_best_response"])
        improvement_ci = _mean_ci95(group["improvement"])
        auc_ci = _mean_ci95(group["mean_improvement_over_budget"])
        records.append({
            "method": method,
            "runs": int(len(group)),
            "final_response_mean": group["final_best_response"].mean(),
            "final_response_std": group["final_best_response"].std(ddof=1),
            "final_response_median": group["final_best_response"].median(),
            "final_response_ci95_low": response_ci[0],
            "final_response_ci95_high": response_ci[1],
            "improvement_mean": group["improvement"].mean(),
            "improvement_std": group["improvement"].std(ddof=1),
            "improvement_ci95_low": improvement_ci[0],
            "improvement_ci95_high": improvement_ci[1],
            "mean_improvement_auc": group["mean_improvement_over_budget"].mean(),
            "mean_improvement_auc_ci95_low": auc_ci[0],
            "mean_improvement_auc_ci95_high": auc_ci[1],
            "initial_fem_time_mean_seconds": group["initial_fem_seconds"].mean(),
            "online_wall_time_mean_seconds": group[
                "online_optimization_wall_seconds"
            ].mean(),
            "end_to_end_wall_time_mean_seconds": group[
                "end_to_end_wall_seconds"
            ].mean(),
            "end_to_end_wall_time_std_seconds": group[
                "end_to_end_wall_seconds"
            ].std(ddof=1),
            "fem_time_mean_seconds": group["total_fem_seconds"].mean(),
            "gp_time_mean_seconds": group["total_gp_fit_seconds"].mean(),
            "acquisition_time_mean_seconds": group["total_acquisition_seconds"].mean(),
            "relative_improvement_per_hour_mean": group[
                "relative_improvement_per_hour"
            ].mean(),
            "actual_fem_calls_mean": group["actual_fem_calls"].mean(),
            "budget_accounted_fem_calls_mean": group[
                "budget_accounted_fem_calls"
            ].mean(),
        })
    return pd.DataFrame(records)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_df = len(a) + len(b) - 2
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / pooled_df
    )
    if pooled <= 1e-15:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * (len(a) + len(b)) - 9.0)
    return float(correction * (np.mean(a) - np.mean(b)) / pooled)


def statistical_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    """Compare methods without inventing inference for an n=1 baseline."""
    records = []
    for method_a, method_b in combinations(sorted(runs["method"].unique()), 2):
        a = runs.loc[runs["method"] == method_a].copy()
        b = runs.loc[runs["method"] == method_b].copy()
        common = a.dropna(subset=["seed"]).merge(
            b.dropna(subset=["seed"]),
            on="seed",
            suffixes=("_a", "_b"),
        )
        record = {
            "method_a": method_a,
            "method_b": method_b,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_final_difference_a_minus_b": (
                a["final_best_response"].mean() - b["final_best_response"].mean()
            ),
            "test_design": "descriptive_only",
            "test_statistic": np.nan,
            "p_value": np.nan,
            "effect_size_name": "not_available",
            "effect_size": np.nan,
            "common_seed_pairs": int(len(common)),
            "inference_note": (
                "No inferential test: each method needs at least two independent "
                "runs, or at least two valid common-seed pairs."
            ),
        }
        if len(common) >= 2:
            x = common["final_best_response_a"].to_numpy(float)
            y = common["final_best_response_b"].to_numpy(float)
            difference = x - y
            test = stats.ttest_rel(x, y)
            record.update({
                "test_design": "paired_common_seed",
                "test_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "effect_size_name": "cohen_dz",
                "effect_size": (
                    float(difference.mean() / difference.std(ddof=1))
                    if difference.std(ddof=1) > 1e-15 else 0.0
                ),
                "inference_note": "Paired t-test on common seeds.",
            })
        elif len(a) >= 2 and len(b) >= 2:
            x = a["final_best_response"].to_numpy(float)
            y = b["final_best_response"].to_numpy(float)
            test = stats.ttest_ind(x, y, equal_var=False)
            record.update({
                "test_design": "welch_independent",
                "test_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "effect_size_name": "hedges_g",
                "effect_size": _hedges_g(x, y),
                "inference_note": (
                    "Welch t-test; runs are treated as independent because fewer "
                    "than two common seeds were available."
                ),
            })
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BO and baseline outputs.")
    parser.add_argument("--mode", choices=("replay", "online"), default="replay")
    parser.add_argument(
        "--required-methods",
        nargs="+",
        required=True,
        help=(
            "Exact methods that must be present and complete. Example: "
            "expected_improvement_bo nelder_mead particle_swarm"
        ),
    )
    parser.add_argument(
        "--required-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help=(
            "Exact shared initial-design seeds required for every method "
            "(default: 42 43 44 45 46)."
        ),
    )
    parser.add_argument(
        "--initial-fem-reference-seed",
        type=int,
        default=42,
        help=(
            "py11 seed used to estimate the initial 25-point pure-FEM cost for "
            "py7 seeds without their own py11 timing (default: 42)."
        ),
    )
    args = parser.parse_args()
    output = OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True); style()
    data, runs, designs, protocol = collect(
        args.mode,
        args.required_methods,
        args.required_seeds,
    )
    data, runs = add_initial_fem_time(
        data,
        runs,
        reference_seed=args.initial_fem_reference_seed,
    )
    data.to_csv(output / "optimization_history.csv", index=False)
    designs.to_csv(output / "best_designs.csv", index=False)
    protocol = protocol.merge(
        runs[[
            "method",
            "run_id",
            "initial_fem_seconds",
            "initial_fem_timing_source",
            "initial_fem_timing_exact_for_seed",
            "online_optimization_wall_seconds",
            "end_to_end_wall_seconds",
        ]],
        on=["method", "run_id"],
        how="left",
    )
    protocol.to_csv(output / "run_protocol.csv", index=False)
    efficiency = add_efficiency_metrics(runs)
    efficiency = add_history_metrics(efficiency, data)
    efficiency.to_csv(output / "optimization_efficiency.csv", index=False)
    summary = method_statistics(efficiency)
    summary.to_csv(output / "optimization_summary.csv", index=False)
    comparisons = statistical_comparisons(efficiency)
    comparisons.to_csv(output / "statistical_comparisons.csv", index=False)
    curve(data, output / "best_fem_response_vs_evaluations.png")
    final_boxplot(data, output / "final_response_boxplot.png")
    improvement_plot(data, output / "improvement_over_initial_design.png")
    response_time_plot(data, output / "best_response_vs_wall_time.png")
    final_time_scatter(efficiency, output / "final_response_vs_wall_time.png")
    with (output / "analysis_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "required_methods": args.required_methods,
            "required_seeds": args.required_seeds,
            "methods": sorted(data.method.unique()),
            "runs_per_method": protocol.groupby("method").size().to_dict(),
            "optimization_protocol": (
                "fixed FEM-evaluation budget with no predeclared response target "
                "or target-based stopping rule"
            ),
            "time_comparison_policy": (
                "every method: the same py11-estimated 25-point initial-FEM "
                "cost for its shared seed plus its recorded continuation wall time"
            ),
            "py11_initial_fem_reference_seed": args.initial_fem_reference_seed,
            "confidence_interval": "two-sided 95% Student-t interval; unavailable for n=1",
            "inference_policy": (
                "paired t-test for >=2 common seeds; otherwise Welch t-test "
                "only when both methods have >=2 runs; n=1 comparisons are descriptive"
            ),
            "n_rows": len(data),
        }, f, indent=2)
    print(f"Saved analysis to: {output}")


if __name__ == "__main__": main()
