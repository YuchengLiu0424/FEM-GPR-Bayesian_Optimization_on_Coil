"""Physics-focused validation of the three selected optimization designs.

This script performs fresh stationary solves without saving the source MPH.
It extracts the ROI response, full solution-node field values for plotting,
core-domain magnetic-flux-density summaries, and a fixed-total-current check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from py2_high_fidelity_evaluation_engine import (
    ComsolEvaluator,
    MODEL_FILE,
    set_model_parameters,
    scalar_probe_value,
)


ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "py10_output" / "verification_results.csv"
OUTPUT = ROOT / "py16_output"
FIGURE_DIR = ROOT.parents[1] / "paper" / "figure"
METHODS = {
    "expected_improvement_bo": "EI-BO",
    "bobyqa": "BOBYQA",
    "cobyla": "COBYLA",
}
CORE_DOMAIN = 2
ORIGINAL_J = 1.0e6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_designs() -> pd.DataFrame:
    frame = pd.read_csv(VERIFY)
    frame = frame.loc[
        (frame["candidate_type"] == "best_observed")
        & frame["method"].isin(METHODS)
    ].copy()
    rows = []
    for method, label in METHODS.items():
        subset = frame.loc[frame["method"] == method]
        if subset.empty:
            raise ValueError(f"Missing selected design for {method}.")
        best = subset.loc[subset["verified_fem_target"].astype(float).idxmax()].copy()
        best["label"] = label
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def numerical_scalar(model: Any, kind: str, expression: str, domain: int) -> float:
    numericals = model.java.result().numerical()
    tag = f"v{len(list(numericals.tags())) + 1:03d}"
    node = numericals.create(tag, kind)
    try:
        node.selection().set(domain)
        node.set("expr", expression)
        raw = node.getReal()
        values = np.asarray([list(row) for row in raw], dtype=float).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise RuntimeError(f"Unexpected {kind} result for {expression}: {values}")
        return float(values[0])
    finally:
        numericals.remove(tag)


def core_statistics(model: Any) -> dict[str, float]:
    volume = numerical_scalar(model, "IntSurface", "1", CORE_DOMAIN)
    b_integral = numerical_scalar(model, "IntSurface", "mf.normB", CORE_DOMAIN)
    h_integral = numerical_scalar(model, "IntSurface", "mf.normH", CORE_DOMAIN)
    return {
        "core_B_mean_T": b_integral / volume,
        "core_B_max_T": numerical_scalar(model, "MaxSurface", "mf.normB", CORE_DOMAIN),
        "core_H_mean_Am": h_integral / volume,
        "core_H_max_Am": numerical_scalar(model, "MaxSurface", "mf.normH", CORE_DOMAIN),
    }


def field_nodes(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = model.evaluate(["r", "z", "mf.normB"], unit=["m", "m", "T"])
    r, z, b = (np.asarray(item, dtype=float).reshape(-1) for item in values)
    valid = np.isfinite(r) & np.isfinite(z) & np.isfinite(b)
    return r[valid], z[valid], b[valid]


def roi_profile(r: np.ndarray, z: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    radii_mm = np.linspace(0.0, 5.0, 101)
    points = np.column_stack((r * 1e3, z * 1e3))
    query = np.column_stack((radii_mm, np.full_like(radii_mm, -10.0)))
    profile = griddata(points, b, query, method="linear")
    if np.isnan(profile).any():
        nearest = griddata(points, b, query, method="nearest")
        profile = np.where(np.isnan(profile), nearest, profile)
    return pd.DataFrame({"r_mm": radii_mm, "B_T": profile})


def bh_table(model: Any) -> pd.DataFrame:
    raw = (
        model.java.component("comp1").material("mat3")
        .propertyGroup("BHCurve").func("BH").getStringMatrix("table")
    )
    array = np.asarray([[float(str(value)) for value in row] for row in raw], dtype=float)
    if array.shape[1] != 2:
        raise RuntimeError(f"Unexpected B-H table shape: {array.shape}")
    return pd.DataFrame({"H_Am": array[:, 0], "B_T": array[:, 1]})


def solve_design(evaluator: ComsolEvaluator, row: pd.Series, common_current_A: float):
    model = evaluator.model
    if model is None:
        raise RuntimeError("COMSOL evaluator is not open.")
    sample = row.copy()
    sample["sample_id"] = f"physics_{row['label']}"
    set_model_parameters(model, sample)
    model.parameter("Japply", f"{ORIGINAL_J:.12g}[A/m^2]")
    model.build()
    model.mesh()
    model.solve()
    original_target = scalar_probe_value(model)
    stats = core_statistics(model)
    r, z, b = field_nodes(model)
    profile = roi_profile(r, z, b)

    area_m2 = float(row["w1_mm"]) * float(row["h1_mm"]) * 1e-6
    original_current = ORIGINAL_J * area_m2
    fixed_j = common_current_A / area_m2
    model.parameter("Japply", f"{fixed_j:.12g}[A/m^2]")
    model.solve()
    fixed_target = scalar_probe_value(model)
    return (
        {
            "method": row["method"],
            "label": row["label"],
            "seed": int(row["seed"]),
            "original_current_A": original_current,
            "original_J_Am2": ORIGINAL_J,
            "original_target_T": original_target,
            "fixed_current_A": common_current_A,
            "fixed_J_Am2": fixed_j,
            "fixed_current_target_T": fixed_target,
            "fixed_vs_original_relative": (fixed_target - original_target) / original_target,
            **stats,
        },
        profile,
        (r, z, b),
    )


def plot_results(fields, profiles, summary: pd.DataFrame, bh: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    vmax = max(np.nanpercentile(field[2] * 1e3, 99.5) for field in fields.values())
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), constrained_layout=True)
    image = None
    for ax, (label, (r, z, b)) in zip(axes, fields.items()):
        x = r * 1e3; y = z * 1e3; values = b * 1e3
        mask = (x >= 0) & (x <= 32) & (y >= -18) & (y <= 35)
        gx, gy = np.meshgrid(np.linspace(0, 32, 220), np.linspace(-18, 35, 260))
        gz = griddata(np.column_stack((x[mask], y[mask])), values[mask], (gx, gy), method="linear")
        image = ax.contourf(gx, gy, gz, levels=np.linspace(0, vmax, 31), cmap="viridis", extend="max")
        ax.plot([0, 5], [-10, -10], color="red", lw=2)
        ax.set_title(label); ax.set_xlabel("r (mm)"); ax.set_aspect("equal")
    axes[0].set_ylabel("z (mm)")
    fig.colorbar(image, ax=axes, label=r"$|B|$ (mT)", shrink=0.88)
    fig.savefig(FIGURE_DIR / "selected_design_field_maps.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    for label, profile in profiles.items():
        ax.plot(profile["r_mm"], profile["B_T"] * 1e3, label=label)
    ax.set(xlabel="ROI radius r (mm)", ylabel=r"$|B(r,z=-10\,\mathrm{mm})|$ (mT)")
    ax.grid(alpha=0.25); ax.legend()
    fig.savefig(FIGURE_DIR / "selected_design_roi_profiles.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    ax.plot(bh["H_Am"], bh["B_T"], color="black", lw=1.5, label="Input B-H curve")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (_, row) in zip(colors, summary.iterrows()):
        ax.axhspan(row["core_B_mean_T"], row["core_B_max_T"], color=color, alpha=0.16)
        ax.axhline(row["core_B_mean_T"], color=color, lw=1.1, label=f"{row['label']} mean-to-max B")
    ax.set(xlabel="Magnetic field strength H (A/m)", ylabel="Magnetic flux density B (T)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.savefig(FIGURE_DIR / "bh_curve_operating_ranges.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(MODEL_FILE)
    designs = selected_designs()
    currents = ORIGINAL_J * designs["w1_mm"].astype(float) * designs["h1_mm"].astype(float) * 1e-6
    common_current = float(currents.min())
    summaries = []
    profiles = {}
    fields = {}
    with ComsolEvaluator() as evaluator:
        bh = bh_table(evaluator.model)
        for _, row in designs.iterrows():
            print(f"Solving {row['label']} at original and common total current...")
            summary, profile, field = solve_design(evaluator, row, common_current)
            summaries.append(summary)
            profile.insert(0, "label", row["label"])
            profiles[row["label"]] = profile
            fields[row["label"]] = field
    source_hash_after = sha256_file(MODEL_FILE)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source MPH hash changed during validation.")
    summary = pd.DataFrame(summaries)
    summary["fixed_current_rank"] = summary["fixed_current_target_T"].rank(method="min", ascending=False).astype(int)
    summary.to_csv(OUTPUT / "design_summary.csv", index=False)
    pd.concat(profiles.values(), ignore_index=True).to_csv(OUTPUT / "roi_profiles.csv", index=False)
    bh.to_csv(OUTPUT / "bh_curve.csv", index=False)
    plot_results(fields, profiles, summary, bh)
    metadata = {
        "purpose": "selected-design field, core B-range, ROI-profile, and fixed-current validation",
        "source_model": str(MODEL_FILE),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_model_saved": False,
        "common_total_current_A": common_current,
        "fixed_current_choice": "minimum original total current among the three selected designs",
        "fresh_stationary_solves": 6,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
