"""Low-cost COMSOL sensitivity checks around a contract-valid best design.

The script does not re-optimize. It varies ``l1`` only within the current
axial-fit/manufacturing/mass constraints, or varies the automatic mesh level
and square outer-domain size while holding the design variables fixed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import py1_generate_parameter_table as py1
from py2_high_fidelity_evaluation_engine import (
    ComsolEvaluator,
    DESIGN_COLUMNS,
    set_model_parameters,
    scalar_probe_value,
)
from py11_time_initial_fem_samples import configure_stationary_point_evaluation


ROOT = Path(__file__).resolve().parent
BEST = ROOT / "py7_output" / "seed_45" / "best_design.csv"
OUTPUT = ROOT / "py14_output"
MESH_LEVELS = (3, 2, 1)
DOMAIN_SIZES_MM = (200.0, 300.0, 400.0)


def read_scalar(node: Any, name: str) -> float:
    for reader in ("getDouble", "getString"):
        try:
            return float(getattr(node, reader)(name))
        except Exception:
            continue
    return float("nan")


def normalized_design(sample: pd.Series) -> np.ndarray:
    physical = sample.loc[list(py1.PHYSICAL_COLUMNS)].to_numpy(dtype=float)
    return py1.physical_to_unit(physical)


def feasible_l1_values(base: pd.Series) -> list[float]:
    """Choose local l1 checks without crossing the revised feasible domain."""
    base_l1 = float(base["l1_mm"])
    clearance_l1 = float(base["h1_mm"]) + py1.COIL_BOTTOM_Z_MM
    raw = (
        base_l1,
        clearance_l1,
        0.5 * (base_l1 + clearance_l1),
        max(clearance_l1, base_l1 - 0.5),
        min(py1.PARAM_BOUNDS_MM["l1"][1], base_l1 + 0.5),
    )
    accepted: list[float] = []
    for value in raw:
        if any(abs(value - existing) < 1e-10 for existing in accepted):
            continue
        candidate = base.copy()
        candidate["l1_mm"] = value
        try:
            unit = normalized_design(candidate)
        except ValueError:
            continue
        if py1.is_unit_candidate_feasible(unit):
            accepted.append(float(value))
    if not accepted or abs(accepted[0] - base_l1) > 1e-10:
        raise ValueError(
            "The stored best design is infeasible under the current design-space "
            "contract. Regenerate py1, py2, and the optimization outputs first."
        )
    return accepted


def run_case(
    evaluator: ComsolEvaluator,
    base: pd.Series,
    *,
    case: str,
    l1_mm: float,
    mesh_level: int,
    domain_size_mm: float,
) -> dict[str, float | int | str]:
    if evaluator.model is None:
        raise RuntimeError("COMSOL evaluator is not open.")
    model = evaluator.model
    java = model.java
    component = java.component("comp1")
    square = component.geom("geom1").feature("sq1")
    size_feature = component.mesh("mesh1").feature("size")

    sample = base.copy()
    sample["sample_id"] = case
    sample["l1_mm"] = float(l1_mm)
    unit = normalized_design(sample)
    evaluated = py1.evaluate_unit_candidate(unit, sample_id=case)
    if not bool(evaluated["feasible"]):
        raise ValueError(f"Refusing infeasible sensitivity case: {case}")
    for column, value in zip(py1.NORMALIZED_COLUMNS, unit):
        sample[column] = float(value)
    set_model_parameters(model, sample)
    square.set("size", f"{domain_size_mm:.12g}")
    square.set("pos", ["0", f"{-domain_size_mm / 2.0:.12g}"])
    size_feature.set("hauto", str(int(mesh_level)))

    start = time.perf_counter()
    model.build()
    model.mesh()
    mesh_seconds = time.perf_counter() - start
    mesh = component.mesh("mesh1")
    try:
        element_count = int(mesh.getNumElem())
    except Exception:
        element_count = -1
    solve_start = time.perf_counter()
    model.solve()
    solve_seconds = time.perf_counter() - solve_start

    return {
        "case": case,
        "l1_mm": float(l1_mm),
        "mesh_hauto": int(mesh_level),
        "outer_domain_size_mm": float(domain_size_mm),
        "outer_r_max_mm": float(domain_size_mm),
        "outer_z_abs_max_mm": float(domain_size_mm / 2.0),
        "mesh_element_count": element_count,
        "mesh_nominal_hmax_mm": read_scalar(size_feature, "hmax"),
        "mesh_nominal_hmin_mm": read_scalar(size_feature, "hmin"),
        "target_T": scalar_probe_value(model),
        "mesh_seconds": mesh_seconds,
        "solve_seconds": solve_seconds,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_metadata_path = BEST.parent / "metadata.json"
    if not run_metadata_path.exists():
        raise FileNotFoundError(f"Missing optimization metadata: {run_metadata_path}")
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if run_metadata.get("design_space_fingerprint") != py1.design_space_fingerprint():
        raise ValueError(
            "The stored optimization result uses an obsolete design-space "
            "contract. Regenerate the experiment before sensitivity checks."
        )
    base = pd.read_csv(BEST).iloc[0].copy()
    l1_values_mm = feasible_l1_values(base)
    rows: list[dict[str, float | int | str]] = []

    with ComsolEvaluator() as evaluator:
        study = configure_stationary_point_evaluation(evaluator)

        for index, value in enumerate(l1_values_mm):
            rows.append(
                run_case(
                    evaluator,
                    base,
                    case="l1_baseline" if index == 0 else f"l1_feasible_{index}",
                    l1_mm=value,
                    mesh_level=3,
                    domain_size_mm=200.0,
                )
            )

        for level in MESH_LEVELS:
            rows.append(
                run_case(
                    evaluator,
                    base,
                    case=f"mesh_hauto_{level}",
                    l1_mm=float(base["l1_mm"]),
                    mesh_level=level,
                    domain_size_mm=200.0,
                )
            )

        for size in DOMAIN_SIZES_MM:
            rows.append(
                run_case(
                    evaluator,
                    base,
                    case=f"domain_{size:g}_mm",
                    l1_mm=float(base["l1_mm"]),
                    mesh_level=3,
                    domain_size_mm=size,
                )
            )

    frame = pd.DataFrame(rows)
    fine = float(
        frame.loc[frame["case"] == "mesh_hauto_1", "target_T"].iloc[0]
    )
    largest = float(
        frame.loc[frame["case"] == "domain_400_mm", "target_T"].iloc[0]
    )
    baseline = float(
        frame.loc[frame["case"] == "l1_baseline", "target_T"].iloc[0]
    )
    frame["relative_to_fine_mesh"] = np.abs(frame["target_T"] - fine) / abs(fine)
    frame["relative_to_largest_domain"] = (
        np.abs(frame["target_T"] - largest) / abs(largest)
    )
    frame["relative_to_original_l1"] = (
        np.abs(frame["target_T"] - baseline) / abs(baseline)
    )
    frame.to_csv(OUTPUT / "sensitivity_results.csv", index=False)

    metadata = {
        "purpose": "local sensitivity only; no re-optimization",
        "base_design": str(BEST),
        "fixed_design_columns": DESIGN_COLUMNS,
        "l1_values_mm": l1_values_mm,
        "mesh_levels": list(MESH_LEVELS),
        "domain_sizes_mm": list(DOMAIN_SIZES_MM),
        "study_configuration": study,
        "design_space_definition": py1.design_space_definition(),
        "design_space_fingerprint": py1.design_space_fingerprint(),
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
