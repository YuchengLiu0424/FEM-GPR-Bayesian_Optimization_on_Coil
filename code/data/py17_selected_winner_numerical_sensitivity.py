"""Mesh and outer-domain checks for the selected BOBYQA and COBYLA winners.

The existing py14 check covers EI-BO.  This companion script evaluates the
other two winners in fresh in-memory COMSOL sessions and never saves the MPH.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from py2_high_fidelity_evaluation_engine import (
    ComsolEvaluator,
    MODEL_FILE,
    set_model_parameters,
    scalar_probe_value,
)


ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "py10_output" / "verification_results.csv"
OUTPUT = ROOT / "py17_output"
METHODS = {"bobyqa": "BOBYQA", "cobyla": "COBYLA"}
MESH_LEVELS = (3, 2, 1)
DOMAIN_SIZES_MM = (200.0, 300.0, 400.0)


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


def run_case(evaluator: ComsolEvaluator, design: pd.Series, mesh_level: int, domain_mm: float) -> dict:
    model = evaluator.model
    if model is None:
        raise RuntimeError("COMSOL evaluator is not open.")
    component = model.java.component("comp1")
    square = component.geom("geom1").feature("sq1")
    size = component.mesh("mesh1").feature("size")
    sample = design.copy()
    sample["sample_id"] = f"{design['label']}_m{mesh_level}_d{domain_mm:g}"
    set_model_parameters(model, sample)
    model.parameter("Japply", "1e6[A/m^2]")
    square.set("size", f"{domain_mm:.12g}")
    square.set("pos", ["0", f"{-domain_mm / 2:.12g}"])
    size.set("hauto", str(mesh_level))
    start = time.perf_counter()
    model.build(); model.mesh()
    mesh_seconds = time.perf_counter() - start
    try:
        elements = int(component.mesh("mesh1").getNumElem())
    except Exception:
        elements = -1
    start = time.perf_counter(); model.solve(); solve_seconds = time.perf_counter() - start
    return {
        "method": design["method"],
        "label": design["label"],
        "seed": int(design["seed"]),
        "mesh_hauto": mesh_level,
        "outer_domain_size_mm": domain_mm,
        "mesh_element_count": elements,
        "target_T": scalar_probe_value(model),
        "mesh_seconds": mesh_seconds,
        "solve_seconds": solve_seconds,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(MODEL_FILE)
    rows = []
    for _, design in selected_designs().iterrows():
        print(f"Opening fresh COMSOL session for {design['label']}...")
        with ComsolEvaluator() as evaluator:
            # The production setting (hauto=3, 200 mm) is shared by both sweeps.
            for level in MESH_LEVELS:
                print(f"  mesh hauto={level}, domain=200 mm")
                rows.append(run_case(evaluator, design, level, 200.0))
            for domain in DOMAIN_SIZES_MM[1:]:
                print(f"  mesh hauto=3, domain={domain:g} mm")
                rows.append(run_case(evaluator, design, 3, domain))
    source_hash_after = sha256_file(MODEL_FILE)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source MPH hash changed during sensitivity checks.")
    frame = pd.DataFrame(rows)
    frame["relative_to_fine_mesh"] = np.nan
    frame["relative_to_largest_domain"] = np.nan
    for label, subset in frame.groupby("label"):
        fine = float(subset.loc[(subset["mesh_hauto"] == 1) & (subset["outer_domain_size_mm"] == 200), "target_T"].iloc[0])
        largest = float(subset.loc[(subset["mesh_hauto"] == 3) & (subset["outer_domain_size_mm"] == 400), "target_T"].iloc[0])
        index = subset.index
        frame.loc[index, "relative_to_fine_mesh"] = np.abs(frame.loc[index, "target_T"] - fine) / abs(fine)
        frame.loc[index, "relative_to_largest_domain"] = np.abs(frame.loc[index, "target_T"] - largest) / abs(largest)
    frame.to_csv(OUTPUT / "sensitivity_results.csv", index=False)
    metadata = {
        "purpose": "mesh and outer-domain checks for BOBYQA and COBYLA winners; no re-optimization",
        "source_model": str(MODEL_FILE),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_model_saved": False,
        "mesh_levels": list(MESH_LEVELS),
        "domain_sizes_mm": list(DOMAIN_SIZES_MM),
        "fresh_stationary_solves": len(frame),
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
