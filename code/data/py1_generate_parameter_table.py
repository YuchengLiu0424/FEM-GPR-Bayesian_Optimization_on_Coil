"""Generate COMSOL design points as an Excel workbook.

Workbook structure:
- Samples: one row = one datapoint; normalized variables first, then physical values
- Parameter_Info: ranges are stated once here
- Metadata: sampling and constraint settings

This script does NOT call COMSOL. It only generates a design table that you can
use later in COMSOL and in Python surrogate training.

Robust output behavior:
- Prefer xlsxwriter when available
- Fall back to openpyxl when available
- If neither is available, write CSV sidecars instead of failing
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import qmc


# =============================================================================
# USER-EDITABLE CONFIG
# =============================================================================

N_SAMPLES = 250
SEED = 42

# How many candidates to generate before filtering constraints.
# Increase this if the feasible region is narrow.
OVERGENERATION_FACTOR = 8
MAX_ATTEMPTS = 20

# Editable mass constraints (grams)
CORE_MASS_LIMIT_G = 200.0
COIL_MASS_LIMIT_G = 50.0

# Material densities (kg/m^3)
RHO_CU_KG_M3 = 8940.0
RHO_CORE_KG_M3 = 7000.0

# Independent design variables and ranges in mm.
# These are the variables to be sampled.
# NOTE:
#   r2 is NOT sampled directly. It is derived as r2 = r1 + t1.
PARAM_BOUNDS_MM = {
    "c1": (0.5, 5.0),    # hole radius; manufacturing minimum
    "r1": (10.0, 20.0),  # core 1 radius
    "t1": (0.5, 10.0),   # radial extension; manufacturing minimum
    "l1": (2.5, 20.0),   # implied minimum from h1 >= 0.5 and coil bottom z = 2
    "l2": (0.5, 20.0),   # core 2 height; manufacturing minimum
    "w1": (0.5, 5.0),    # coil radial width
    "h1": (0.5, 5.0),    # coil height
}

# Optional geometric sanity constraints.
MIN_R1_MINUS_C1_MM = 0.0
MAX_CORE_TOTAL_HEIGHT_MM = None  # e.g. 20.0 or None
MIN_MANUFACTURED_DIMENSION_MM = 0.5
COIL_BOTTOM_Z_MM = 2.0

OUTPUT_DIR = Path(__file__).resolve().parent / "py1_output"
OUTPUT_XLSX = OUTPUT_DIR / "design_points.xlsx"


def design_space_definition() -> dict:
    """Return the versioned design-space contract shared by downstream stages."""
    return {
        "parameter_bounds_mm": {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in PARAM_BOUNDS_MM.items()
        },
        "min_manufactured_dimension_mm": MIN_MANUFACTURED_DIMENSION_MM,
        "coil_bottom_z_mm": COIL_BOTTOM_Z_MM,
        "axial_fit_constraint": "l1_mm >= h1_mm + coil_bottom_z_mm",
        "min_r1_minus_c1_mm": MIN_R1_MINUS_C1_MM,
        "max_core_total_height_mm": MAX_CORE_TOTAL_HEIGHT_MM,
        "core_mass_limit_g": CORE_MASS_LIMIT_G,
        "coil_mass_limit_g": COIL_MASS_LIMIT_G,
        "rho_cu_kg_m3": RHO_CU_KG_M3,
        "rho_core_kg_m3": RHO_CORE_KG_M3,
    }


def design_space_fingerprint() -> str:
    """Stable hash used to reject datasets created under obsolete constraints."""
    payload = json.dumps(
        design_space_definition(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# HELPERS
# =============================================================================

@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo_mm: float
    hi_mm: float

    @property
    def range_str(self) -> str:
        return f"[{self.lo_mm:g}, {self.hi_mm:g}] mm"


def mm_to_m(x_mm: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x_mm) / 1000.0


def denormalize(u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + u * (hi - lo)


def build_specs() -> List[ParamSpec]:
    return [ParamSpec(name=k, lo_mm=v[0], hi_mm=v[1]) for k, v in PARAM_BOUNDS_MM.items()]


DESIGN_VARIABLE_NAMES = tuple(PARAM_BOUNDS_MM)
NORMALIZED_COLUMNS = tuple(f"u_{name}" for name in DESIGN_VARIABLE_NAMES)
PHYSICAL_COLUMNS = tuple(f"{name}_mm" for name in DESIGN_VARIABLE_NAMES)


def unit_to_physical(u: np.ndarray) -> np.ndarray:
    """Map one or more 7-D unit-hypercube points to physical millimetres."""
    values = np.asarray(u, dtype=float)
    if values.shape[-1] != len(DESIGN_VARIABLE_NAMES):
        raise ValueError(f"Expected {len(DESIGN_VARIABLE_NAMES)} design variables, got {values.shape[-1]}.")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Normalized design variables must be finite and lie in [0, 1].")
    lower = np.asarray([PARAM_BOUNDS_MM[name][0] for name in DESIGN_VARIABLE_NAMES])
    upper = np.asarray([PARAM_BOUNDS_MM[name][1] for name in DESIGN_VARIABLE_NAMES])
    return lower + values * (upper - lower)


def physical_to_unit(x_mm: np.ndarray) -> np.ndarray:
    """Map physical millimetre values back to the 7-D unit hypercube."""
    values = np.asarray(x_mm, dtype=float)
    if values.shape[-1] != len(DESIGN_VARIABLE_NAMES) or not np.isfinite(values).all():
        raise ValueError("Physical design values must be finite with seven variables.")
    lower = np.asarray([PARAM_BOUNDS_MM[name][0] for name in DESIGN_VARIABLE_NAMES])
    upper = np.asarray([PARAM_BOUNDS_MM[name][1] for name in DESIGN_VARIABLE_NAMES])
    unit = (values - lower) / (upper - lower)
    if np.any(unit < 0.0) or np.any(unit > 1.0):
        raise ValueError("Physical design values fall outside PARAM_BOUNDS_MM.")
    return unit


def evaluate_unit_candidate(u: np.ndarray, sample_id: str = "candidate") -> pd.Series:
    """Return geometry, mass, and feasibility information for one unit point."""
    unit = np.asarray(u, dtype=float).reshape(-1)
    physical = unit_to_physical(unit)
    row = {"sample_id": str(sample_id)}
    row.update({column: float(value) for column, value in zip(NORMALIZED_COLUMNS, unit)})
    row.update({column: float(value) for column, value in zip(PHYSICAL_COLUMNS, physical)})
    table = compute_masses(add_derived_geometry(pd.DataFrame([row])))
    geometric_feasible = (
        table["c1_mm"].iloc[0] >= MIN_MANUFACTURED_DIMENSION_MM
        and table["r1_mm"].iloc[0] >= table["c1_mm"].iloc[0] + MIN_R1_MINUS_C1_MM
        and table["t1_mm"].iloc[0] >= MIN_MANUFACTURED_DIMENSION_MM
        and table["l1_mm"].iloc[0]
        >= table["h1_mm"].iloc[0] + COIL_BOTTOM_Z_MM
        and table["l2_mm"].iloc[0] >= MIN_MANUFACTURED_DIMENSION_MM
        and table["w1_mm"].iloc[0] >= MIN_MANUFACTURED_DIMENSION_MM
        and table["h1_mm"].iloc[0] >= MIN_MANUFACTURED_DIMENSION_MM
    )
    if MAX_CORE_TOTAL_HEIGHT_MM is not None:
        geometric_feasible = geometric_feasible and (
            table["core_total_height_mm"].iloc[0] <= MAX_CORE_TOTAL_HEIGHT_MM
        )
    table["geometric_feasible"] = geometric_feasible
    table["feasible"] = geometric_feasible and bool(table["total_feasible"].iloc[0])
    return table.iloc[0].copy()


def evaluate_unit_candidates(
    u: np.ndarray,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Vectorized version of the single-point design/constraint interface."""
    unit = np.asarray(u, dtype=float)
    if unit.ndim != 2 or unit.shape[1] != len(DESIGN_VARIABLE_NAMES):
        raise ValueError(f"Expected an (n, {len(DESIGN_VARIABLE_NAMES)}) array.")
    if not np.isfinite(unit).all() or np.any(unit < 0.0) or np.any(unit > 1.0):
        raise ValueError("Normalized design variables must be finite and lie in [0, 1].")
    if sample_ids is None:
        sample_ids = [f"candidate_{index:06d}" for index in range(len(unit))]
    if len(sample_ids) != len(unit):
        raise ValueError("sample_ids must contain one identifier per candidate.")

    physical = unit_to_physical(unit)
    table = pd.DataFrame({"sample_id": list(map(str, sample_ids))})
    for index, column in enumerate(NORMALIZED_COLUMNS):
        table[column] = unit[:, index]
    for index, column in enumerate(PHYSICAL_COLUMNS):
        table[column] = physical[:, index]
    table = compute_masses(add_derived_geometry(table))

    geometric = (
        (table["c1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM)
        & (table["r1_mm"] >= table["c1_mm"] + MIN_R1_MINUS_C1_MM)
        & (table["t1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM)
        & (table["l1_mm"] >= table["h1_mm"] + COIL_BOTTOM_Z_MM)
        & (table["l2_mm"] >= MIN_MANUFACTURED_DIMENSION_MM)
        & (table["w1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM)
        & (table["h1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM)
    )
    if MAX_CORE_TOTAL_HEIGHT_MM is not None:
        geometric &= table["core_total_height_mm"] <= MAX_CORE_TOTAL_HEIGHT_MM
    table["geometric_feasible"] = geometric
    table["feasible"] = geometric & table["total_feasible"]
    return table


def is_unit_candidate_feasible(u: np.ndarray) -> bool:
    """Return py1's single source-of-truth feasibility decision for a unit point."""
    return bool(evaluate_unit_candidate(u)["feasible"])


def validate_fem_dataset_contract(
    table: pd.DataFrame,
    metadata_file: Path,
) -> None:
    """Reject FEM datasets generated under obsolete bounds or constraints."""
    required = ["sample_id", *NORMALIZED_COLUMNS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"FEM dataset is missing contract columns: {missing}")
    if not metadata_file.exists():
        raise ValueError(
            f"Missing FEM metadata file {metadata_file}; the design-space "
            "fingerprint cannot be verified."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("completed") is not True:
        raise ValueError(
            f"FEM dataset is not marked complete in {metadata_file}. "
            "Resume or regenerate py2 before using it downstream."
        )
    stored = metadata.get("design_space_fingerprint")
    current = design_space_fingerprint()
    if stored != current:
        raise ValueError(
            "FEM dataset design-space fingerprint is missing or obsolete. "
            "Regenerate py1 and py2 before fitting or optimizing."
        )
    unit = table.loc[:, NORMALIZED_COLUMNS].to_numpy(dtype=float)
    evaluated = evaluate_unit_candidates(
        unit,
        sample_ids=table["sample_id"].astype(str).tolist(),
    )
    infeasible = evaluated.loc[~evaluated["feasible"].to_numpy(dtype=bool), "sample_id"]
    if not infeasible.empty:
        examples = ", ".join(infeasible.head(8).astype(str))
        raise ValueError(
            f"FEM dataset contains {len(infeasible)} infeasible sample(s) "
            f"under the current contract (examples: {examples})."
        )


def latin_hypercube_candidates(n: int, dim: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=dim, seed=seed, optimization="random-cd")
    return sampler.random(n=n)


def add_derived_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["r2_mm"] = out["r1_mm"] + out["t1_mm"]      # core-2 radius
    out["r0_mm"] = out["r1_mm"] + out["w1_mm"]      # coil outer radius

    out["core_total_height_mm"] = out["l1_mm"] + out["l2_mm"]
    out["c1_margin_mm"] = out["r1_mm"] - out["c1_mm"]
    out["coil_axial_fit_margin_mm"] = (
        out["l1_mm"] - out["h1_mm"] - COIL_BOTTOM_Z_MM
    )

    return out


def compute_masses(df: pd.DataFrame) -> pd.DataFrame:
    """Analytic mass estimates using the finalized COMSOL parameterization.

    Core formula (axisymmetric):
        Mcore = pi * rho_core *
                [ (r1^2 - c1^2) * l1
                  + (r2^2 - c1^2) * l2 ]

        where
            r2 = r1 + t1

    Coil formula:
        Mcoil = pi * rho_cu *
                (r0^2 - r1^2) * h1

        where
            r0 = r1 + w1
    """

    r1 = mm_to_m(df["r1_mm"].to_numpy())
    c1 = mm_to_m(df["c1_mm"].to_numpy())
    l1 = mm_to_m(df["l1_mm"].to_numpy())
    l2 = mm_to_m(df["l2_mm"].to_numpy())
    h1 = mm_to_m(df["h1_mm"].to_numpy())

    # Derived geometric parameters (consistent with COMSOL)
    r2 = mm_to_m(df["r2_mm"].to_numpy())
    r0 = mm_to_m(df["r0_mm"].to_numpy())

    # Core mass
    mcore_kg = (
        math.pi
        * RHO_CORE_KG_M3
        * (
            (r1**2 - c1**2) * l1
            + (r2**2 - c1**2) * l2
        )
    )

    # Coil mass
    mcoil_kg = (
        math.pi
        * RHO_CU_KG_M3
        * (r0**2 - r1**2)
        * h1
    )

    out = df.copy()
    out["Mcore_g"] = mcore_kg * 1000.0
    out["Mcoil_g"] = mcoil_kg * 1000.0
    out["Mtotal_g"] = out["Mcore_g"] + out["Mcoil_g"]

    out["core_feasible"] = out["Mcore_g"] <= CORE_MASS_LIMIT_G
    out["coil_feasible"] = out["Mcoil_g"] <= COIL_MASS_LIMIT_G
    out["total_feasible"] = (
        out["core_feasible"] & out["coil_feasible"]
    )

    return out


def apply_constraints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    # Basic geometry sanity.
    mask &= out["c1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM
    mask &= out["r1_mm"] >= out["c1_mm"] + MIN_R1_MINUS_C1_MM
    mask &= out["t1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM
    mask &= out["l1_mm"] >= out["h1_mm"] + COIL_BOTTOM_Z_MM
    mask &= out["l2_mm"] >= MIN_MANUFACTURED_DIMENSION_MM
    mask &= out["w1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM
    mask &= out["h1_mm"] >= MIN_MANUFACTURED_DIMENSION_MM

    if MAX_CORE_TOTAL_HEIGHT_MM is not None:
        mask &= (out["l1_mm"] + out["l2_mm"]) <= MAX_CORE_TOTAL_HEIGHT_MM

    out = out.loc[mask].copy()
    out = add_derived_geometry(out)
    out = compute_masses(out)
    out = out.loc[out["total_feasible"]].copy()
    return out


def build_candidates(n: int, specs: List[ParamSpec], seed: int, start_idx: int = 0) -> pd.DataFrame:
    dim = len(specs)
    u = latin_hypercube_candidates(n=n, dim=dim, seed=seed)

    rows = []
    for i in range(n):
        row = {"sample_id": f"pt_{start_idx + i:04d}"}
        for j, spec in enumerate(specs):
            val_mm = float(denormalize(u[i, j], spec.lo_mm, spec.hi_mm))
            row[f"u_{spec.name}"] = float(u[i, j])
            row[f"{spec.name}_mm"] = val_mm
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_derived_geometry(df)
    df = compute_masses(df)
    return df


def build_final_table(n_keep: int, specs: List[ParamSpec], seed: int) -> pd.DataFrame:
    feasible_parts = []
    collected = 0
    attempt = 0
    start_idx = 0

    while collected < n_keep and attempt < MAX_ATTEMPTS:
        attempt += 1
        n_candidates = int(max(n_keep * OVERGENERATION_FACTOR, n_keep + 50))
        df = build_candidates(n_candidates, specs, seed + attempt - 1, start_idx=start_idx)
        df = apply_constraints(df)
        if len(df) > 0:
            feasible_parts.append(df)
            collected += len(df)
        start_idx += n_candidates

    if not feasible_parts:
        raise RuntimeError("No feasible samples were generated. Relax ranges or constraints.")

    out = pd.concat(feasible_parts, ignore_index=True)
    if len(out) < n_keep:
        raise RuntimeError(
            f"Only {len(out)} feasible samples were generated, fewer than requested {n_keep}.\n"
            f"Consider increasing ranges, relaxing constraints, or raising OVERGENERATION_FACTOR/MAX_ATTEMPTS."
        )

    out = out.head(n_keep).copy().reset_index(drop=True)
    return out


def reorder_columns(df: pd.DataFrame, specs: List[ParamSpec]) -> pd.DataFrame:
    cols = ["sample_id"]

    # normalized columns first
    for spec in specs:
        cols.append(f"u_{spec.name}")

    # then physical values
    for spec in specs:
        cols.append(f"{spec.name}_mm")

    # derived values and feasibility flags
    cols += [
        "r2_mm",
        "r0_mm",
        "core_total_height_mm",
        "c1_margin_mm",
        "coil_axial_fit_margin_mm",
        "Mcore_g",
        "Mcoil_g",
        "Mtotal_g",
        "core_feasible",
        "coil_feasible",
        "total_feasible",
    ]

    return df[cols].copy()


def build_parameter_info(specs: List[ParamSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "Parameter": spec.name,
                "Min_mm": spec.lo_mm,
                "Max_mm": spec.hi_mm,
                "Unit": "mm",
                "Role": "sampled design variable",
            }
        )

    rows += [
        {"Parameter": "CORE_MASS_LIMIT_G", "Min_mm": CORE_MASS_LIMIT_G, "Max_mm": CORE_MASS_LIMIT_G, "Unit": "g", "Role": "constraint"},
        {"Parameter": "COIL_MASS_LIMIT_G", "Min_mm": COIL_MASS_LIMIT_G, "Max_mm": COIL_MASS_LIMIT_G, "Unit": "g", "Role": "constraint"},
        {"Parameter": "MIN_R1_MINUS_C1_MM", "Min_mm": MIN_R1_MINUS_C1_MM, "Max_mm": MIN_R1_MINUS_C1_MM, "Unit": "mm", "Role": "geometry constraint"},
        {"Parameter": "MIN_MANUFACTURED_DIMENSION_MM", "Min_mm": MIN_MANUFACTURED_DIMENSION_MM, "Max_mm": MIN_MANUFACTURED_DIMENSION_MM, "Unit": "mm", "Role": "manufacturing constraint"},
        {"Parameter": "COIL_BOTTOM_Z_MM", "Min_mm": COIL_BOTTOM_Z_MM, "Max_mm": COIL_BOTTOM_Z_MM, "Unit": "mm", "Role": "axial fit constraint"},
    ]
    if MAX_CORE_TOTAL_HEIGHT_MM is not None:
        rows.append(
            {
                "Parameter": "MAX_CORE_TOTAL_HEIGHT_MM",
                "Min_mm": MAX_CORE_TOTAL_HEIGHT_MM,
                "Max_mm": MAX_CORE_TOTAL_HEIGHT_MM,
                "Unit": "mm",
                "Role": "geometry constraint",
            }
        )
    return pd.DataFrame(rows)


def build_metadata(n_keep: int, specs: List[ParamSpec]) -> pd.DataFrame:
    rows = [
        {"Item": "N_SAMPLES", "Value": n_keep},
        {"Item": "SEED", "Value": SEED},
        {"Item": "OVERGENERATION_FACTOR", "Value": OVERGENERATION_FACTOR},
        {"Item": "MAX_ATTEMPTS", "Value": MAX_ATTEMPTS},
        {"Item": "Sampling_method", "Value": "Latin Hypercube Sampling (random-cd)"},
        {"Item": "Core_mass_limit_g", "Value": CORE_MASS_LIMIT_G},
        {"Item": "Coil_mass_limit_g", "Value": COIL_MASS_LIMIT_G},
        {"Item": "MIN_R1_MINUS_C1_MM", "Value": MIN_R1_MINUS_C1_MM},
        {"Item": "MIN_MANUFACTURED_DIMENSION_MM", "Value": MIN_MANUFACTURED_DIMENSION_MM},
        {"Item": "COIL_BOTTOM_Z_MM", "Value": COIL_BOTTOM_Z_MM},
        {"Item": "Axial_fit_constraint", "Value": "l1_mm >= h1_mm + COIL_BOTTOM_Z_MM"},
        {"Item": "Design_space_fingerprint", "Value": design_space_fingerprint()},
        {"Item": "MAX_CORE_TOTAL_HEIGHT_MM", "Value": MAX_CORE_TOTAL_HEIGHT_MM},
        {"Item": "RHO_CU_KG_M3", "Value": RHO_CU_KG_M3},
        {"Item": "RHO_CORE_KG_M3", "Value": RHO_CORE_KG_M3},
        {"Item": "Sample_sheet_columns", "Value": ", ".join(["sample_id"] + [f"u_{s.name}" for s in specs] + [f"{s.name}_mm" for s in specs] + ["r2_mm", "r0_mm", "core_total_height_mm", "c1_margin_mm", "coil_axial_fit_margin_mm", "Mcore_g", "Mcoil_g", "Mtotal_g", "core_feasible", "coil_feasible", "total_feasible"])},
    ]
    return pd.DataFrame(rows)


def _select_excel_engine() -> str | None:
    """Pick an installed Excel writer engine without assuming openpyxl exists."""
    for engine in ("xlsxwriter", "openpyxl"):
        try:
            __import__(engine)
            return engine
        except Exception:
            continue
    return None


def write_outputs(outdir: Path, samples: pd.DataFrame, specs: List[ParamSpec]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    parameter_info = build_parameter_info(specs)
    metadata = build_metadata(len(samples), specs)

    # CSV files are the canonical machine-readable interface used by py2 and
    # downstream scripts. Always refresh them, even when Excel export succeeds,
    # so the workbook and CSV sidecars cannot describe different design spaces
    # or constraint versions.
    samples.to_csv(outdir / "design_points_samples.csv", index=False)
    parameter_info.to_csv(outdir / "design_points_parameter_info.csv", index=False)
    metadata.to_csv(outdir / "design_points_metadata.csv", index=False)

    engine = _select_excel_engine()
    excel_written = False

    if engine is not None:
        try:
            with pd.ExcelWriter(OUTPUT_XLSX, engine=engine) as writer:
                samples.to_excel(writer, sheet_name="Samples", index=False)
                parameter_info.to_excel(writer, sheet_name="Parameter_Info", index=False)
                metadata.to_excel(writer, sheet_name="Metadata", index=False)
            excel_written = True
        except Exception as exc:
            print(f"[warn] Excel export failed with engine={engine}: {exc}")

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": int(len(samples)),
                "seed": SEED,
                "output_xlsx": str(OUTPUT_XLSX),
                "excel_written": excel_written,
                "excel_engine": engine,
                "core_mass_limit_g": CORE_MASS_LIMIT_G,
                "coil_mass_limit_g": COIL_MASS_LIMIT_G,
                "design_space_definition": design_space_definition(),
                "design_space_fingerprint": design_space_fingerprint(),
                "mass_stats_g": {
                    "core_min": float(samples["Mcore_g"].min()),
                    "core_max": float(samples["Mcore_g"].max()),
                    "coil_min": float(samples["Mcoil_g"].min()),
                    "coil_max": float(samples["Mcoil_g"].max()),
                    "total_min": float(samples["Mtotal_g"].min()),
                    "total_max": float(samples["Mtotal_g"].max()),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(outdir / "notes.md", "w", encoding="utf-8") as f:
        f.write("# Py1 design table\n\n")
        f.write("## Workbook structure\n\n")
        f.write(f"- Samples: {len(samples)} datapoints, no repeated range columns\n")
        f.write("- Parameter_Info: ranges stated once\n")
        f.write("- Metadata: sampling and constraint settings\n\n")
        f.write("## Editable settings\n\n")
        f.write(f"- N_SAMPLES = {N_SAMPLES}\n")
        f.write(f"- SEED = {SEED}\n")
        f.write(f"- OVERGENERATION_FACTOR = {OVERGENERATION_FACTOR}\n")
        f.write(f"- MAX_ATTEMPTS = {MAX_ATTEMPTS}\n")
        f.write(f"- CORE_MASS_LIMIT_G = {CORE_MASS_LIMIT_G}\n")
        f.write(f"- COIL_MASS_LIMIT_G = {COIL_MASS_LIMIT_G}\n")
        f.write(f"- MIN_R1_MINUS_C1_MM = {MIN_R1_MINUS_C1_MM}\n")
        f.write(f"- MIN_MANUFACTURED_DIMENSION_MM = {MIN_MANUFACTURED_DIMENSION_MM}\n")
        f.write(f"- COIL_BOTTOM_Z_MM = {COIL_BOTTOM_Z_MM}\n")
        f.write(f"- Axial fit: l1_mm >= h1_mm + {COIL_BOTTOM_Z_MM:g}\n")
        f.write(f"- Design-space fingerprint: {design_space_fingerprint()}\n")
        f.write(f"- MAX_CORE_TOTAL_HEIGHT_MM = {MAX_CORE_TOTAL_HEIGHT_MM}\n")
        f.write(f"- Excel engine used: {engine if excel_written else 'fallback CSV'}\n")

def main() -> None:
    specs = build_specs()
    table = build_final_table(N_SAMPLES, specs, SEED)
    table = reorder_columns(table, specs)
    write_outputs(OUTPUT_DIR, table, specs)

    print(table.to_string(index=False))
    print(f"\nSaved workbook to: {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
