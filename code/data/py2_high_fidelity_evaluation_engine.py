"""Generate Gaussian-process training targets from the COMSOL model.

The output schema intentionally matches the established py2 contract:
``sample_id, u_c1, ..., u_h1, target``.  This lets py3 and py4 use either
the placeholder or the high-fidelity FEM data without code changes.

Run with an ASCII-path Python installation on this computer, for example::

    D:\\Tools\\Python\\python.exe py2_high_fidelity_evaluation_engine.py --limit 1
    D:\\Tools\\Python\\python.exe py2_high_fidelity_evaluation_engine.py

The first command is a one-sample smoke test saved separately from the
finalized dataset.  The second evaluates all design points and writes
``py2_output/output.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mph
import numpy as np
import pandas as pd

import py1_generate_parameter_table as py1


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PY1_DIR = SCRIPT_DIR / "py1_output"
PY2_DIR = SCRIPT_DIR / "py2_output"
MODEL_FILE = PROJECT_DIR / "surrogate_simulation" / "ML surrogate.mph"

OUTPUT_FILE = PY2_DIR / "output.csv"
METADATA_FILE = PY2_DIR / "metadata.json"
SMOKE_OUTPUT_FILE = PY2_DIR / "smoke_test_output.csv"
SMOKE_METADATA_FILE = PY2_DIR / "smoke_test_metadata.json"

COMSOL_VERSION = "6.3"
GEOMETRY_TAG = "geom1"
MESH_TAG = "mesh1"
STUDY_TAG = "std1"
PROBE_TAG = "pev1"
TARGET_NAME = "target"
TARGET_EXPRESSION = "mf.normB"
TARGET_UNIT = "T"

DESIGN_COLUMNS = ["c1_mm", "r1_mm", "t1_mm", "l1_mm", "l2_mm", "w1_mm", "h1_mm"]
NORMALIZED_COLUMNS = ["u_c1", "u_r1", "u_t1", "u_l1", "u_l2", "u_w1", "u_h1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preconfigured COMSOL ROI surface-average probe for py1 design samples."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (use 1 for a smoke test).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching incomplete checkpoint and skip completed sample IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of refusing to start.",
    )
    return parser.parse_args()


def find_design_file() -> Path:
    candidates = [
        PY1_DIR / "design_points_samples.csv",
        PY1_DIR / "design_points.xlsx",
    ]
    design_file = next((path for path in candidates if path.exists()), None)
    if design_file is None:
        raise FileNotFoundError(
            "No py1 design table found. Expected design_points_samples.csv or design_points.xlsx."
        )
    return design_file


def load_design_table(limit: int | None = None) -> pd.DataFrame:
    """Load py1's design table and validate the py2 interface contract."""
    design_file = find_design_file()

    if design_file.suffix.lower() == ".csv":
        table = pd.read_csv(design_file)
    else:
        table = pd.read_excel(design_file, sheet_name="Samples")

    required = ["sample_id", *NORMALIZED_COLUMNS, *DESIGN_COLUMNS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Design table is missing required columns: {missing}")

    if table["sample_id"].astype(str).duplicated().any():
        raise ValueError("Design table contains duplicate sample_id values.")

    physical = table[DESIGN_COLUMNS].to_numpy(dtype=float)
    normalized = table[NORMALIZED_COLUMNS].to_numpy(dtype=float)
    try:
        expected_normalized = py1.physical_to_unit(physical)
    except ValueError as exc:
        raise ValueError(
            "Design table was generated with obsolete parameter bounds. "
            "Regenerate py1_output before running COMSOL."
        ) from exc
    if not np.allclose(normalized, expected_normalized, rtol=0.0, atol=1e-10):
        raise ValueError(
            "Stored normalized and physical design coordinates do not match "
            "the current py1 bounds. Regenerate py1_output."
        )
    evaluated = py1.evaluate_unit_candidates(
        expected_normalized,
        sample_ids=table["sample_id"].astype(str).tolist(),
    )
    infeasible = evaluated.loc[~evaluated["feasible"].to_numpy(dtype=bool), "sample_id"]
    if not infeasible.empty:
        examples = ", ".join(infeasible.head(8).astype(str))
        raise ValueError(
            f"Design table contains {len(infeasible)} infeasible sample(s) "
            f"under the current manufacturing, fit, and mass constraints "
            f"(examples: {examples}). Regenerate py1_output."
        )

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        table = table.head(limit).copy()

    if table.empty:
        raise ValueError("The selected design table is empty.")
    if limit is None and len(table) != py1.N_SAMPLES:
        raise ValueError(
            f"Full FEM generation requires exactly {py1.N_SAMPLES} designs; "
            f"found {len(table)} in {design_file}."
        )

    return table


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_stationary_point_evaluation(model: Any) -> dict[str, Any]:
    """Enable only the Stationary study step needed for one design evaluation."""
    study = model.java.study(STUDY_TAG)
    feature_types: dict[str, str] = {}
    stationary: list[str] = []
    disabled_optimization: list[str] = []
    for tag in list(study.feature().tags()):
        feature = study.feature(tag)
        feature_type = str(feature.getType())
        feature_types[str(tag)] = feature_type
        if feature_type.lower() == "optimization":
            feature.active(False)
            disabled_optimization.append(str(tag))
        elif feature_type.lower() == "stationary":
            feature.active(True)
            stationary.append(str(tag))
    if not stationary:
        raise RuntimeError(
            f"Study {STUDY_TAG!r} has no Stationary step: {feature_types}"
        )
    return {
        "study_feature_types": feature_types,
        "stationary_features": stationary,
        "disabled_optimization_features": disabled_optimization,
    }


def validate_model_contract(model: Any) -> dict[str, Any]:
    """Abort before FEM work if the source, domains, or probe have drifted."""
    java = model.java
    component = java.component("comp1")
    magnetic = component.physics("mf")
    coil = magnetic.feature("coil1")
    domain_contract = {
        "coil": list(coil.selection().entities()),
        "copper": list(component.material("mat1").selection().entities()),
        "air": list(component.material("mat2").selection().entities()),
        "core": list(component.material("mat3").selection().entities()),
    }
    expected_domains = {
        "coil": [3],
        "copper": [3],
        "air": [1],
        "core": [2],
    }
    if domain_contract != expected_domains:
        raise RuntimeError(
            f"COMSOL domain contract drifted: {domain_contract}; "
            f"expected {expected_domains}."
        )
    coil_current = str(coil.getString("ICoil")).replace(" ", "")
    if coil_current != "h1*w1*Japply":
        raise RuntimeError(
            f"Unexpected coil-current expression {coil_current!r}; "
            "expected 'h1*w1*Japply'."
        )
    excitation = str(coil.getString("CoilExcitation"))
    if excitation.lower() != "current":
        raise RuntimeError(
            f"Unexpected coil excitation {excitation!r}; expected current."
        )
    japply = str(java.param().get("Japply")).replace(" ", "")
    if japply.lower() not in {"1e6[a/m^2]", "1000000[a/m^2]"}:
        raise RuntimeError(
            f"Unexpected Japply={japply!r}; expected 1e6[A/m^2]."
        )
    probe = java.result().numerical(PROBE_TAG)
    probe_expression = str(probe.getString("expr")).replace(" ", "")
    if probe_expression != TARGET_EXPRESSION:
        raise RuntimeError(
            f"Unexpected probe expression {probe_expression!r}; "
            f"expected {TARGET_EXPRESSION!r}."
        )
    study_contract = configure_stationary_point_evaluation(model)
    return {
        "domains": domain_contract,
        "coil_excitation": excitation,
        "coil_current_expression": coil_current,
        "Japply": japply,
        "probe_expression": probe_expression,
        **study_contract,
    }


def set_model_parameters(model: Any, sample: pd.Series) -> None:
    """Set the seven independent geometric parameters in COMSOL, in millimetres."""
    for column in DESIGN_COLUMNS:
        parameter = column.removesuffix("_mm")
        value_mm = float(sample[column])
        if not np.isfinite(value_mm):
            raise ValueError(f"{sample['sample_id']}: {column} is not finite.")
        model.parameter(parameter, f"{value_mm:.12g}[mm]")


def scalar_probe_value(model: Any) -> float:
    """Read the preconfigured ROI surface-average numerical evaluation."""
    raw_values = model.java.result().numerical(PROBE_TAG).getReal()
    values = np.asarray([list(row) for row in raw_values], dtype=float).reshape(-1)
    if values.size != 1:
        raise RuntimeError(
            f"Expected one value from COMSOL probe {PROBE_TAG!r}, got {values.size}: {values!r}"
        )
    value = float(values[0])
    if not np.isfinite(value):
        raise RuntimeError(f"COMSOL probe {PROBE_TAG!r} returned a non-finite value: {value!r}")
    return value


def evaluate_sample(model: Any, sample: pd.Series) -> dict[str, float | str]:
    """Set geometry, solve the stationary study, and return one py3-compatible row."""
    set_model_parameters(model, sample)
    # mph's high-level methods address nodes by their UI labels, while this
    # Chinese-language model has localized labels.  Running all sequences is
    # unambiguous here because the model contains exactly one geometry, mesh,
    # and study; the internal tags above remain metadata for traceability.
    model.build()
    model.mesh()
    model.java.study(STUDY_TAG).run()

    row: dict[str, float | str] = {"sample_id": str(sample["sample_id"])}
    row.update({column: float(sample[column]) for column in NORMALIZED_COLUMNS})
    row[TARGET_NAME] = scalar_probe_value(model)
    return row


class ComsolEvaluator:
    """One reusable COMSOL session for both bulk and sequential evaluation."""

    def __init__(self, model_file: Path = MODEL_FILE, comsol_version: str = COMSOL_VERSION) -> None:
        self.model_file = Path(model_file)
        self.comsol_version = comsol_version
        self.client: Any | None = None
        self.model: Any | None = None

    def open(self) -> "ComsolEvaluator":
        if self.model is not None:
            return self
        if not self.model_file.exists():
            raise FileNotFoundError(f"COMSOL model file was not found: {self.model_file}")
        self.client = mph.start(version=self.comsol_version)
        self.model = self.client.load(str(self.model_file))
        probe_tags = list(self.model.java.result().numerical().tags())
        if PROBE_TAG not in probe_tags:
            self.close()
            raise RuntimeError(f"Configured probe {PROBE_TAG!r} was not found. Available probes: {probe_tags}")
        validate_model_contract(self.model)
        return self

    def evaluate(self, sample: pd.Series) -> dict[str, float | str]:
        if self.model is None:
            raise RuntimeError("COMSOL session is not open; call open() first.")
        return evaluate_sample(self.model, sample)

    def close(self) -> None:
        if self.client is not None and self.model is not None:
            try:
                self.client.remove(self.model)
            except Exception:
                # COMSOL/MPh may already have removed the model while closing
                # the server.  Cleanup must remain idempotent.
                pass
        if self.client is not None:
            try:
                self.client.clear()
            except Exception:
                pass
        self.client, self.model = None, None

    def __enter__(self) -> "ComsolEvaluator":
        return self.open()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def save_outputs(
    dataset: pd.DataFrame,
    metadata: dict[str, Any],
    output_file: Path = OUTPUT_FILE,
    metadata_file: Path = METADATA_FILE,
) -> None:
    expected_columns = ["sample_id", *NORMALIZED_COLUMNS, TARGET_NAME]
    if list(dataset.columns) != expected_columns:
        raise RuntimeError(f"Unexpected py2 output columns: {list(dataset.columns)}")
    if dataset.empty or not np.isfinite(dataset[TARGET_NAME].to_numpy(dtype=float)).all():
        raise RuntimeError("Refusing to save an empty or non-finite FEM target dataset.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_metadata = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    dataset.to_csv(temporary_output, index=False)
    with temporary_metadata.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    temporary_output.replace(output_file)
    temporary_metadata.replace(metadata_file)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    design_table = load_design_table(args.limit)
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"COMSOL model file was not found: {MODEL_FILE}")

    output_file = SMOKE_OUTPUT_FILE if args.limit is not None else OUTPUT_FILE
    metadata_file = SMOKE_METADATA_FILE if args.limit is not None else METADATA_FILE
    if output_file.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"{output_file} already exists. Use --resume for an incomplete "
            "checkpoint or --overwrite for a deliberate fresh run."
        )

    existing = pd.DataFrame(columns=["sample_id", *NORMALIZED_COLUMNS, TARGET_NAME])
    if args.resume:
        if not output_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                "--resume requires both the existing output CSV and metadata JSON."
            )
        stored_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if stored_metadata.get("design_space_fingerprint") != py1.design_space_fingerprint():
            raise ValueError("Cannot resume a FEM checkpoint from another design space.")
        if stored_metadata.get("model_sha256") != sha256_file(MODEL_FILE):
            raise ValueError("Cannot resume because the source MPH hash changed.")
        existing = pd.read_csv(output_file)
        expected_columns = ["sample_id", *NORMALIZED_COLUMNS, TARGET_NAME]
        if list(existing.columns) != expected_columns:
            raise ValueError(f"Checkpoint columns do not match: {list(existing.columns)}")
        if existing["sample_id"].astype(str).duplicated().any():
            raise ValueError("Checkpoint contains duplicate sample IDs.")
        completed_count = len(existing)
        expected_prefix = design_table.head(completed_count)
        if existing["sample_id"].astype(str).tolist() != expected_prefix["sample_id"].astype(str).tolist():
            raise ValueError("Checkpoint sample IDs are not the design-table prefix.")
        if not np.allclose(
            existing[NORMALIZED_COLUMNS].to_numpy(float),
            expected_prefix[NORMALIZED_COLUMNS].to_numpy(float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("Checkpoint coordinates do not match the design table.")

    print(
        f"Loading {len(design_table)} design sample(s); "
        f"{len(existing)} already completed."
    )
    print(f"COMSOL model: {MODEL_FILE}")
    print(f"Target: {PROBE_TAG} ({TARGET_EXPRESSION}, {TARGET_UNIT})")

    start = time.perf_counter()
    design_file = find_design_file()
    base_metadata = {
        "solver": "COMSOL",
        "comsol_version": COMSOL_VERSION,
        "model_file": str(MODEL_FILE),
        "model_sha256": sha256_file(MODEL_FILE),
        "design_file": str(design_file),
        "design_file_sha256": sha256_file(design_file),
        "study_tag": STUDY_TAG,
        "probe_tag": PROBE_TAG,
        "target_expression": TARGET_EXPRESSION,
        "target_unit": TARGET_UNIT,
        "expected_samples": int(len(design_table)),
        "generation_mode": "smoke_test" if args.limit is not None else "full_dataset",
        "design_space_definition": py1.design_space_definition(),
        "design_space_fingerprint": py1.design_space_fingerprint(),
        "completed": False,
        "output_file": str(output_file),
    }
    with ComsolEvaluator() as evaluator:
        rows = existing.to_dict("records")
        for index, (_, sample) in enumerate(
            design_table.iloc[len(existing):].iterrows(),
            start=len(existing) + 1,
        ):
            sample_start = time.perf_counter()
            result = evaluator.evaluate(sample)
            rows.append(result)
            elapsed = time.perf_counter() - sample_start
            print(
                f"[{index}/{len(design_table)}] {result['sample_id']}: "
                f"{TARGET_NAME}={result[TARGET_NAME]:.8g} {TARGET_UNIT} ({elapsed:.1f} s)"
            )
            checkpoint = pd.DataFrame(
                rows,
                columns=["sample_id", *NORMALIZED_COLUMNS, TARGET_NAME],
            )
            checkpoint_metadata = {
                **base_metadata,
                "n_samples": int(len(checkpoint)),
                "generation_time_seconds": time.perf_counter() - start,
            }
            save_outputs(
                checkpoint,
                checkpoint_metadata,
                output_file,
                metadata_file,
            )

        dataset = pd.DataFrame(
            rows,
            columns=["sample_id", *NORMALIZED_COLUMNS, TARGET_NAME],
        )
        if len(dataset) != len(design_table):
            raise RuntimeError(
                f"FEM dataset stopped at {len(dataset)}/{len(design_table)} samples."
            )
        metadata = {
            **base_metadata,
            "n_samples": int(len(dataset)),
            "generation_time_seconds": time.perf_counter() - start,
            "completed": True,
        }
        save_outputs(dataset, metadata, output_file, metadata_file)
        print(f"Saved FEM targets to: {output_file}")


if __name__ == "__main__":
    main()
