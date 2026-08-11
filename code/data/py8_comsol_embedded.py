"""Run the tuned optimization study embedded in the COMSOL model.

This runner deliberately does not reimplement optimization algorithms in
Python. It opens ``ML surrogate.mph``, enables the existing Optimization
study step, selects each requested COMSOL optimizer, runs the study, and
reads both results and timing from COMSOL itself.

The Optimization study step is disabled again in memory in a ``finally``
block. The source MPH is strictly read-only: this runner never saves any
in-memory parameter, geometry, physics, material, study, or result changes
back to the model file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import jpype
import mph
import numpy as np
import pandas as pd

import py1_generate_parameter_table as py1
import py6_optimization_common as common


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_MODEL_FILE = PROJECT_DIR / "surrogate_simulation" / "ML surrogate.mph"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "py8_output"
PY7_DIR = SCRIPT_DIR / "py7_output"

COMSOL_VERSION = "6.3"
MIN_START_BOX_MARGIN = 0.10
STUDY_TAG = "std1"
OPTIMIZATION_TAG = "opt"
OBJECTIVE_TABLE_TAG = "tbl8"

METHODS = {
    "nelder_mead": "neldermead",
    "bobyqa": "bobyqa",
    "cobyla": "cobyla",
    "ego": "ego",
}
DISPLAY_NAMES = {
    "nelder_mead": "Nelder-Mead",
    "bobyqa": "BOBYQA",
    "cobyla": "COBYLA",
    "ego": "EGO",
}

TIME_PATTERN = re.compile(
    r"(?:Solution\s+time|Solver\s+time|求解时间)\s*[:：]\s*"
    r"(?P<seconds>\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?",
    flags=re.IGNORECASE,
)
EVALUATION_COUNT_PATTERN = re.compile(
    r"(?:Objective\s+function\s+evaluations|目标函数计算次数)\s*[:：]\s*"
    r"(?P<count>\d+)",
    flags=re.IGNORECASE,
)
CHINESE_TIMESTAMP_PATTERN = re.compile(
    r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日\s*"
    r"(?P<ampm>上午|下午)(?P<hour>\d{1,2}):"
    r"(?P<minute>\d{2}):(?P<second>\d{2})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the optimization algorithms embedded in the COMSOL MPH model."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=tuple(METHODS),
        help="COMSOL optimizers to run (default: all four).",
    )
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(common.SEEDS),
        help="Shared py7 initial-design seeds (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=25,
        help="Number of shared initial FEM observations per seed (default: 25).",
    )
    parser.add_argument(
        "--continuation-budget",
        type=int,
        default=50,
        help=(
            "Maximum COMSOL optimizer objective-evaluation attempts after the "
            "shared initial design (default: 50)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an explicitly selected existing run directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip already completed method-seed pairs whose saved protocol "
            "matches this invocation; useful for safely resuming long runs."
        ),
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The source MPH is always read-only "
            "and is never saved."
        ),
    )
    parser.add_argument(
        "--comsol-version",
        default=COMSOL_VERSION,
        help="COMSOL version passed to MPh (default: 6.3).",
    )
    return parser.parse_args()


def validate_model(java: Any) -> Any:
    study_tags = list(java.study().tags())
    if STUDY_TAG not in study_tags:
        raise RuntimeError(
            f"Study {STUDY_TAG!r} was not found. Available studies: {study_tags}"
        )
    study = java.study(STUDY_TAG)
    feature_tags = list(study.feature().tags())
    if OPTIMIZATION_TAG not in feature_tags:
        raise RuntimeError(
            f"Optimization feature {OPTIMIZATION_TAG!r} was not found in "
            f"{STUDY_TAG!r}. Available features: {feature_tags}"
        )
    table_tags = list(java.result().table().tags())
    if OBJECTIVE_TABLE_TAG not in table_tags:
        raise RuntimeError(
            f"Objective table {OBJECTIVE_TABLE_TAG!r} was not found. "
            f"Available tables: {table_tags}"
        )
    optimization = study.feature(OPTIMIZATION_TAG)

    def tokens(name: str) -> list[str]:
        try:
            value = str(optimization.getString(name))
        except Exception:
            value = str(optimization.get(name))
        return [token.strip().replace(" ", "") for token in value.split(",")]

    expected = {
        "pname": list(py1.DESIGN_VARIABLE_NAMES),
        "lbound": [
            f"{py1.PARAM_BOUNDS_MM[name][0]:g}[mm]"
            for name in py1.DESIGN_VARIABLE_NAMES
        ],
        "ubound": [
            f"{py1.PARAM_BOUNDS_MM[name][1]:g}[mm]"
            for name in py1.DESIGN_VARIABLE_NAMES
        ],
        "constraintExpression": ["Mcore", "Mcoil", "l1-h1-2[mm]"],
        "constraintLbound": ["0", "0", "0"],
        "constraintUbound": [
            f"{py1.CORE_MASS_LIMIT_G:g}[g]",
            f"{py1.COIL_MASS_LIMIT_G:g}[g]",
            "",
        ],
    }
    observed = {name: tokens(name) for name in expected}
    mismatches = {
        name: {"observed": observed[name], "expected": values}
        for name, values in expected.items()
        if observed[name] != [value.replace(" ", "") for value in values]
    }
    if mismatches:
        raise RuntimeError(
            "Embedded COMSOL bounds or constraints disagree with py1: "
            f"{mismatches}"
        )
    return optimization


def java_property(node: Any, name: str) -> str:
    """Read a scalar COMSOL feature property without relying on localization."""
    try:
        return str(node.getString(name))
    except Exception:
        return str(node.get(name))


def objective_table(java: Any) -> tuple[list[str], np.ndarray]:
    table = java.result().table(OBJECTIVE_TABLE_TAG)
    headers = [str(value) for value in table.getColumnHeaders()]
    rows = np.asarray([list(row) for row in table.getReal()], dtype=float)
    expected_columns = len(py1.DESIGN_VARIABLE_NAMES) + 1
    if rows.ndim != 2 or rows.shape[1] != expected_columns:
        raise RuntimeError(
            f"Unexpected COMSOL objective table shape {rows.shape}; expected "
            f"(n, {expected_columns}). Headers: {headers}"
        )
    if len(rows) == 0:
        raise RuntimeError("COMSOL objective table is empty after the optimization run.")
    if not np.isfinite(rows[:, :-1]).all():
        raise RuntimeError("COMSOL objective table contains non-finite control variables.")
    if not np.isfinite(rows[:, -1]).any():
        raise RuntimeError("COMSOL objective table contains no finite objective value.")
    return headers, rows


def decoded_progress_texts(progress_file: Path) -> list[str]:
    """Decode COMSOL progress output, including its localized mojibake form."""
    raw = progress_file.read_bytes()
    decoded: list[str] = []
    utf8_text: str | None = None
    try:
        utf8_text = raw.decode("utf-8")
        decoded.append(utf8_text)
    except UnicodeDecodeError:
        pass
    # The localized COMSOL server can write already-mojibaked Chinese text.
    # Reverse that specific GB18030/UTF-8 round trip before parsing timestamps.
    if utf8_text is not None:
        try:
            decoded.append(utf8_text.encode("gb18030").decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    for encoding in ("utf-16", "gb18030"):
        try:
            decoded.append(raw.decode(encoding))
        except UnicodeDecodeError:
            pass
    return decoded


def parse_comsol_evaluation_count(progress_file: Path) -> int:
    """Read COMSOL's reported objective-function evaluation count."""
    counts: list[int] = []
    for text in decoded_progress_texts(progress_file):
        counts.extend(
            int(match.group("count"))
            for match in EVALUATION_COUNT_PATTERN.finditer(text)
        )
    if not counts:
        raise RuntimeError(
            "COMSOL did not report an objective-function evaluation count. "
            f"Inspect the temporary progress log before cleanup: {progress_file}"
        )
    return counts[-1]


def parse_comsol_reported_seconds(progress_file: Path) -> tuple[float, str]:
    """Return COMSOL's own reported solve time, never Python wall-clock time."""

    timestamps: list[tuple[datetime, str]] = []
    candidates: list[tuple[float, str]] = []
    for text in decoded_progress_texts(progress_file):
        for line in text.splitlines():
            timestamp = CHINESE_TIMESTAMP_PATTERN.search(line)
            if timestamp and ("开始于" in line or "结束时间" in line):
                hour = int(timestamp.group("hour"))
                if timestamp.group("ampm") == "下午" and hour != 12:
                    hour += 12
                if timestamp.group("ampm") == "上午" and hour == 12:
                    hour = 0
                value = datetime(
                    int(timestamp.group("year")),
                    int(timestamp.group("month")),
                    int(timestamp.group("day")),
                    hour,
                    int(timestamp.group("minute")),
                    int(timestamp.group("second")),
                )
                timestamps.append((value, line.strip()))
            match = TIME_PATTERN.search(line)
            if match:
                candidates.append((float(match.group("seconds")), line.strip()))
    if timestamps:
        first = min(timestamps, key=lambda item: item[0])
        last = max(timestamps, key=lambda item: item[0])
        elapsed = (last[0] - first[0]).total_seconds()
        if elapsed > 0:
            source = (
                f"COMSOL progress timestamps: {first[1]} -> {last[1]}"
            )
            return elapsed, source
    if not candidates:
        raise RuntimeError(
            "COMSOL did not emit parseable solve times or start/end timestamps. "
            f"Inspect the temporary progress log before cleanup: {progress_file}"
        )
    return max(candidates, key=lambda item: item[0])


def canonical_observations(
    method: str, seed: int, rows: np.ndarray
) -> tuple[pd.DataFrame, pd.Series]:
    records: list[dict[str, Any]] = []
    lower = np.asarray(
        [py1.PARAM_BOUNDS_MM[name][0] for name in py1.DESIGN_VARIABLE_NAMES],
        dtype=float,
    )
    upper = np.asarray(
        [py1.PARAM_BOUNDS_MM[name][1] for name in py1.DESIGN_VARIABLE_NAMES],
        dtype=float,
    )
    tolerance = 1e-7 * np.maximum(upper - lower, 1.0)
    for index, row in enumerate(rows, start=1):
        physical_mm = row[:-1] * 1000.0
        within_bounds = bool(
            np.all(physical_mm >= lower - tolerance)
            and np.all(physical_mm <= upper + tolerance)
        )
        # COBYLA can deliberately sample outside simple bounds while building
        # its local linear model. Preserve those true COMSOL evaluations
        # instead of clipping or discarding them; they are marked infeasible
        # and can never become the reported best design.
        unit = (physical_mm - lower) / (upper - lower)
        record: dict[str, Any] = {
            "sample_id": f"comsol_{method}_{seed}_{index:04d}",
            **{
                name: float(value)
                for name, value in zip(py1.NORMALIZED_COLUMNS, unit)
            },
            **{
                name: float(value)
                for name, value in zip(py1.PHYSICAL_COLUMNS, physical_mm)
            },
        }
        frame = pd.DataFrame([record])
        frame = py1.compute_masses(py1.add_derived_geometry(frame))
        record.update(frame.iloc[0].to_dict())
        geometric_feasible = bool(
            record["c1_mm"] >= py1.MIN_MANUFACTURED_DIMENSION_MM
            and record["r1_mm"] >= record["c1_mm"] + py1.MIN_R1_MINUS_C1_MM
            and record["t1_mm"] >= py1.MIN_MANUFACTURED_DIMENSION_MM
            and record["l1_mm"]
            >= record["h1_mm"] + py1.COIL_BOTTOM_Z_MM
            and record["l2_mm"] >= py1.MIN_MANUFACTURED_DIMENSION_MM
            and record["w1_mm"] >= py1.MIN_MANUFACTURED_DIMENSION_MM
            and record["h1_mm"] >= py1.MIN_MANUFACTURED_DIMENSION_MM
        )
        if py1.MAX_CORE_TOTAL_HEIGHT_MM is not None:
            geometric_feasible = geometric_feasible and bool(
                record["core_total_height_mm"]
                <= py1.MAX_CORE_TOTAL_HEIGHT_MM
            )
        record["within_design_bounds"] = within_bounds
        record["geometric_feasible"] = geometric_feasible
        record["feasible"] = bool(
            within_bounds
            and geometric_feasible
            and record["total_feasible"]
        )
        record.update(
            {
                common.TARGET_COLUMN: float(row[-1]),
                "source": "comsol_embedded_optimizer",
                "comsol_objective_evaluation": index,
            }
        )
        records.append(record)
    observations = pd.DataFrame(records)
    feasible_mask = observations["feasible"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    )
    finite = observations.loc[
        feasible_mask, common.TARGET_COLUMN
    ].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise RuntimeError("COMSOL returned no finite feasible objective value.")
    best = observations.loc[finite.idxmax()].copy()
    return observations, best


def output_path(output_dir: Path, method: str, run_id: str) -> Path:
    return output_dir / method / run_id


def assert_output_available(
    output_dir: Path,
    methods: list[str],
    seeds: list[int],
    overwrite: bool,
    resume: bool,
) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    existing = [
        output_path(output_dir, method, f"seed_{seed}")
        for method in methods
        for seed in seeds
        if output_path(output_dir, method, f"seed_{seed}").exists()
    ]
    if existing and not (overwrite or resume):
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing COMSOL run directories. "
            "Pass --overwrite only after archiving the existing outputs:\n"
            + formatted
        )


def resumable_summary(
    run_dir: Path,
    method: str,
    seed: int,
    initial_size: int,
    continuation_budget: int,
    model_sha256: str,
) -> dict[str, Any] | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = {
        "method": method,
        "seed": seed,
        "initial_size": initial_size,
        "total_budget": initial_size + continuation_budget,
        "configured_continuation_attempt_budget": continuation_budget,
        "configured_nsolvemax": (
            continuation_budget - 1
            if method == "ego"
            else continuation_budget
        ),
        "optimizer_start_selection": (
            "highest-response shared feasible point with every normalized "
            f"box-bound margin >= {MIN_START_BOX_MARGIN:g}"
        ),
        "model_sha256": model_sha256,
        "design_space_fingerprint": py1.design_space_fingerprint(),
        "completed": True,
    }
    mismatches = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Cannot resume incompatible saved run {run_dir}: {mismatches}"
        )
    for required in ("observations.csv", "history.csv", "best_design.json"):
        if not (run_dir / required).exists():
            raise RuntimeError(
                f"Cannot resume incomplete saved run {run_dir}: missing {required}"
            )
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shared_initial_design(
    seed: int,
    initial_size: int,
    expected_total_budget: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    run_dir = PY7_DIR / f"seed_{seed}"
    metadata_path = run_dir / "metadata.json"
    observations_path = run_dir / "observations.csv"
    if not metadata_path.exists() or not observations_path.exists():
        raise FileNotFoundError(
            f"Missing completed py7 run for shared seed {seed}: {run_dir}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "method": "expected_improvement_bo",
        "mode": "online",
        "seed": seed,
        "initial_size": initial_size,
        "completed": True,
        "design_space_fingerprint": py1.design_space_fingerprint(),
    }
    if expected_total_budget is not None:
        expected["total_budget"] = expected_total_budget
    mismatches = {
        key: {"observed": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"py7 seed {seed} does not satisfy the shared-start contract: "
            f"{mismatches}"
        )
    source = pd.read_csv(observations_path)
    if len(source) < initial_size:
        raise ValueError(
            f"py7 seed {seed} contains only {len(source)} observations; "
            f"{initial_size} initial rows are required."
        )
    required = [common.ID_COLUMN, *common.FEATURE_COLUMNS, common.TARGET_COLUMN]
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"py7 seed {seed} is missing columns: {missing}")
    initial = source.iloc[:initial_size][required].copy().reset_index(drop=True)
    values = initial[[*common.FEATURE_COLUMNS, common.TARGET_COLUMN]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError(f"py7 seed {seed} initial design contains non-finite values.")
    if initial[common.ID_COLUMN].astype(str).duplicated().any():
        raise ValueError(f"py7 seed {seed} initial sample IDs are not unique.")

    canonical_records: list[dict[str, Any]] = []
    for row in initial.itertuples(index=False):
        unit = np.asarray(
            [getattr(row, name) for name in common.FEATURE_COLUMNS],
            dtype=float,
        )
        record = common.candidate_record(
            unit,
            str(getattr(row, common.ID_COLUMN)),
        )
        if not bool(record["feasible"]):
            raise ValueError(
                f"Shared initial point {record[common.ID_COLUMN]} for seed "
                f"{seed} is infeasible."
            )
        record[common.TARGET_COLUMN] = float(
            getattr(row, common.TARGET_COLUMN)
        )
        record["source"] = "shared_initial_design"
        record["comsol_objective_evaluation"] = np.nan
        canonical_records.append(record)
    canonical = pd.DataFrame(canonical_records)
    shared_best = canonical.loc[
        pd.to_numeric(canonical[common.TARGET_COLUMN], errors="coerce").idxmax()
    ].copy()
    unit_values = canonical[common.FEATURE_COLUMNS].to_numpy(float)
    box_margins = np.minimum(unit_values, 1.0 - unit_values).min(axis=1)
    interior = canonical.loc[box_margins >= MIN_START_BOX_MARGIN].copy()
    if interior.empty:
        raise ValueError(
            f"py7 seed {seed} has no shared initial point with normalized "
            f"box margin >= {MIN_START_BOX_MARGIN:g}; cannot construct a "
            "strictly interior common start for the COMSOL optimizers."
        )
    optimizer_start = interior.loc[
        pd.to_numeric(
            interior[common.TARGET_COLUMN], errors="coerce"
        ).idxmax()
    ].copy()
    optimizer_start["normalized_box_margin"] = float(
        box_margins[int(optimizer_start.name)]
    )
    return canonical, shared_best, optimizer_start, metadata


def write_result(
    *,
    output_dir: Path,
    run_id: str,
    method: str,
    seed: int,
    initial_observations: pd.DataFrame,
    optimizer_observations: pd.DataFrame,
    initial_best: pd.Series,
    optimizer_start: pd.Series,
    best: pd.Series,
    table_headers: list[str],
    reported_seconds: float,
    timing_line: str,
    comsol_version: str,
    model_file: Path,
    model_sha256: str,
    continuation_budget: int,
    configured_nsolvemax: int,
    progress_file: Path,
    orchestration_seconds: float,
    py7_metadata: dict[str, Any],
) -> dict[str, Any]:
    run_dir = output_path(output_dir, method, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    optimizer_attempts = int(len(optimizer_observations))
    finite_target_mask = np.isfinite(
        pd.to_numeric(
            optimizer_observations[common.TARGET_COLUMN], errors="coerce"
        ).to_numpy(float)
    )
    optimizer_fem_calls = int(finite_target_mask.sum())
    finite_rows = optimizer_observations.loc[finite_target_mask].copy()
    budget_rows = finite_rows.drop_duplicates(
        subset=common.FEATURE_COLUMNS, keep="first"
    ).copy()
    optimizer_unique_fem_designs = int(len(budget_rows))
    optimizer_repeat_evaluations = (
        optimizer_fem_calls - optimizer_unique_fem_designs
    )
    feasible_mask = optimizer_observations["feasible"].map(
        lambda value: str(value).strip().lower() in {"true", "1"}
    ).to_numpy(bool)
    feasible_optimizer_fem_calls = int((finite_target_mask & feasible_mask).sum())
    initial_size = int(len(initial_observations))
    total_budget = initial_size + int(continuation_budget)
    actual_fem_calls = initial_size + optimizer_fem_calls
    budget_accounted_fem_calls = initial_size + optimizer_unique_fem_designs
    if budget_accounted_fem_calls > total_budget:
        raise RuntimeError(
            "COMSOL produced more unique optimizer designs than the declared "
            f"budget: {budget_accounted_fem_calls} > {total_budget}."
        )
    observations = pd.concat(
        [initial_observations, optimizer_observations],
        ignore_index=True,
        sort=False,
    )
    best_target = float(best[common.TARGET_COLUMN])
    initial_best_target = float(initial_best[common.TARGET_COLUMN])
    budget_exhausted = optimizer_unique_fem_designs >= continuation_budget
    metadata = {
        "method": method,
        "mode": "online",
        "seed": seed,
        "run_id": run_id,
        "initial_size": initial_size,
        "total_budget": total_budget,
        "configured_nsolvemax": int(configured_nsolvemax),
        "configured_continuation_attempt_budget": int(continuation_budget),
        "optimizer_evaluation_attempts": optimizer_attempts,
        "optimizer_finite_fem_calls": optimizer_fem_calls,
        "optimizer_unique_fem_designs": optimizer_unique_fem_designs,
        "optimizer_repeat_evaluations": optimizer_repeat_evaluations,
        "optimizer_feasible_fem_calls": feasible_optimizer_fem_calls,
        "actual_fem_calls": actual_fem_calls,
        "budget_accounted_fem_calls": budget_accounted_fem_calls,
        "budget_exhausted": budget_exhausted,
        "completed": True,
        "completion_reason": (
            f"COMSOL {STUDY_TAG} completed after {optimizer_attempts} "
            f"optimizer attempts ({optimizer_fem_calls} finite FEM targets; "
            f"{optimizer_unique_fem_designs} unique designs; "
            f"cap {continuation_budget})"
        ),
        "endpoint_only": False,
        "convergence_history_available": True,
        "held_out_validation_size": int(
            py7_metadata.get("held_out_validation_size", 0)
        ),
        "validation_seed": py7_metadata.get("validation_seed"),
        "candidate_pool_size": None,
        "min_distance": None,
        "optimizer": f"COMSOL built-in {DISPLAY_NAMES[method]}",
        "optimizer_random_seed": seed if method == "ego" else None,
        "optimizer_gp_random_seed": seed if method == "ego" else None,
        "comsol_version": comsol_version,
        "model_file": str(model_file),
        "model_sha256": model_sha256,
        "source_model_write_policy": "read_only; never saved",
        "model_session_policy": (
            "fresh MPH load for this method-seed run; model removed after run"
        ),
        "shared_initial_source": str(
            PY7_DIR / f"seed_{seed}" / "observations.csv"
        ),
        "shared_initial_best_sample_id": str(initial_best[common.ID_COLUMN]),
        "shared_initial_best_response": initial_best_target,
        "optimizer_start_sample_id": str(optimizer_start[common.ID_COLUMN]),
        "optimizer_start_response": float(
            optimizer_start[common.TARGET_COLUMN]
        ),
        "optimizer_start_selection": (
            "highest-response shared feasible point with every normalized "
            f"box-bound margin >= {MIN_START_BOX_MARGIN:g}"
        ),
        "optimizer_start_normalized_box_margin": float(
            optimizer_start["normalized_box_margin"]
        ),
        "explicit_start_parameters_mm": {
            name: float(optimizer_start[column])
            for name, column in zip(
                py1.DESIGN_VARIABLE_NAMES, py1.PHYSICAL_COLUMNS
            )
        },
        "study_tag": STUDY_TAG,
        "optimization_feature_tag": OPTIMIZATION_TAG,
        "objective_table_tag": OBJECTIVE_TABLE_TAG,
        "objective_table_headers": table_headers,
        "comsol_reported_study_seconds": reported_seconds,
        "timing_definition": (
            "Optimization elapsed time derived from COMSOL's own displayed "
            "start/end timestamps in its progress/solver log; not Python wall-clock time"
        ),
        "trajectory_time_allocation": (
            "COMSOL exposes only total continuation time; intermediate finite "
            "objective rows are assigned a linear fraction solely for trajectory "
            "visualization"
        ),
        "comsol_timing_log_line": timing_line,
        "comsol_progress_log": None,
        "comsol_progress_log_persisted": False,
        "python_orchestration_wall_seconds": orchestration_seconds,
        "fem_call_count": actual_fem_calls,
        "fem_call_count_source": (
            f"{initial_size} shared initial FEM records plus "
            f"{optimizer_fem_calls} finite objective values in "
            f"{OBJECTIVE_TABLE_TAG}; {optimizer_repeat_evaluations} exact "
            "terminal/restart repeats are retained as solver overhead but "
            "do not consume an additional unique-design BO slot; blank "
            "constrained attempts excluded"
        ),
        "parameter_input_unit": "m",
        "target_unit": "T",
        "data_provenance": "Automatically extracted from COMSOL Java API",
        "design_space_definition": py1.design_space_definition(),
        "design_space_fingerprint": py1.design_space_fingerprint(),
    }

    observations.to_csv(run_dir / "observations.csv", index=False)
    history_records: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "phase": "initial_design",
            "cumulative_fem_calls": initial_size,
            "best_before": initial_best_target,
            "observed_target": np.nan,
            "best_so_far": initial_best_target,
            "gp_mean_before": np.nan,
            "gp_std_before": np.nan,
            "acquisition": np.nan,
            "gp_fit_seconds": 0.0,
            "acquisition_seconds": 0.0,
            "fem_evaluation_seconds": np.nan,
            "iteration_wall_seconds": 0.0,
            "cumulative_wall_seconds": 0.0,
            "endpoint_only": False,
        }
    ]
    incumbent = initial_best_target
    for number, (_, row) in enumerate(budget_rows.iterrows(), start=1):
        observed = float(row[common.TARGET_COLUMN])
        best_before = incumbent
        if bool(row["feasible"]):
            incumbent = max(incumbent, observed)
        history_records.append(
            {
                "iteration": number,
                "phase": "embedded_optimizer",
                "cumulative_fem_calls": initial_size + number,
                "best_before": best_before,
                "observed_target": observed,
                "best_so_far": incumbent,
                "gp_mean_before": np.nan,
                "gp_std_before": np.nan,
                "acquisition": np.nan,
                "gp_fit_seconds": 0.0,
                "acquisition_seconds": 0.0,
                "fem_evaluation_seconds": np.nan,
                "iteration_wall_seconds": np.nan,
                "cumulative_wall_seconds": (
                    reported_seconds
                    * number
                    / max(optimizer_unique_fem_designs, 1)
                ),
                "endpoint_only": False,
            }
        )
    history = pd.DataFrame(history_records)
    history.to_csv(run_dir / "iteration_history.csv", index=False)

    best_design = best.to_dict()
    best_design.update({"method": method, "mode": "online", "seed": seed})
    pd.DataFrame([best_design]).to_csv(run_dir / "best_design.csv", index=False)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    summary = {
        **metadata,
        "initial_best_response": initial_best_target,
        "final_best_response": best_target,
        "improvement": best_target - initial_best_target,
        "optimization_wall_seconds": reported_seconds,
        "total_fem_seconds": None,
        "total_gp_fit_seconds": 0.0,
        "total_acquisition_seconds": 0.0,
        "improvement_per_hour": None,
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return {
        "method": method,
        "run_id": run_id,
        "seed": seed,
        "optimizer": DISPLAY_NAMES[method],
        "actual_fem_calls": actual_fem_calls,
        "optimizer_evaluation_attempts": optimizer_attempts,
        "optimizer_finite_fem_calls": optimizer_fem_calls,
        "best_target_T": best_target,
        "comsol_reported_optimization_seconds": reported_seconds,
        "timing_source": "COMSOL progress/solver log",
        "output_directory": str(run_dir),
    }


def write_timing_table(output_dir: Path, new_rows: list[dict[str, Any]]) -> None:
    path = output_dir / "optimization_times.csv"
    columns = [
        "method",
        "run_id",
        "seed",
        "optimizer",
        "actual_fem_calls",
        "optimizer_evaluation_attempts",
        "optimizer_finite_fem_calls",
        "best_target_T",
        "comsol_reported_optimization_seconds",
        "timing_source",
        "output_directory",
    ]
    rows = list(new_rows)
    for summary_path in output_dir.glob("*/seed_*/run_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        method = str(summary["method"])
        rows.append(
            {
                "method": method,
                "run_id": str(summary["run_id"]),
                "seed": int(summary["seed"]),
                "optimizer": DISPLAY_NAMES[method],
                "actual_fem_calls": int(summary["actual_fem_calls"]),
                "optimizer_evaluation_attempts": int(
                    summary["optimizer_evaluation_attempts"]
                ),
                "optimizer_finite_fem_calls": int(
                    summary["optimizer_finite_fem_calls"]
                ),
                "best_target_T": float(summary["final_best_response"]),
                "comsol_reported_optimization_seconds": float(
                    summary["comsol_reported_study_seconds"]
                ),
                "timing_source": "COMSOL progress/solver log",
                "output_directory": str(summary_path.parent),
            }
        )
    combined = pd.DataFrame(rows, columns=columns)
    combined = combined.drop_duplicates(
        subset=["method", "seed"], keep="last"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["seed", "method"]).to_csv(path, index=False)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    model_file = args.model_file.resolve()
    output_dir = args.output_dir.resolve()
    methods = list(dict.fromkeys(args.methods))
    seeds = list(dict.fromkeys(args.seeds))
    if not model_file.exists():
        raise FileNotFoundError(f"COMSOL model file was not found: {model_file}")
    if args.initial_size <= 0:
        raise ValueError("--initial-size must be positive.")
    if args.continuation_budget <= 0:
        raise ValueError("--continuation-budget must be positive.")
    assert_output_available(
        output_dir, methods, seeds, args.overwrite, args.resume
    )
    model_hash_before = file_sha256(model_file)

    client: Any | None = None
    model: Any | None = None
    summaries: list[dict[str, Any]] = []
    try:
        client = mph.start(version=args.comsol_version)
        model_util = jpype.JClass("com.comsol.model.util.ModelUtil")

        print(f"COMSOL model: {model_file}")
        print(f"Source model SHA-256: {model_hash_before}")
        print(
            "Shared protocol: "
            f"{args.initial_size} initial FEM points + at most "
            f"{args.continuation_budget} optimizer attempts."
        )
        for seed in seeds:
            (
                initial,
                initial_best,
                optimizer_start,
                py7_metadata,
            ) = load_shared_initial_design(
                seed,
                args.initial_size,
                args.initial_size + args.continuation_budget,
            )
            for method in methods:
                run_id = f"seed_{seed}"
                run_dir = output_path(output_dir, method, run_id)
                if args.resume:
                    saved = resumable_summary(
                        run_dir=run_dir,
                        method=method,
                        seed=seed,
                        initial_size=args.initial_size,
                        continuation_budget=args.continuation_budget,
                        model_sha256=model_hash_before,
                    )
                    if saved is not None:
                        summaries.append(saved)
                        print(
                            f"Skipping completed {DISPLAY_NAMES[method]}, "
                            f"seed {seed}."
                        )
                        continue
                run_dir.mkdir(parents=True, exist_ok=True)
                temporary_progress = (
                    Path(tempfile.gettempdir())
                    / (
                        f"py8_comsol_{method}_seed{seed}_"
                        f"{uuid.uuid4().hex}.tmp"
                    )
                )
                try:
                    # Each method-seed pair receives a fresh model object loaded
                    # from the unchanged source MPH. No optimizer can inherit an
                    # in-memory terminal design from an earlier run.
                    model = client.load(str(model_file))
                    java = model.java
                    optimization = validate_model(java)
                    comsol_version = str(model.version())
                    for parameter, column in zip(
                        py1.DESIGN_VARIABLE_NAMES, py1.PHYSICAL_COLUMNS
                    ):
                        model.parameter(
                            parameter,
                            f"{float(optimizer_start[column]):.12g}[mm]",
                        )
                    start_expressions = [
                        f"{float(optimizer_start[column]):.12g}[mm]"
                        for column in py1.PHYSICAL_COLUMNS
                    ]
                    optimization.set(
                        "initval",
                        jpype.JArray(jpype.JString)(start_expressions),
                    )
                    observed_start = [
                        token.strip().replace(" ", "")
                        for token in java_property(
                            optimization, "initval"
                        ).split(",")
                    ]
                    expected_start = [
                        value.replace(" ", "") for value in start_expressions
                    ]
                    if observed_start != expected_start:
                        raise RuntimeError(
                            "COMSOL rejected the explicit shared start: "
                            f"observed={observed_start}, "
                            f"expected={expected_start}"
                        )
                    java.result().table(OBJECTIVE_TABLE_TAG).clearTableData()
                    optimization.set("optsolver", METHODS[method])
                    configured_nsolvemax = (
                        args.continuation_budget - 1
                        if method == "ego"
                        else args.continuation_budget
                    )
                    if configured_nsolvemax <= 0:
                        raise ValueError(
                            "EGO requires --continuation-budget >= 3."
                        )
                    optimization.set(
                        "nsolvemax", str(int(configured_nsolvemax))
                    )
                    if method == "ego":
                        optimization.set("useseed", "on")
                        optimization.set("randseed", str(seed))
                        optimization.set("useseedgp", "manual")
                        optimization.set("randseedgp", str(seed))
                        observed_random_seed = int(
                            float(java_property(optimization, "randseed"))
                        )
                        observed_gp_seed = int(
                            float(java_property(optimization, "randseedgp"))
                        )
                        if (
                            observed_random_seed != seed
                            or observed_gp_seed != seed
                        ):
                            raise RuntimeError(
                                "COMSOL EGO random seeds were not applied: "
                                f"optimizer={observed_random_seed}, "
                                f"gp={observed_gp_seed}, expected={seed}"
                            )
                    optimization.set("plot", "off")
                    optimization.active(True)
                    orchestration_start = time.perf_counter()
                    try:
                        model_util.showProgress(str(temporary_progress))
                        print(
                            f"Running {DISPLAY_NAMES[method]}, seed {seed}, "
                            f"start={optimizer_start[common.ID_COLUMN]}..."
                        )
                        java.study(STUDY_TAG).run()
                    finally:
                        optimization.active(False)
                        model_util.showProgress(False)
                    orchestration_seconds = (
                        time.perf_counter() - orchestration_start
                    )

                    if not temporary_progress.exists():
                        raise RuntimeError(
                            "COMSOL did not create the requested progress log: "
                            f"{temporary_progress}"
                        )
                    table_headers, rows = objective_table(java)
                    reported_evaluations = parse_comsol_evaluation_count(
                        temporary_progress
                    )
                    if len(rows) < reported_evaluations:
                        raise RuntimeError(
                            f"COMSOL reported {reported_evaluations} objective "
                            f"evaluations, but {OBJECTIVE_TABLE_TAG} contains "
                            f"only {len(rows)} rows."
                        )
                    rows = rows[-reported_evaluations:]
                    optimizer_observations, optimizer_best = (
                        canonical_observations(method, seed, rows)
                    )
                    eligible = pd.concat(
                        [
                            pd.DataFrame([initial_best]),
                            pd.DataFrame([optimizer_best]),
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                    best = eligible.loc[
                        pd.to_numeric(
                            eligible[common.TARGET_COLUMN], errors="coerce"
                        ).idxmax()
                    ].copy()
                    reported_seconds, timing_line = (
                        parse_comsol_reported_seconds(temporary_progress)
                    )
                    summary = write_result(
                        output_dir=output_dir,
                        run_id=run_id,
                        method=method,
                        seed=seed,
                        initial_observations=initial,
                        optimizer_observations=optimizer_observations,
                        initial_best=initial_best,
                        optimizer_start=optimizer_start,
                        best=best,
                        table_headers=table_headers,
                        reported_seconds=reported_seconds,
                        timing_line=timing_line,
                        comsol_version=comsol_version,
                        model_file=model_file,
                        model_sha256=model_hash_before,
                        continuation_budget=args.continuation_budget,
                        configured_nsolvemax=configured_nsolvemax,
                        progress_file=temporary_progress,
                        orchestration_seconds=orchestration_seconds,
                        py7_metadata=py7_metadata,
                    )
                    summaries.append(summary)
                    write_timing_table(output_dir, [summary])
                    print(
                        f"Saved {DISPLAY_NAMES[method]}, seed {seed}: "
                        f"{reported_seconds:g} s, "
                        f"{summary['optimizer_evaluation_attempts']} attempts, "
                        f"{summary['optimizer_finite_fem_calls']} finite "
                        "optimizer FEM targets."
                    )
                finally:
                    if model is not None:
                        try:
                            client.remove(model)
                        except Exception:
                            pass
                        model = None
                    try:
                        temporary_progress.unlink(missing_ok=True)
                    except PermissionError:
                        pass
    finally:
        if client is not None and model is not None:
            try:
                client.remove(model)
            except Exception:
                pass
        if client is not None:
            try:
                client.clear()
            except Exception:
                pass
        print(f"Source MPH was not saved: {model_file}")

    model_hash_after = file_sha256(model_file)
    if model_hash_after != model_hash_before:
        raise RuntimeError(
            "Source MPH changed during py8 despite the read-only policy: "
            f"{model_hash_before} -> {model_hash_after}"
        )
    return summaries


def main() -> None:
    args = parse_args()
    summaries = run(args)
    print(f"Timing table: {args.output_dir.resolve() / 'optimization_times.csv'}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
