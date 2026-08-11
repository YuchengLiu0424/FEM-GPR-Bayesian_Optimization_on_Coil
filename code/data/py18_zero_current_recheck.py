"""Fresh-session recheck of the stored-geometry zero-current manuscript claim.

Each case reloads the source MPH, rebuilds, meshes, solves, and closes without
saving.  The output records source hashes before and after the two solves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from py2_high_fidelity_evaluation_engine import ComsolEvaluator, MODEL_FILE, scalar_probe_value


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "py18_output"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(japply: float) -> dict:
    with ComsolEvaluator() as evaluator:
        model = evaluator.model
        if model is None:
            raise RuntimeError("COMSOL model was not loaded.")
        stored_geometry = {
            name: str(model.java.param().get(name))
            for name in ("c1", "r1", "t1", "l1", "l2", "w1", "h1")
        }
        model.parameter("Japply", f"{japply:.12g}[A/m^2]")
        model.build(); model.mesh(); model.solve()
        return {
            "Japply_A_per_m2": japply,
            "target_T": scalar_probe_value(model),
            "stored_geometry_parameters": stored_geometry,
            "coil_current_expression": str(model.java.component("comp1").physics("mf").feature("coil1").getString("ICoil")),
            "coil_domains": list(model.java.component("comp1").physics("mf").feature("coil1").selection().entities()),
            "probe_expression": str(model.java.result().numerical("pev1").getString("expr")),
        }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = sha256(MODEL_FILE)
    cases = [run_case(0.0), run_case(1e6)]
    after = sha256(MODEL_FILE)
    if before != after:
        raise RuntimeError("Source MPH hash changed during zero-current audit.")
    report = {
        "fresh_model_session_per_case": True,
        "source_model_saved": False,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "cases": cases,
    }
    (OUTPUT / "zero_current_recheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
