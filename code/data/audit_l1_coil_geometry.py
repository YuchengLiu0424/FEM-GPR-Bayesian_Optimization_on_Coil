"""Diagnose whether short ``l1`` values disrupt the COMSOL coil domain.

Each case is loaded into a fresh in-memory model, and the source MPH is never
saved.  The valid cases bracket the axial-fit boundary
``l1 = h1 + coil_bottom_z``; the deliberately invalid case reproduces the
geometric overlap caused by an excessively short lower-core section.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from py11_time_initial_fem_samples import configure_stationary_point_evaluation
from py2_high_fidelity_evaluation_engine import ComsolEvaluator, scalar_probe_value


BASELINE_MM = {
    "c1": 2.5,
    "r1": 16.0,
    "t1": 5.0,
    "l2": 10.0,
    "w1": 5.0,
}

CASES = (
    ("default_current", 10.0, 5.0, 1.0e6),
    ("default_zero_current", 10.0, 5.0, 0.0),
    ("exact_axial_fit", 7.0, 5.0, 1.0e6),
    ("half_mm_clearance", 7.5, 5.0, 1.0e6),
    ("invalid_short_l1", 1.62, 3.0, 1.0e6),
)


def safe(callable_: Any, default: Any = "<unavailable>") -> Any:
    try:
        return callable_()
    except Exception:
        return default


def domain_integral(model: Any, domain: int) -> float:
    """Integrate unity over one 2-D domain using a temporary result node."""
    numericals = model.java.result().numerical()
    tag = "int_l1_audit"
    if tag in list(numericals.tags()):
        numericals.remove(tag)
    node = numericals.create(tag, "IntSurface")
    try:
        node.selection().set(domain)
        node.set("expr", "1")
        values = np.asarray(
            [list(row) for row in node.getReal()],
            dtype=float,
        ).reshape(-1)
        if values.size != 1:
            raise RuntimeError(f"Unexpected domain-integral result: {values!r}")
        return float(values[0])
    finally:
        numericals.remove(tag)


def run_case(name: str, l1_mm: float, h1_mm: float, japply: float) -> None:
    with ComsolEvaluator() as evaluator:
        if evaluator.model is None:
            raise RuntimeError("COMSOL evaluator did not load the model.")
        model = evaluator.model
        configure_stationary_point_evaluation(evaluator)
        for parameter, value in BASELINE_MM.items():
            model.parameter(parameter, f"{value:.12g}[mm]")
        model.parameter("l1", f"{l1_mm:.12g}[mm]")
        model.parameter("h1", f"{h1_mm:.12g}[mm]")
        model.parameter("Japply", f"{japply:.12g}[A/m^2]")

        model.build()
        component = model.java.component("comp1")
        geometry = component.geom("geom1")
        coil = component.physics("mf").feature("coil1")
        copper = component.material("mat1")
        core = component.material("mat3")

        geometry_summary = {
            "domains": safe(lambda: geometry.getNDom()),
            "domains_alternative": safe(lambda: geometry.getNDomains()),
            "coil_domains": safe(lambda: list(coil.selection().entities())),
            "copper_domains": safe(lambda: list(copper.selection().entities())),
            "core_domains": safe(lambda: list(core.selection().entities())),
        }
        model.mesh()
        model.solve()
        coil_domain_integral = domain_integral(model, 3)
        print(
            {
                "case": name,
                "l1_mm": l1_mm,
                "h1_mm": h1_mm,
                "axial_fit_margin_mm": l1_mm - h1_mm - 2.0,
                "Japply_A_per_m2": japply,
                "target_T": scalar_probe_value(model),
                "coil_domain_integral": coil_domain_integral,
                **geometry_summary,
            }
        )


def main() -> None:
    for case in CASES:
        try:
            run_case(*case)
        except Exception as exc:
            print(
                {
                    "case": case[0],
                    "l1_mm": case[1],
                    "h1_mm": case[2],
                    "axial_fit_margin_mm": case[1] - case[2] - 2.0,
                    "Japply_A_per_m2": case[3],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    print("SOURCE_MPH_SAVED False")


if __name__ == "__main__":
    main()
