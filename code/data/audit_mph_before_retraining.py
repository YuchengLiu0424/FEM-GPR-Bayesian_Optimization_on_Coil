"""Read-only audit of the COMSOL model before regenerating FEM data.

The script intentionally never calls ``model.save``. It prints the parameter,
geometry, material, physics, and embedded-optimization settings needed to
check that the Python and COMSOL feasible regions are identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mph


MODEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "surrogate_simulation"
    / "ML surrogate.mph"
)


def safe(callable_: Any, default: Any = "<unavailable>") -> Any:
    try:
        return callable_()
    except Exception:
        return default


def properties(feature: Any) -> dict[str, str]:
    names = safe(lambda: list(feature.properties()), [])
    return {str(name): str(safe(lambda name=name: feature.getString(name))) for name in names}


def selected(props: dict[str, str], needles: tuple[str, ...]) -> dict[str, str]:
    return {
        key: value
        for key, value in props.items()
        if any(needle.lower() in key.lower() for needle in needles)
    }


def main() -> None:
    client = None
    model = None
    try:
        client = mph.start(version="6.3")
        model = client.load(str(MODEL_FILE))
        java = model.java

        print("MODEL", MODEL_FILE)
        print("PARAMETERS")
        for name in ("c1", "r1", "t1", "l1", "l2", "w1", "h1", "Japply"):
            print(name, safe(lambda name=name: java.param().get(name)))

        component = java.component("comp1")
        geometry = component.geom("geom1")
        print("GEOMETRY")
        for tag in list(geometry.feature().tags()):
            feature = geometry.feature(tag)
            props = properties(feature)
            focus = selected(
                props,
                ("pos", "size", "base", "axis", "selresult", "contributeto"),
            )
            print(tag, safe(feature.label), safe(feature.getType), focus)

        print("MATERIALS")
        for tag in list(component.material().tags()):
            material = component.material(tag)
            print(tag, safe(material.label), "domains", safe(material.selection().entities))
            for group_tag in list(safe(lambda: material.propertyGroup().tags(), [])):
                group = material.propertyGroup(group_tag)
                props = properties(group)
                function_tags = list(safe(lambda: group.func().tags(), []))
                print(" ", group_tag, safe(group.label), "functions", function_tags)
                focus = selected(
                    props,
                    (
                        "ConstitutiveRelation",
                        "remanent",
                        "Br",
                        "BH",
                        "HB",
                        "mur",
                        "normB",
                        "normH",
                        "Wpm",
                        "density",
                        "electricconductivity",
                    ),
                )
                if focus:
                    print("  properties", focus)
                for function_tag in function_tags:
                    function = group.func(function_tag)
                    print(
                        "  function",
                        function_tag,
                        safe(function.label),
                        selected(
                            properties(function),
                            (
                                "funcname",
                                "table",
                                "source",
                                "filename",
                                "extrap",
                                "argunit",
                                "fununit",
                            ),
                        ),
                    )
                    table_data = safe(
                        lambda: function.getStringMatrix("table"),
                        [],
                    )
                    table_rows = [
                        [str(value) for value in row]
                        for row in table_data
                    ]
                    print("   table_rows", len(table_rows))
                    print("   table_first", table_rows[:5])
                    print("   table_last", table_rows[-5:])
                    if function_tag == "BH":
                        for field_strength in (0, 100, 1000, 10000, 100000, 1000000):
                            print(
                                "   BH_sample",
                                field_strength,
                                safe(
                                    lambda field_strength=field_strength: model.evaluate(
                                        f"BH({field_strength}[A/m])"
                                    )
                                ),
                            )

        print("MAGNETIC_PHYSICS")
        magnetic = component.physics("mf")
        for tag in list(magnetic.feature().tags()):
            feature = magnetic.feature(tag)
            props = properties(feature)
            focus = selected(
                props,
                (
                    "ConductorModel",
                    "ICoil",
                    "N",
                    "J",
                    "ConstitutiveRelation",
                    "Br",
                    "mur",
                    "sigma",
                ),
            )
            print(
                tag,
                safe(feature.label),
                safe(feature.getType),
                "domains",
                safe(feature.selection().entities),
                focus,
            )

        print("OPTIMIZATION")
        optimization = java.study("std1").feature("opt")
        print(
            "opt",
            selected(
                properties(optimization),
                (
                    "solver",
                    "nsolve",
                    "objective",
                    "constraint",
                    "control",
                    "parameter",
                    "lower",
                    "upper",
                    "pname",
                    "initval",
                    "lbound",
                    "ubound",
                    "scale",
                ),
            ),
        )
        for tag in list(safe(lambda: optimization.feature().tags(), [])):
            feature = optimization.feature(tag)
            print(
                tag,
                safe(feature.label),
                safe(feature.getType),
                selected(
                    properties(feature),
                    (
                        "name",
                        "expr",
                        "initial",
                        "lower",
                        "upper",
                        "constraint",
                        "scale",
                    ),
                ),
            )

        print("RESULT_NUMERICALS")
        print("RESULT_DATASETS")
        for tag in list(java.result().dataset().tags()):
            dataset = java.result().dataset(tag)
            print(
                tag,
                safe(dataset.label),
                safe(dataset.getType),
                selected(
                    properties(dataset),
                    ("data", "selection", "expr", "axis", "method"),
                ),
            )
        for tag in list(java.result().numerical().tags()):
            numerical = java.result().numerical(tag)
            print(
                tag,
                safe(numerical.label),
                safe(numerical.getType),
                "entities",
                safe(numerical.selection().entities),
                selected(properties(numerical), ("expr", "unit", "data", "selection")),
            )
    finally:
        if client is not None and model is not None:
            safe(lambda: client.remove(model))
        if client is not None:
            safe(client.clear)
        print("SOURCE_MPH_SAVED False")


if __name__ == "__main__":
    main()
