"""Persist a read-only snapshot of manuscript-relevant COMSOL model settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mph


ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parents[1] / "surrogate_simulation" / "ML surrogate.mph"
OUTPUT = ROOT / "py18_output" / "model_contract_snapshot.json"


def safe(call, default=None):
    try:
        return call()
    except Exception:
        return default


def props(node: Any) -> dict[str, str]:
    return {str(name): str(safe(lambda name=name: node.getString(name), "<unavailable>")) for name in safe(lambda: list(node.properties()), [])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    before = sha256(MODEL)
    client = mph.start(version="6.3")
    model = client.load(str(MODEL))
    try:
        java = model.java
        component = java.component("comp1")
        geometry = component.geom("geom1")
        datasets = {}
        for tag in list(java.result().dataset().tags()):
            node = java.result().dataset(tag)
            datasets[str(tag)] = {"type": str(node.getType()), "properties": props(node), "entities": safe(lambda node=node: list(node.selection().entities()))}
        numericals = {}
        for tag in list(java.result().numerical().tags()):
            node = java.result().numerical(tag)
            numericals[str(tag)] = {"type": str(node.getType()), "properties": props(node), "entities": safe(lambda node=node: list(node.selection().entities()))}
        probes = {}
        probe_collection = safe(lambda: component.probe())
        if probe_collection is not None:
            for tag in list(safe(lambda: probe_collection.tags(), [])):
                node = component.probe(tag)
                probes[str(tag)] = {"type": str(node.getType()), "properties": props(node), "entities": safe(lambda node=node: list(node.selection().entities()))}
        geom_features = {}
        for tag in list(geometry.feature().tags()):
            node = geometry.feature(tag)
            values = props(node)
            if any(key in values for key in ("p", "coord", "pos", "size", "x", "y")) or str(node.getType()).lower() in {"square", "linesegment"}:
                geom_features[str(tag)] = {"type": str(node.getType()), "properties": values}
        physics = {}
        magnetic = component.physics("mf")
        for tag in list(magnetic.feature().tags()):
            node = magnetic.feature(tag)
            if str(tag) in {"coil1", "als1", "als2", "mi1"}:
                physics[str(tag)] = {"type": str(node.getType()), "properties": props(node), "entities": safe(lambda node=node: list(node.selection().entities()))}
        material = component.material("mat3").propertyGroup("BHCurve").func("BH")
        bh = [[str(value) for value in row] for row in material.getStringMatrix("table")]
        snapshot = {
            "source_sha256_before": before,
            "parameters": {name: str(java.param().get(name)) for name in ("c1", "r1", "t1", "l1", "l2", "w1", "h1", "Japply")},
            "geometry_features": geom_features,
            "physics": physics,
            "datasets": datasets,
            "numericals": numericals,
            "probes": probes,
            "bh_table_rows": len(bh),
            "bh_first_rows": bh[:3],
            "bh_last_rows": bh[-3:],
            "source_model_saved": False,
        }
    finally:
        client.remove(model); client.clear()
    snapshot["source_sha256_after"] = sha256(MODEL)
    if snapshot["source_sha256_after"] != before:
        raise RuntimeError("Source MPH hash changed during contract snapshot.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "datasets": {key: value["type"] for key, value in snapshot["datasets"].items()},
        "numericals": {key: {"type": value["type"], "data": value["properties"].get("data")} for key, value in snapshot["numericals"].items()},
        "probes": list(snapshot["probes"]),
        "bh_table_rows": snapshot["bh_table_rows"],
        "source_hash_unchanged": snapshot["source_sha256_before"] == snapshot["source_sha256_after"],
    }, indent=2))


if __name__ == "__main__":
    main()
