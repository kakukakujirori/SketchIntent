"""Per-annotation feature extraction and display-type helpers."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

from .config import DIMENSIONAL_DISPLAY_TYPES, REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sketchgraphs.data as datalib  # noqa: E402
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

EXCLUDED_TYPES = {"Subnode", "Unknown"}

_CTX = None
_CONVERTER = None


def _converter() -> SketchFreecadConverter:
    global _CTX, _CONVERTER
    if _CONVERTER is None:
        _CTX = SketchContext()
        _CONVERTER = SketchFreecadConverter(_CTX, verbose=False, raise_on_error=False)
    return _CONVERTER


def ref_strings(constraint) -> list[str]:
    return [
        str(p.value)
        for p in constraint.parameters
        if isinstance(p, datalib.LocalReferenceParameter)
    ]


def ref_kind(ref: str, entities: dict) -> str:
    if "." in ref:
        return "pt"
    if ref.startswith("-"):
        return "edge"
    ent = entities.get(ref)
    return "pt" if isinstance(ent, datalib.Point) else "edge"


def display_type(constraint, entities: dict) -> str:
    """Canonical display name used by dataset figures and per-type F1."""
    name = constraint.constraint_type.name
    if name == "Distance":
        for p in constraint.parameters:
            if getattr(p, "parameterId", None) == "direction":
                direction = str(getattr(p, "value", ""))
                if direction == "HORIZONTAL":
                    return "DistanceX"
                if direction == "VERTICAL":
                    return "DistanceY"
        return "Distance"
    if name == "Coincident":
        kinds = sorted(ref_kind(r, entities) for r in ref_strings(constraint))
        if kinds == ["edge", "pt"]:
            return "PointOnObject"
        if kinds == ["edge", "edge"]:
            return "Collinear"
        return "Coincident"
    return name


def extract_features(filepath: str | Path) -> dict:
    """Load one annotated FCStd and return the JSON-record feature dict."""
    FreeCAD, _, _ = ensure_freecad_imported()
    doc, fc = load_FCStd_sketch(str(filepath))
    try:
        dof = int(fc.DoF)
        fc_counts: dict[str, int] = collections.Counter()
        for c in fc.Constraints:
            fc_counts[c.Type] += 1

        sk = _converter().from_freecad(fc)

        ent_counts: dict[str, int] = collections.Counter()
        construction_counts: dict[str, int] = collections.Counter()
        for e in sk.entities.values():
            tname = type(e).__name__
            if getattr(e, "isConstruction", False):
                construction_counts[tname] += 1
            else:
                ent_counts[tname] += 1

        con_counts: dict[str, int] = collections.Counter()
        for c in sk.constraints.values():
            if c.constraint_type.name in EXCLUDED_TYPES:
                continue
            con_counts[display_type(c, sk.entities)] += 1

        n_dimensional = sum(
            v for k, v in con_counts.items() if k in DIMENSIONAL_DISPLAY_TYPES
        )
        n_relational = sum(
            v for k, v in con_counts.items() if k not in DIMENSIONAL_DISPLAY_TYPES
        )
        return {
            "filepath": str(filepath),
            "dof": dof,
            "n_primitives": sum(ent_counts.values()),
            "n_construction": sum(construction_counts.values()),
            "primitive_counts": dict(ent_counts),
            "construction_counts": dict(construction_counts),
            "n_constraints": sum(con_counts.values()),
            "n_relational": n_relational,
            "n_dimensional": n_dimensional,
            "constraint_counts": dict(con_counts),
            "fc_constraint_counts": dict(fc_counts),
            "n_block": fc_counts.get("Block", 0),
        }
    finally:
        FreeCAD.closeDocument(doc.Name)
