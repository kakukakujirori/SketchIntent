"""Reference-aware constraint sub-type classification (diagnostic logging).

Several SketchGraphs/Onshape constraint *tokens* are polymorphic: their meaning
depends on the entity types of the references, not on the head token.  The most
design-intent-relevant splits are:

- ``Coincident`` -> point-point (polyline stitching, low intent) /
  point-on-object (PointOnObject) / collinear (line-line);
- ``Equal``      -> equal length (line-line) / equal radius (circle/arc);
- ``Tangent``    -> tangent at a shared point / edge-edge tangency.

The token-level :mod:`src.metrics.constraint_stats` cannot see these (it only
counts head tokens).  This module classifies a *reconstructed* ``datalib.Sketch``
by resolving each constraint's references to their entity types, mirroring the
branching already used in the constraint handlers' ``residual`` methods
(``coincident.py`` / ``equal.py``).  It is used for validation-time logging so we
can watch whether the policy produces the high-intent sub-cases (e.g.
PointOnObject, equal-radius) before deciding to add a sub-type-aware reward.
"""
from __future__ import annotations

from collections import Counter

import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter


# Stable label order for charts.
CONSTRAINT_SUBTYPE_LABELS: tuple[str, ...] = (
    "Coincident.point_point",
    "Coincident.point_on_object",
    "Coincident.collinear",
    "Equal.length",
    "Equal.radius",
    "Tangent.point",
    "Tangent.edge",
)

_POLYMORPHIC_TYPES = frozenset({"Coincident", "Equal", "Tangent"})


def _ref_kind(sketch: datalib.Sketch, ref: str) -> tuple[str, str]:
    """Return ``(role, entity_type)`` for one reference string.

    ``role`` is ``"edge"`` when the reference is a bare entity id pointing at a
    non-Point edge (Line/Circle/Arc), else ``"pt"`` (a point entity or a
    ``eid.subnode`` point reference).  ``entity_type`` is the class name of the
    referenced entity (``Line``/``Circle``/``Arc``/``Point``/``None``).
    """
    eid = ref.split(".")[0]
    entity = sketch.entities.get(eid)
    is_edge = ("." not in ref) and not isinstance(entity, datalib.Point)
    entity_type = type(entity).__name__ if entity is not None else "None"
    return ("edge" if is_edge else "pt"), entity_type


def classify_constraint_subtype(
    constr: datalib.Constraint, sketch: datalib.Sketch
) -> str | None:
    """Return a ``"<Type>.<subtype>"`` label, or ``None`` for non-polymorphic types."""
    type_name = getattr(constr.type, "name", None)
    if type_name not in _POLYMORPHIC_TYPES:
        return None
    refs = [p.value for p in constr.parameters if isinstance(p, LocalReferenceParameter)]
    if len(refs) < 2:
        return None
    (k1, e1), (k2, e2) = _ref_kind(sketch, refs[0]), _ref_kind(sketch, refs[1])
    sig = tuple(sorted((k1, k2)))

    if type_name == "Coincident":
        if sig == ("edge", "edge"):
            return "Coincident.collinear"
        if sig == ("edge", "pt"):
            return "Coincident.point_on_object"
        return "Coincident.point_point"
    if type_name == "Equal":
        # Equal is edge-edge; radius when any side is a circle/arc, else length.
        if {e1, e2} & {"Circle", "Arc"}:
            return "Equal.radius"
        return "Equal.length"
    # Tangent
    if sig == ("edge", "edge"):
        return "Tangent.edge"
    return "Tangent.point"


def constraint_subtype_counts(sketch: datalib.Sketch) -> Counter:
    """Count the polymorphic constraint sub-types in one reconstructed sketch."""
    counts: Counter = Counter()
    for constr in sketch.constraints.values():
        label = classify_constraint_subtype(constr, sketch)
        if label is not None:
            counts[label] += 1
    return counts


def subtype_base(label: str) -> str:
    """Return the base constraint-type name of a sub-type ``label``.

    ``"Coincident.point_on_object" -> "Coincident"``; a bare type name maps to
    itself.  Used to decide whether a sub-type weighting key belongs to a
    polymorphic base and to share the base-type "key relation" flag.
    """
    return label.split(".", 1)[0]


def classify_constraint_label(
    constr: datalib.Constraint, sketch: datalib.Sketch
) -> str:
    """Return the *effective key* one constraint maps to for sub-type weighting.

    This is the single source of truth shared by the relational class-prior
    reward and any sub-type-aware logging: the polymorphic sub-type label when
    the constraint is one of the polymorphic types and its references resolve
    (e.g. ``"Coincident.point_on_object"``, ``"Equal.radius"``), otherwise the
    plain head-token name (``"Parallel"``, or ``"Coincident"`` when a polymorphic
    constraint cannot be classified — e.g. fewer than two references).
    """
    sub = classify_constraint_subtype(constr, sketch)
    if sub is not None:
        return sub
    return getattr(constr.type, "name", "Unknown")
