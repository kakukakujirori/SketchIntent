"""Synthetic intrinsic-size probes used by reward computations.

Builds one probe per primitive whose intrinsic size is a natural driving
parameter (Line → Length, Arc/Circle → Radius). Construction geometry and
Points are skipped.
"""
from __future__ import annotations

import sketchgraphs.data as datalib
from sketchgraphs.data import (
    Arc,
    Circle,
    Constraint,
    ConstraintType,
    EnumParameter,
    Line,
    LocalReferenceParameter,
    QuantityParameter,
)

from src.sketch_handlers.sketch_utils import complete_constraint_values


def make_raw_probe(
    ctype_name: str, refs: tuple[str, ...]
) -> datalib.Constraint:
    """Construct a synthetic dim constraint with an un-watermarked placeholder
    value.  ``complete_constraint_values`` is responsible for filling in the
    real value computed from primitive geometry and adding the watermark
    required by ``ConstraintHandler.get_constraint_props``."""
    ctype = getattr(ConstraintType, ctype_name)
    params: list = [
        LocalReferenceParameter(parameterId=f"local{i}", value=ref)
        for i, ref in enumerate(refs)
    ]
    pid = "angle" if ctype_name == "Angle" else "length"
    params.append(QuantityParameter(
        parameterId=pid, value=0.0, expression="0.0"))
    if ctype_name == "Length":
        # Required by complete_constraint_values → compute_line_length.
        params.append(EnumParameter(parameterId="direction", value="MINIMUM"))
    return Constraint(
        identifier=f"synthetic_{ctype_name}",
        constraint_type=ctype,
        parameters=params,
    )


def is_construction_entity(entity) -> bool:
    return bool(
        getattr(entity, "isConstruction", False)
        or getattr(entity, "Construction", False)
    )


def build_intrinsic_probes(sketch: datalib.Sketch) -> list[datalib.Constraint]:
    """One probe per primitive whose intrinsic size is a natural driving param.

    - Line  → Length (= norm(end - start))
    - Arc   → Radius
    - Circle→ Radius
    - Point → skipped (no intrinsic size)
    - Construction geometry → skipped (auxiliary, not user-facing edit intent)

    All probes have a single-dimensional length residual, so row units are
    homogeneous and downstream gain mean-square aggregation stays
    scale-consistent.  Probe values are filled in by
    ``complete_constraint_values`` on a temporary sketch that shares the
    source sketch's entities; this follows the project protocol that
    ``param.expression`` must carry the ``VALUE_ASSIGNED:`` watermark before
    ``ConstraintHandler.residual`` can be invoked.
    """
    raw: dict[str, datalib.Constraint] = {}
    for eid, ent in sketch.entities.items():
        if is_construction_entity(ent):
            continue
        if isinstance(ent, Line):
            raw[f"_probe_L_{eid}"] = make_raw_probe("Length", (eid,))
        elif isinstance(ent, (Arc, Circle)):
            raw[f"_probe_R_{eid}"] = make_raw_probe("Radius", (eid,))
    if not raw:
        return []
    temp = datalib.Sketch(entities=sketch.entities, constraints=raw)
    temp, _ = complete_constraint_values(temp, skip_invalid_constraints=True)
    return list(temp.constraints.values())
