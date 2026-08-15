"""Strict matching of dimensional drivers shared by two annotations.

Identity is ``constraint_signature`` (type, canonicalised refs, direction) — the same
canonicalisation the existing metrics use. No approximate matching: a signature that is
not unique within one annotation is dropped rather than guessed at.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import sketchgraphs.data as datalib
from sketchgraphs.data import QuantityParameter

from src.metrics.edit_response_distance import _is_dimensional, constraint_signature
from src.metrics.histcad_2d import _evaluate_observable
from src.sketch_handlers.sketch_solver import SketchState

QUANTITY_KINDS = {"length": "length", "angle": "angle"}
POSITIVE_QUANTITY_TYPES = frozenset(
    {"Radius", "Diameter", "Minor_Diameter", "Major_Diameter", "Length", "Distance"}
)


@dataclass(frozen=True)
class DriverSlot:
    cid: str
    kind: str
    q0: float
    baseline_residual: float = 0.0
    """q0 minus the constraint's stored target. Non-zero means the solver did not honour
    this datum (or the handler and FreeCAD disagree on a convention, e.g. an angle read
    2*pi away). Edits stay self-consistent because the same observable defines the target,
    so this is reported rather than used as a filter."""


@dataclass(frozen=True)
class DriverMatch:
    signature: tuple
    driver_type: str
    kind: str
    slot_p: DriverSlot
    slot_g: DriverSlot

    @property
    def requires_positive_target(self) -> bool:
        return self.driver_type in POSITIVE_QUANTITY_TYPES


def stored_target(constraint: datalib.Constraint, context) -> float:
    handler = context.get_constraint_handler(constraint.constraint_type.name)
    return float(handler.get_constraint_props(constraint)["value"])


def quantity_kind(constraint: datalib.Constraint) -> str | None:
    for parameter in constraint.parameters:
        if isinstance(parameter, QuantityParameter) and parameter.parameterId in QUANTITY_KINDS:
            return QUANTITY_KINDS[parameter.parameterId]
    return None


def dimensional_drivers(
    sketch: datalib.Sketch, state: SketchState, context
) -> tuple[dict[tuple, DriverSlot], Counter]:
    """Signature -> unique driver slot, plus a counter of why constraints were dropped."""
    slots: dict[tuple, DriverSlot] = {}
    duplicated: set[tuple] = set()
    skips: Counter = Counter()

    for cid, constraint in sketch.constraints.items():
        if not _is_dimensional(constraint):
            continue
        if context.get_constraint_handler(constraint.constraint_type.name) is None:
            skips["skip_no_handler"] += 1
            continue
        kind = quantity_kind(constraint)
        if kind is None:
            skips["skip_no_quantity_parameter"] += 1
            continue
        q0 = _evaluate_observable(constraint, state, context)
        if q0 is None:
            skips["skip_unevaluable_q0"] += 1
            continue
        signature = constraint_signature(constraint)
        if signature in slots or signature in duplicated:
            duplicated.add(signature)
            slots.pop(signature, None)
            continue
        slots[signature] = DriverSlot(
            cid=cid, kind=kind, q0=q0, baseline_residual=q0 - stored_target(constraint, context)
        )

    if duplicated:
        skips["skip_duplicate_signature"] += len(duplicated)
    return slots, skips


def match_common_drivers(
    slots_p: dict[tuple, DriverSlot],
    slots_g: dict[tuple, DriverSlot],
    *,
    quantity_rtol: float = 1e-6,
) -> tuple[list[DriverMatch], Counter]:
    matches: list[DriverMatch] = []
    skips: Counter = Counter()

    for signature in sorted(slots_p, key=repr):
        slot_g = slots_g.get(signature)
        if slot_g is None:
            continue
        slot_p = slots_p[signature]
        if slot_p.kind != slot_g.kind:
            skips["skip_quantity_kind_mismatch"] += 1
            continue
        if not baseline_quantities_agree(slot_p.q0, slot_g.q0, rtol=quantity_rtol):
            skips["skip_baseline_quantity_mismatch"] += 1
            continue
        matches.append(
            DriverMatch(
                signature=signature,
                driver_type=signature[0],
                kind=slot_p.kind,
                slot_p=slot_p,
                slot_g=slot_g,
            )
        )
    return matches, skips


def baseline_quantities_agree(q_p: float, q_g: float, *, rtol: float = 1e-6) -> bool:
    return abs(q_p - q_g) <= rtol * max(1.0, abs(q_p))
