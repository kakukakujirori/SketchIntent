"""FreeCAD solving that stays in the caller's entity/constraint id space.

``_solve_via_freecad`` returns the sketch that ``converter.from_freecad`` rebuilt, whose
entity ids are renumbered to the FreeCAD geometry index and whose constraint ids become
``c_<i>``. Canonical base_eids, ``constraint_signature`` and cached constraint ids would
all be invalid on that object, so every solve here writes the solved geometry back onto
the template sketch, which keeps the canonical ids and the original constraints.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import sketchgraphs.data as datalib
import torch

from src.metrics.histcad_2d import _refs_present, _solve_via_freecad
from src.sketch_handlers.sketch_solver import SketchState

_GEOMETRY_BOOL_FIELDS = ("clockwise",)


@dataclass(frozen=True)
class SolveOutcome:
    sketch: datalib.Sketch | None
    status: str
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.sketch is not None


def rebase_geometry(template: datalib.Sketch, solved: datalib.Sketch) -> datalib.Sketch | None:
    """Copy solved geometry onto a deep copy of ``template``, keeping template ids.

    Returns ``None`` when the round trip did not preserve the entity sequence, so the
    caller can record a skip instead of silently comparing mismatched geometry.
    """
    template_entities = list(template.entities.values())
    solved_entities = list(solved.entities.values())
    if len(template_entities) != len(solved_entities):
        return None
    if any(type(a) is not type(b) for a, b in zip(template_entities, solved_entities)):
        return None

    out = copy.deepcopy(template)
    for target, source in zip(out.entities.values(), solved_entities):
        for field in type(target).float_ids:
            setattr(target, field, float(getattr(source, field)))
        for field in _GEOMETRY_BOOL_FIELDS:
            if hasattr(target, field):
                setattr(target, field, bool(getattr(source, field)))
    return out


def solve_from(sketch: datalib.Sketch, converter) -> SolveOutcome:
    started = time.perf_counter()
    solved, status = _solve_via_freecad(sketch, converter)
    elapsed = time.perf_counter() - started
    if solved is None:
        return SolveOutcome(None, status, elapsed)
    rebased = rebase_geometry(sketch, solved)
    if rebased is None:
        return SolveOutcome(None, "roundtrip_layout_mismatch", elapsed)
    if not geometry_is_finite(rebased):
        return SolveOutcome(None, "nonfinite_geometry", elapsed)
    return SolveOutcome(rebased, status, elapsed)


def geometry_is_finite(sketch: datalib.Sketch) -> bool:
    return all(
        math.isfinite(float(getattr(entity, field)))
        for entity in sketch.entities.values()
        for field in type(entity).float_ids
    )


def constraint_residual_norm(sketch: datalib.Sketch, state: SketchState, context) -> float:
    """RMS residual over every constraint whose handler can evaluate it at ``state``."""
    squares = 0.0
    rows = 0
    for constraint in sketch.constraints.values():
        handler = context.get_constraint_handler(constraint.constraint_type.name)
        if handler is None or not _refs_present(constraint, state):
            continue
        try:
            residual = handler.residual(constraint, state).detach().reshape(-1)
        except Exception:
            continue
        squares += float(torch.sum(residual**2))
        rows += int(residual.numel())
    return math.sqrt(squares / rows) if rows else 0.0
