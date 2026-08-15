"""Exact nonlinear observable phi(x), evaluated from geometry rather than from J_phi.

The reference collection order, the pair enumeration order and the trailing radii block
mirror ``src.metrics.edit_response_distance.pairwise_distance_observable`` exactly, so a
baseline phi computed here lives in the same coordinate system as the metric's J_phi.
Finite responses must never be extrapolated as ``J_phi @ delta_x``; this module is what
makes that unnecessary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sketchgraphs.data as datalib

from src.sketch_handlers.sketch_solver import SketchState


@dataclass(frozen=True)
class ObservableLayout:
    point_refs: tuple[str, ...]
    radius_eids: tuple[str, ...]

    @property
    def dimension(self) -> int:
        n = len(self.point_refs)
        return (n * (n - 1)) // 2 + len(self.radius_eids)


def build_observable_layout(sketch: datalib.Sketch, base_eids=None) -> ObservableLayout:
    allowed = set(base_eids) if base_eids is not None else None
    point_refs: list[str] = []
    radius_eids: list[str] = []
    for eid, entity in sketch.entities.items():
        if allowed is not None and eid not in allowed:
            continue
        if isinstance(entity, datalib.Line):
            point_refs.append(f"{eid}.start")
            point_refs.append(f"{eid}.end")
        elif isinstance(entity, datalib.Circle):
            point_refs.append(f"{eid}.center")
            radius_eids.append(eid)
        elif isinstance(entity, datalib.Arc):
            point_refs.append(f"{eid}.start")
            point_refs.append(f"{eid}.end")
            point_refs.append(f"{eid}.center")
            radius_eids.append(eid)
        elif isinstance(entity, datalib.Point):
            point_refs.append(eid)
    return ObservableLayout(tuple(point_refs), tuple(radius_eids))


def layouts_match(a: ObservableLayout, b: ObservableLayout) -> bool:
    return a.point_refs == b.point_refs and a.radius_eids == b.radius_eids


def evaluate_observable(sketch: datalib.Sketch, layout: ObservableLayout) -> np.ndarray:
    state = SketchState(sketch)
    parts: list[np.ndarray] = []
    if len(layout.point_refs) >= 2:
        points = np.stack(
            [state.get_point(ref).detach().numpy() for ref in layout.point_refs], axis=0
        )
        ia, ib = np.triu_indices(len(layout.point_refs), k=1)
        parts.append(np.linalg.norm(points[ia] - points[ib], axis=1))
    if layout.radius_eids:
        parts.append(
            np.array([float(state.get_radius(eid)) for eid in layout.radius_eids], dtype=float)
        )
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts)
