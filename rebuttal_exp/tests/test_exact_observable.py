"""The exact nonlinear phi must agree with the metric's J_phi in ordering and in derivative."""
from __future__ import annotations

import numpy as np
import torch

from rebuttal_exp.exact_observable import (
    build_observable_layout,
    evaluate_observable,
    layouts_match,
)
from rebuttal_exp.tests.fixtures import mixed_primitive_sketch, reordered
from src.metrics.edit_response_distance import pairwise_distance_observable
from src.sketch_handlers.sketch_solver import SketchState

ENTITY_ORDER = ("e_line", "e_circle", "e_arc", "e_point")


def test_landmark_order_follows_primitive_convention():
    layout = build_observable_layout(mixed_primitive_sketch())
    assert layout.point_refs == (
        "e_line.start", "e_line.end",
        "e_circle.center",
        "e_arc.start", "e_arc.end", "e_arc.center",
        "e_point",
    )
    assert layout.radius_eids == ("e_circle", "e_arc")


def test_dimension_matches_j_phi_rows():
    sketch = mixed_primitive_sketch()
    layout = build_observable_layout(sketch)
    assert layout.dimension == pairwise_distance_observable(sketch).shape[0]
    assert evaluate_observable(sketch, layout).shape == (layout.dimension,)


def test_base_eids_restrict_both_representations_identically():
    sketch = mixed_primitive_sketch()
    base_eids = ["e_line", "e_circle"]
    layout = build_observable_layout(sketch, base_eids)
    assert layout.point_refs == ("e_line.start", "e_line.end", "e_circle.center")
    assert layout.radius_eids == ("e_circle",)
    assert layout.dimension == pairwise_distance_observable(sketch, base_eids=base_eids).shape[0]


def test_layout_follows_entity_iteration_order_and_mismatches_are_detected():
    """The layout tracks the sketch's own entity order, exactly like J_phi does.

    Bundle sketches all inherit the canonical entity order, so their layouts agree; a
    sketch whose order differs must be reported rather than silently compared.
    """
    sketch = mixed_primitive_sketch()
    assert layouts_match(
        build_observable_layout(sketch, ENTITY_ORDER),
        build_observable_layout(reordered(sketch, ENTITY_ORDER), ENTITY_ORDER),
    )
    assert not layouts_match(
        build_observable_layout(sketch, ENTITY_ORDER),
        build_observable_layout(reordered(sketch, ("e_point", "e_arc", "e_circle", "e_line")),
                                ENTITY_ORDER),
    )


def test_derivative_of_exact_phi_equals_j_phi():
    """phi is recomputed from geometry; its numerical Jacobian must reproduce J_phi."""
    sketch = mixed_primitive_sketch()
    layout = build_observable_layout(sketch)
    j_phi = pairwise_distance_observable(sketch).numpy()

    state = SketchState(sketch)
    q0 = state.q.detach().clone()
    step = 1e-6
    numerical = np.empty_like(j_phi)
    for column in range(q0.numel()):
        perturbed = q0.clone()
        perturbed[column] = q0[column] + step
        plus = evaluate_observable(_with_q(sketch, perturbed), layout)
        perturbed[column] = q0[column] - step
        minus = evaluate_observable(_with_q(sketch, perturbed), layout)
        numerical[:, column] = (plus - minus) / (2 * step)

    assert np.allclose(numerical, j_phi, rtol=1e-5, atol=1e-6)


def _with_q(sketch, q: torch.Tensor):
    """Rebuild the sketch geometry from a perturbed SketchState parameter vector."""
    import copy

    import sketchgraphs.data as datalib

    out = copy.deepcopy(sketch)
    state = SketchState(sketch)
    values = q.detach().numpy()
    for eid, entity in out.entities.items():
        chunk = values[state.slices[eid]]
        if isinstance(entity, datalib.Line):
            (entity.pntX, entity.pntY, entity.dirX,
             entity.dirY, entity.startParam, entity.endParam) = (float(v) for v in chunk)
        elif isinstance(entity, datalib.Circle):
            entity.xCenter, entity.yCenter, entity.radius = (float(v) for v in chunk)
        elif isinstance(entity, datalib.Arc):
            (entity.xCenter, entity.yCenter, entity.radius,
             entity.startParam, entity.endParam) = (float(v) for v in chunk)
        elif isinstance(entity, datalib.Point):
            entity.x, entity.y = (float(v) for v in chunk)
    return out
