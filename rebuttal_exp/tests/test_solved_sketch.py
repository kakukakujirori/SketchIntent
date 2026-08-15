"""FreeCAD round trips must come back in the caller's id space, or be rejected.

Requires FreeCAD and the evaluation dataset.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import sketchgraphs.data as datalib

from rebuttal_exp.common import build_context, discover_items, load_bundle
from rebuttal_exp.exact_observable import build_observable_layout, evaluate_observable
from rebuttal_exp.solved_sketch import (
    constraint_residual_norm,
    geometry_is_finite,
    rebase_geometry,
    solve_from,
)
from rebuttal_exp.tests.fixtures import circle, mixed_primitive_sketch
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_solver import SketchState

try:
    ensure_freecad_imported()
except ImportError:  # pragma: no cover - environment without FreeCAD
    pytest.skip("FreeCAD is unavailable", allow_module_level=True)


@pytest.fixture(scope="module")
def context_and_converter():
    return build_context()


@pytest.fixture(scope="module")
def first_bundle(context_and_converter):
    _context, converter = context_and_converter
    item = discover_items(limit=1)[0]
    return item, load_bundle(item, converter, 5e-1)


def test_rebase_keeps_template_ids_and_takes_solved_geometry():
    template = mixed_primitive_sketch()
    solved = mixed_primitive_sketch()
    solved.entities["e_circle"].radius = 9.5
    solved.entities["e_point"].x = -7.0

    rebased = rebase_geometry(template, solved)

    assert list(rebased.entities) == list(template.entities)
    assert rebased.entities["e_circle"].radius == pytest.approx(9.5)
    assert rebased.entities["e_point"].x == pytest.approx(-7.0)
    assert template.entities["e_circle"].radius == pytest.approx(0.75), "template untouched"


def test_rebase_rejects_a_changed_entity_sequence():
    template = mixed_primitive_sketch()
    shorter = datalib.Sketch(
        entities=OrderedDict([("e_line", template.entities["e_line"])]),
        constraints=OrderedDict(),
    )
    assert rebase_geometry(template, shorter) is None

    retyped = datalib.Sketch(
        entities=OrderedDict(
            (eid, circle(eid, 0.0, 0.0, 1.0)) for eid in template.entities
        ),
        constraints=OrderedDict(),
    )
    assert rebase_geometry(template, retyped) is None


def test_solve_returns_canonical_ids_and_a_consistent_geometry(first_bundle, context_and_converter):
    item, bundle = first_bundle
    context, converter = context_and_converter
    label = item.human_labels[0]
    sketch = bundle.sketches[label]

    outcome = solve_from(sketch, converter)

    assert outcome.ok, outcome.status
    assert list(outcome.sketch.entities) == list(sketch.entities)
    assert list(outcome.sketch.constraints) == list(sketch.constraints)
    assert geometry_is_finite(outcome.sketch)
    assert constraint_residual_norm(
        outcome.sketch, SketchState(outcome.sketch), context
    ) < 1e-6


def test_observable_layout_survives_the_round_trip(first_bundle, context_and_converter):
    item, bundle = first_bundle
    _context, converter = context_and_converter
    label = item.human_labels[0]
    sketch = bundle.sketches[label]

    layout = build_observable_layout(sketch)
    solved = solve_from(sketch, converter).sketch

    assert build_observable_layout(solved) == layout
    before = evaluate_observable(sketch, layout)
    after = evaluate_observable(solved, layout)
    assert before.shape == after.shape
    assert np.all(np.isfinite(after))
