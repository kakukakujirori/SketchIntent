"""Common-driver matching must be exact: no approximate or ambiguous matches."""
from __future__ import annotations

from collections import OrderedDict

import pytest
import sketchgraphs.data as datalib

from rebuttal_exp.driver_matching import (
    baseline_quantities_agree,
    dimensional_drivers,
    match_common_drivers,
    quantity_kind,
)
from rebuttal_exp.tests.fixtures import constraint, line
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_solver import SketchState


@pytest.fixture(scope="module")
def context() -> SketchContext:
    return SketchContext()


def _two_line_sketch(constraints: OrderedDict) -> datalib.Sketch:
    entities = OrderedDict([
        ("0", line("0", 0.0, 0.0, 2.0, 0.0)),
        ("1", line("1", 0.0, 3.0, 2.0, 3.0)),
    ])
    return datalib.Sketch(entities=entities, constraints=constraints)


def _distance(context, refs, value):
    return constraint(context, "Distance", refs, value=value)


def _drivers(context, constraints):
    sketch = _two_line_sketch(constraints)
    return dimensional_drivers(sketch, SketchState(sketch), context)


def test_same_driver_at_different_list_positions_still_matches(context):
    shared = ("0.start", "1.start")
    slots_p, _ = _drivers(context, OrderedDict([
        ("a", _distance(context, shared, 3.0)),
        ("b", _distance(context, ("0.end", "1.end"), 3.0)),
    ]))
    slots_g, _ = _drivers(context, OrderedDict([
        ("z", _distance(context, ("0.end", "1.end"), 3.0)),
        ("y", _distance(context, shared, 3.0)),
    ]))
    matches, skips = match_common_drivers(slots_p, slots_g)

    assert len(matches) == 2
    assert not skips
    by_p_cid = {m.slot_p.cid: m.slot_g.cid for m in matches}
    assert by_p_cid == {"a": "y", "b": "z"}


def test_reference_order_is_canonicalised(context):
    slots_p, _ = _drivers(context, OrderedDict(
        [("a", _distance(context, ("0.start", "1.start"), 3.0))]
    ))
    slots_g, _ = _drivers(context, OrderedDict(
        [("z", _distance(context, ("1.start", "0.start"), 3.0))]
    ))
    matches, _ = match_common_drivers(slots_p, slots_g)
    assert len(matches) == 1


def test_different_endpoint_does_not_match(context):
    slots_p, _ = _drivers(context, OrderedDict(
        [("a", _distance(context, ("0.start", "1.start"), 3.0))]
    ))
    slots_g, _ = _drivers(context, OrderedDict(
        [("z", _distance(context, ("0.end", "1.start"), 3.0))]
    ))
    matches, _ = match_common_drivers(slots_p, slots_g)
    assert matches == []


def test_duplicate_signature_is_dropped_as_ambiguous(context):
    refs = ("0.start", "1.start")
    slots, skips = _drivers(context, OrderedDict([
        ("a", _distance(context, refs, 3.0)),
        ("b", _distance(context, refs, 3.0)),
    ]))
    assert slots == {}
    assert skips["skip_duplicate_signature"] == 1


def test_angle_and_length_drivers_are_distinct(context):
    slots_p, _ = _drivers(context, OrderedDict(
        [("a", _distance(context, ("0.start", "1.start"), 3.0))]
    ))
    slots_g, _ = _drivers(context, OrderedDict(
        [("z", constraint(context, "Angle", ("0", "1"), value=0.0))]
    ))
    matches, _ = match_common_drivers(slots_p, slots_g)
    assert matches == []


def test_quantity_kind_reads_the_parameter_id(context):
    assert quantity_kind(_distance(context, ("0.start", "1.start"), 3.0)) == "length"
    assert quantity_kind(constraint(context, "Angle", ("0", "1"), value=0.0)) == "angle"


def test_baseline_quantity_mismatch_is_skipped_not_approximated(context):
    slots_p, _ = _drivers(context, OrderedDict(
        [("a", _distance(context, ("0.start", "1.start"), 3.0))]
    ))
    slots_g = dict(slots_p)
    signature = next(iter(slots_g))
    slots_g[signature] = type(slots_g[signature])(cid="z", kind="length", q0=slots_p[signature].q0 + 0.5)

    matches, skips = match_common_drivers(slots_p, slots_g)
    assert matches == []
    assert skips["skip_baseline_quantity_mismatch"] == 1


def test_quantity_agreement_tolerance_is_relative_above_unit_scale():
    assert baseline_quantities_agree(1000.0, 1000.0 + 1e-4)
    assert not baseline_quantities_agree(1000.0, 1000.0 + 1e-2)
    assert baseline_quantities_agree(0.001, 0.001 + 1e-7)
