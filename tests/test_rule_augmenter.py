"""Tests for the rule-based geometric constraint detector (src.sketch_handlers.rule_augmenter).

Builds sketches with known geometry and asserts the detector fires exactly the
predicates that hold (with small normalized margins) and not the ones that don't.
"""
from __future__ import annotations

from collections import Counter, OrderedDict

import numpy as np
import sketchgraphs.data as datalib

from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.primitives.arc import arc_from_three_points
from src.sketch_handlers.rule_augmenter import (
    augment_sketch,
    detect_candidates,
    filter_candidates,
)

CTX = SketchContext()


# --------------------------------------------------------------------------- #
#  Builders
# --------------------------------------------------------------------------- #

def _line(eid: str, p0: tuple[float, float], p1: tuple[float, float]) -> datalib.Line:
    """Line whose endpoints are exactly p0 (start) and p1 (end)."""
    (x0, y0), (x1, y1) = p0, p1
    vec = np.array([x1 - x0, y1 - y0], dtype=float)
    length = float(np.linalg.norm(vec))
    dx, dy = (vec / length).tolist()
    return datalib.Line(
        entityId=eid, isConstruction=False,
        pntX=(x0 + x1) / 2.0, pntY=(y0 + y1) / 2.0,
        dirX=dx, dirY=dy, startParam=-length / 2.0, endParam=length / 2.0,
    )


def _circle(eid: str, cx: float, cy: float, r: float) -> datalib.Circle:
    return datalib.Circle(entityId=eid, isConstruction=False,
                          xCenter=cx, yCenter=cy, radius=r, clockwise=False)


def _arc(
    eid: str,
    p0: tuple[float, float],
    pm: tuple[float, float],
    p1: tuple[float, float],
) -> datalib.Arc:
    return arc_from_three_points(eid, p0, pm, p1)


def _point(eid: str, x: float, y: float) -> datalib.Point:
    return datalib.Point(entityId=eid, isConstruction=False, x=x, y=y)


def _sketch(entities) -> datalib.Sketch:
    return datalib.Sketch(entities=OrderedDict(entities), constraints=OrderedDict())


def _by_type(cands) -> Counter:
    return Counter(c.type_name for c in cands)


def _by_subtype(cands) -> Counter:
    return Counter(c.subtype for c in cands)


def _unit_square() -> datalib.Sketch:
    # corners (0,0)-(1,0)-(1,1)-(0,1); each line's endpoints meet at the corners.
    return _sketch([
        ("b", _line("b", (0, 0), (1, 0))),
        ("r", _line("r", (1, 0), (1, 1))),
        ("t", _line("t", (0, 1), (1, 1))),
        ("l", _line("l", (0, 0), (0, 1))),
    ])


# --------------------------------------------------------------------------- #
#  Square: the canonical orientation / coincidence / equality case
# --------------------------------------------------------------------------- #

def test_square_detects_exactly_the_true_relations():
    accepted = filter_candidates(detect_candidates(_unit_square(), CTX))
    by = _by_type(accepted)

    assert by["Coincident"] == 4          # the 4 corners (point-point only)
    assert by["Horizontal"] == 8          # 2 line-based (bottom, top) + 6 two-point pairs at same Y
    assert by["Vertical"] == 8            # 2 line-based (left, right) + 6 two-point pairs at same X
    assert by["Parallel"] == 2            # (b,t), (l,r)
    assert by["Perpendicular"] == 4       # the 4 right-angle pairs
    assert by["Equal"] == 6               # all C(4,2) equal-length pairs
    # Nothing else (no Tangent/Concentric/Midpoint/PointOnObject on a bare square)
    assert set(by) == {
        "Coincident", "Horizontal", "Vertical", "Parallel", "Perpendicular", "Equal",
    }


def test_square_subtypes_are_resolved():
    accepted = filter_candidates(detect_candidates(_unit_square(), CTX))
    subs = _by_subtype(accepted)
    assert subs["Coincident.point_point"] == 4
    assert subs["Equal.length"] == 6


def test_square_margins_are_near_zero_for_true_relations():
    accepted = filter_candidates(detect_candidates(_unit_square(), CTX))
    assert all(c.margin < 1e-6 for c in accepted), [
        (c.type_name, c.refs, c.margin) for c in accepted if c.margin >= 1e-6
    ]


def test_rotated_line_pair_is_parallel_not_axis_aligned():
    # Two parallel oblique lines at 30°: Parallel holds, line-form Horizontal/Vertical don't.
    # But two-point Vertical(0.start, 1.start) fires because both start at x=0.
    ang = np.radians(30.0)
    d = (float(np.cos(ang)), float(np.sin(ang)))
    s = _sketch([
        ("0", _line("0", (0.0, 0.0), (d[0], d[1]))),
        ("1", _line("1", (0.0, 1.0), (d[0], 1.0 + d[1]))),
    ])
    by = _by_type(filter_candidates(detect_candidates(s, CTX)))
    assert by["Parallel"] == 1
    assert by["Equal"] == 1               # same length
    assert by["Horizontal"] == 0          # no line-based or two-point H
    assert by["Vertical"] == 2            # Vertical(0.start, 1.start) + Vertical(0.end, 1.end)
    assert by["Perpendicular"] == 0


# --------------------------------------------------------------------------- #
#  Circles: Equal-radius vs Concentric must not be confused
# --------------------------------------------------------------------------- #

def test_equal_radius_circles_apart():
    s = _sketch([("0", _circle("0", 0.0, 0.0, 1.0)), ("1", _circle("1", 3.0, 0.0, 1.0))])
    by = _by_type(filter_candidates(detect_candidates(s, CTX)))
    subs = _by_subtype(filter_candidates(detect_candidates(s, CTX)))
    assert by["Equal"] == 1
    assert subs["Equal.radius"] == 1
    assert by.get("Concentric", 0) == 0   # different centers
    assert by.get("Tangent", 0) == 0


def test_concentric_circles_different_radius():
    s = _sketch([("0", _circle("0", 0.0, 0.0, 1.0)), ("1", _circle("1", 0.0, 0.0, 2.0))])
    by = _by_type(filter_candidates(detect_candidates(s, CTX)))
    assert by["Concentric"] == 1
    assert by.get("Equal", 0) == 0        # radii differ (1 vs 2)


# --------------------------------------------------------------------------- #
#  Tangent: line-circle tangency is side-invariant
# --------------------------------------------------------------------------- #

def test_line_circle_tangent_is_detected_on_either_side_of_line():
    for cy in (1.0, -1.0):
        s = _sketch([
            ("L", _line("L", (0.0, 0.0), (2.0, 0.0))),
            ("C", _circle("C", 1.0, cy, 1.0)),
        ])
        cands = filter_candidates(detect_candidates(s, CTX))
        assert _by_subtype(cands)["Tangent.edge"] == 1


def test_line_endpoint_circle_tangent_is_detected_as_point_tangent():
    s = _sketch([
        ("L", _line("L", (0.0, 1.0), (2.0, 1.0))),
        ("C", _circle("C", 0.0, 0.0, 1.0)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    subs = _by_subtype(cands)
    assert subs["Tangent.point"] >= 1
    assert any(c.type_name == "Tangent" and "L.start" in c.refs for c in cands)


# --------------------------------------------------------------------------- #
#  Mirror: existing line axis only
# --------------------------------------------------------------------------- #

def test_point_pair_mirror_about_existing_axis_line():
    s = _sketch([
        ("p0", _point("p0", 0.0, 1.0)),
        ("p1", _point("p1", 0.0, -1.0)),
        ("axis", _line("axis", (-2.0, 0.0), (2.0, 0.0))),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    by = _by_type(cands)
    assert by["Mirror"] == 1


# --------------------------------------------------------------------------- #
#  Point-on-object and Midpoint (interior point on a segment)
# --------------------------------------------------------------------------- #

def test_interior_point_is_point_on_object_and_midpoint():
    # Point exactly at the midpoint of a horizontal segment.
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (2.0, 0.0))),
        ("p", _point("p", 1.0, 0.0)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    by = _by_type(cands)
    subs = _by_subtype(cands)
    assert subs["Coincident.point_on_object"] == 1
    assert by["Midpoint"] == 1
    # The point sits at the centre, not at an endpoint → no point-point coincidence.
    assert subs.get("Coincident.point_point", 0) == 0


def test_curve_center_can_be_the_midpoint_of_a_line():
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (2.0, 0.0))),
        ("C", _circle("C", 1.0, 0.0, 0.25)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    mids = [c for c in cands if c.type_name == "Midpoint"]
    assert any(c.refs == ("C.center", "L") for c in mids)


def test_arc_endpoint_three_ref_midpoint_is_detected_without_line_entity():
    s = _sketch([
        ("A", _arc("A", (0.0, 0.0), (0.5, 1.0), (2.0, 0.0))),
        ("m", _point("m", 1.0, 0.0)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    mids = [c for c in cands if c.type_name == "Midpoint"]
    assert len(mids) == 1
    assert set(mids[0].refs[:2]) == {"A.start", "A.end"}
    assert mids[0].refs[2] == "m"


def test_collinear_point_beyond_segment_is_point_on_object():
    # PointOnObject is defined against the infinite line, so a collinear point
    # beyond the segment's end IS a valid point-on-object (extrapolation allowed).
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (1.0, 0.0))),
        ("p", _point("p", 5.0, 0.0)),  # collinear, beyond the segment end
    ])
    subs = _by_subtype(filter_candidates(detect_candidates(s, CTX)))
    assert subs.get("Coincident.point_on_object", 0) == 1


def test_point_at_segment_endpoint_is_vertex_coincidence_not_point_on_object():
    # A point exactly on the line's endpoint is a vertex coincidence (point-point),
    # not point-on-object — so the corner is not double-counted as both.
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (2.0, 0.0))),
        ("p", _point("p", 0.0, 0.0)),  # exactly at L's start vertex
    ])
    subs = _by_subtype(filter_candidates(detect_candidates(s, CTX)))
    assert subs.get("Coincident.point_on_object", 0) == 0
    assert subs.get("Coincident.point_point", 0) == 1


# --------------------------------------------------------------------------- #
#  augment_sketch: writes accepted, non-duplicate constraints back
# --------------------------------------------------------------------------- #

def test_augment_sketch_adds_and_dedups():
    sketch = _unit_square()
    n_detected = len(filter_candidates(detect_candidates(sketch, CTX)))
    assert n_detected == 32  # 20 original + 12 two-point H/V (6H + 6V)

    augmented = augment_sketch(sketch, CTX)
    assert len(augmented.constraints) == 32
    # idempotent: augmenting again adds nothing (all duplicates).
    again = augment_sketch(augmented, CTX)
    assert len(again.constraints) == 32
    # original sketch is untouched.
    assert len(sketch.constraints) == 0


# --------------------------------------------------------------------------- #
#  Two-point Horizontal / Vertical (cross-entity alignment)
# --------------------------------------------------------------------------- #

def test_two_point_horizontal_circle_center_and_line_endpoint():
    """Horizontal alignment: circle centre at (3, 0) and line endpoint at (0, 0)."""
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (1.0, 1.0))),
        ("C", _circle("C", 3.0, 0.0, 0.5)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    h_cands = [c for c in cands if c.type_name == "Horizontal" and len(c.refs) == 2]
    # L.start (0,0) and C.center (3,0) share the same Y → Horizontal
    assert any(set(c.refs) == {"L.start", "C.center"} for c in h_cands), \
        f"Expected Horizontal(L.start, C.center), got refs: {[c.refs for c in h_cands]}"


def test_two_point_vertical_circle_center_and_line_endpoint():
    """Vertical alignment: circle centre at (0, 3) and line endpoint at (0, 0)."""
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (1.0, 1.0))),
        ("C", _circle("C", 0.0, 3.0, 0.5)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    v_cands = [c for c in cands if c.type_name == "Vertical" and len(c.refs) == 2]
    # L.start (0,0) and C.center (0,3) share the same X → Vertical
    assert any(set(c.refs) == {"L.start", "C.center"} for c in v_cands), \
        f"Expected Vertical(L.start, C.center), got refs: {[c.refs for c in v_cands]}"


def test_two_point_hv_does_not_fire_for_oblique_points():
    """No two-point H/V for points that are neither horizontally nor vertically aligned."""
    s = _sketch([
        ("p0", _point("p0", 0.0, 0.0)),
        ("p1", _point("p1", 1.0, 1.0)),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    hv = [c for c in cands if c.type_name in ("Horizontal", "Vertical")]
    assert len(hv) == 0


def test_two_point_hv_same_entity_endpoints_not_duplicated():
    """Two-point H/V within the same Line is already covered by line-based H/V.
    Verify the same-entity guard prevents duplicate detection."""
    s = _sketch([
        ("L", _line("L", (0.0, 0.0), (1.0, 0.0))),
    ])
    cands = filter_candidates(detect_candidates(s, CTX))
    h_cands = [c for c in cands if c.type_name == "Horizontal"]
    # Only the line-based form Horizontal("L",) — no two-point Horizontal("L.start", "L.end")
    assert len(h_cands) == 1
    assert h_cands[0].refs == ("L",)


# --------------------------------------------------------------------------- #
#  Same-arc Horizontal / Vertical (an Arc has no single-entity H/V form, so its
#  own endpoint chord and centre-to-endpoint radius are real candidates)
# --------------------------------------------------------------------------- #

def test_same_arc_horizontal_chord():
    """The two endpoints of one arc share a Y → Horizontal(arc.start, arc.end).

    Bulge is shallow so the centre is off the chord line, isolating the chord
    from the centre-to-endpoint pairs."""
    s = _sketch([("A", _arc("A", (-1.0, 2.0), (0.0, 2.3), (1.0, 2.0)))])
    cands = filter_candidates(detect_candidates(s, CTX))
    h = [c for c in cands if c.type_name == "Horizontal"]
    assert any(set(c.refs) == {"A.start", "A.end"} for c in h), \
        f"Expected same-arc Horizontal(A.start, A.end), got: {[c.refs for c in h]}"
    # The chord is horizontal but neither centre-to-endpoint pair is, and there
    # is no single-entity arc form.
    assert not any(c.type_name == "Vertical" for c in cands)
    assert all(len(c.refs) == 2 for c in h)  # no spurious single-ref arc form


def test_same_arc_vertical_chord():
    """The two endpoints of one arc share an X → Vertical(arc.start, arc.end)."""
    s = _sketch([("A", _arc("A", (2.0, -1.0), (2.3, 0.0), (2.0, 1.0)))])
    cands = filter_candidates(detect_candidates(s, CTX))
    v = [c for c in cands if c.type_name == "Vertical"]
    assert any(set(c.refs) == {"A.start", "A.end"} for c in v), \
        f"Expected same-arc Vertical(A.start, A.end), got: {[c.refs for c in v]}"
    assert not any(c.type_name == "Horizontal" for c in cands)


def test_same_arc_center_to_endpoint_alignment():
    """Quarter arc centred at origin from (1,0) to (0,1): the centre is
    horizontally aligned with the start and vertically aligned with the end."""
    s = _sketch([("A", _arc("A", (1.0, 0.0), (float(np.cos(np.pi / 4)), float(np.sin(np.pi / 4))), (0.0, 1.0)))])
    cands = filter_candidates(detect_candidates(s, CTX))
    h = [c for c in cands if c.type_name == "Horizontal"]
    v = [c for c in cands if c.type_name == "Vertical"]
    assert any(set(c.refs) == {"A.center", "A.start"} for c in h), \
        f"Expected Horizontal(A.center, A.start), got: {[c.refs for c in h]}"
    assert any(set(c.refs) == {"A.center", "A.end"} for c in v), \
        f"Expected Vertical(A.center, A.end), got: {[c.refs for c in v]}"
    # The chord (1,0)-(0,1) is diagonal → no chord H/V.
    assert not any(set(c.refs) == {"A.start", "A.end"} for c in (h + v))
