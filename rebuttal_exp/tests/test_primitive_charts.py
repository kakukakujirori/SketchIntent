"""The rectangular charts must be exact local reparameterizations, or say why they are not."""
from __future__ import annotations

import math

import pytest
import torch

from rebuttal_exp.primitive_charts import (
    C1,
    C2,
    arc_forward,
    arc_lift,
    arc_three_point_chart,
    build_global_chart,
    identity_chart,
    line_endpoint_chart,
    line_forward,
    line_lift,
    relative_gauge_residual,
)
from rebuttal_exp.tests.fixtures import arc, circle, line, mixed_primitive_sketch
from src.sketch_handlers.sketch_solver import SketchState

SCALE = 1.0


def _line_state(x0: float, y0: float, x1: float, y1: float) -> torch.Tensor:
    entity = line("0", x0, y0, x1, y1)
    return torch.tensor(
        [entity.pntX, entity.pntY, entity.dirX, entity.dirY,
         entity.startParam, entity.endParam], dtype=torch.float64
    )


def _arc_state(entity) -> torch.Tensor:
    return torch.tensor(
        [entity.xCenter, entity.yCenter, entity.radius, entity.startParam, entity.endParam],
        dtype=torch.float64,
    )


def _finite_difference(fn, point: torch.Tensor, step: float = 1e-6) -> torch.Tensor:
    columns = []
    for i in range(point.numel()):
        plus, minus = point.clone(), point.clone()
        plus[i] += step
        minus[i] -= step
        columns.append((fn(plus) - fn(minus)) / (2 * step))
    return torch.stack(columns, dim=1)


# ---------------------------------------------------------------------------
# C1: line endpoints
# ---------------------------------------------------------------------------

def test_line_chart_is_an_exact_local_reparameterization():
    x0 = _line_state(0.0, 0.0, 3.0, 4.0)
    chart = line_endpoint_chart("0", x0, SCALE)

    assert chart.usable
    assert chart.B.shape == (6, 4)
    assert chart.H is not None and chart.H.shape == (4, 6)
    assert chart.diagnostics["rank_H"] == 4
    assert chart.diagnostics["rank_B"] == 4
    assert chart.diagnostics["identity_error"] == pytest.approx(0.0, abs=1e-12)
    assert chart.diagnostics["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(chart.H @ chart.B, torch.eye(4, dtype=torch.float64), atol=1e-12)


def test_line_lift_recovers_the_baseline_state():
    x0 = _line_state(-1.0, 2.0, 5.0, -3.0)
    s0, t0 = float(x0[4]), float(x0[5])
    assert torch.allclose(line_lift(line_forward(x0), s0, t0), x0, atol=1e-12)


def test_line_chart_jacobians_match_finite_differences():
    x0 = _line_state(0.5, -0.5, 2.5, 1.5)
    s0, t0 = float(x0[4]), float(x0[5])
    chart = line_endpoint_chart("0", x0, SCALE)

    assert torch.allclose(chart.H, _finite_difference(line_forward, x0), atol=1e-6)
    assert torch.allclose(
        chart.B, _finite_difference(lambda y: line_lift(y, s0, t0), line_forward(x0)), atol=1e-6
    )


def test_line_gauge_is_the_known_slide_and_scale_pair():
    """ker(H) must be the two directions that leave both endpoints fixed."""
    x0 = _line_state(0.0, 0.0, 2.0, 1.0)
    chart = line_endpoint_chart("0", x0, SCALE)
    gauge = chart.x_gauge_basis

    assert gauge is not None and gauge.shape == (6, 2)
    direction, s, t = x0[2:4], float(x0[4]), float(x0[5])
    slide = torch.zeros(6, dtype=torch.float64)
    slide[0:2], slide[4], slide[5] = direction, -1.0, -1.0
    scale = torch.zeros(6, dtype=torch.float64)
    scale[2:4], scale[4], scale[5] = direction, -s, -t
    known = torch.stack([slide, scale], dim=1)

    assert torch.allclose(chart.H @ known, torch.zeros(4, 2, dtype=torch.float64), atol=1e-12)
    projected = gauge @ (gauge.T @ known)
    assert torch.allclose(projected, known, atol=1e-10), "ker(H) does not span the known gauge"


def test_degenerate_line_is_skipped_not_inverted():
    x0 = _line_state(0.0, 0.0, 1.0, 0.0)
    x0[5] = x0[4]
    chart = line_endpoint_chart("0", x0, SCALE)
    assert chart.skip_reason == "degenerate_line_span"
    assert not chart.usable


# ---------------------------------------------------------------------------
# C2: arc three-point
# ---------------------------------------------------------------------------

ARC_CASES = {
    "ccw_minor": dict(start_param=0.2, end_param=1.3, clockwise=False),
    "cw_minor": dict(start_param=0.2, end_param=1.3, clockwise=True),
    "ccw_major": dict(start_param=0.0, end_param=4.5, clockwise=False),
    "cw_major": dict(start_param=0.0, end_param=4.5, clockwise=True),
    "near_two_pi": dict(start_param=0.0, end_param=2 * math.pi - 0.3, clockwise=False),
    "rotated_frame": dict(start_param=0.3, end_param=2.0, clockwise=False,
                          x_dir=math.cos(0.7), y_dir=math.sin(0.7)),
}


@pytest.mark.parametrize("case", sorted(ARC_CASES))
def test_arc_chart_is_an_exact_local_reparameterization(case):
    entity = arc("0", 1.0, -2.0, 1.7, **ARC_CASES[case])
    x0 = _arc_state(entity)
    chart = arc_three_point_chart("0", entity, x0, SCALE)

    assert chart.usable, chart.skip_reason
    assert chart.B.shape == (5, 6)
    assert chart.H is not None and chart.H.shape == (6, 5)
    assert chart.diagnostics["rank_H"] == 5
    assert chart.diagnostics["rank_B"] == 5
    assert torch.allclose(chart.B @ chart.H, torch.eye(5, dtype=torch.float64), atol=1e-9)
    assert chart.diagnostics["reconstruction_error"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("case", sorted(ARC_CASES))
def test_arc_lift_recovers_the_baseline_state_on_the_fixed_branch(case):
    entity = arc("0", 0.5, 0.25, 2.2, **ARC_CASES[case])
    x0 = _arc_state(entity)
    theta_base = math.atan2(entity.yDir, entity.xDir)
    sign = -1.0 if entity.clockwise else 1.0
    chart = arc_three_point_chart("0", entity, x0, SCALE)

    recovered = arc_lift(
        arc_forward(x0, theta_base, sign), theta_base, sign,
        chart.diagnostics["angle_branch_start"], chart.diagnostics["angle_branch_end"],
    )
    assert torch.allclose(recovered, x0, atol=1e-9)


def test_arc_chart_selects_branch_after_circumcircle_reconstruction():
    """A roundoff-sized center shift at +pi must not select the -pi sweep."""
    entity = arc(
        "0", 9.171407594729557e-17, 0.406513180800687, 0.01521739130434796,
        start_param=5.075305255429244e-14,
        end_param=math.pi - 5.062616992290714e-14,
    )
    x0 = _arc_state(entity)

    chart = arc_three_point_chart("0", entity, x0, SCALE)

    assert chart.usable, chart.skip_reason
    assert chart.diagnostics["raw_state_reconstruction_error"] == pytest.approx(0.0, abs=1e-9)
    assert chart.diagnostics["angle_reconstruction_error"] == pytest.approx(0.0, abs=1e-9)
    assert chart.diagnostics["point_reconstruction_error"] == pytest.approx(0.0, abs=1e-9)


def test_arc_chart_keeps_its_one_redundant_direction():
    """ker(B) is the through-point gauge and must not be squared away."""
    entity = arc("0", 0.0, 0.0, 1.0, start_param=0.1, end_param=2.0)
    chart = arc_three_point_chart("0", entity, _arc_state(entity), SCALE)
    assert chart.diagnostics["y_redundancy_dim"] == 1
    assert int(torch.linalg.matrix_rank(chart.B, rtol=1e-8)) == 5


def test_arc_jacobians_match_finite_differences():
    entity = arc("0", -0.75, 1.25, 1.4, start_param=0.4, end_param=2.6)
    x0 = _arc_state(entity)
    theta_base = math.atan2(entity.yDir, entity.xDir)
    sign = 1.0
    chart = arc_three_point_chart("0", entity, x0, SCALE)

    forward = lambda x: arc_forward(x, theta_base, sign)  # noqa: E731
    lift = lambda y: arc_lift(  # noqa: E731
        y, theta_base, sign,
        chart.diagnostics["angle_branch_start"], chart.diagnostics["angle_branch_end"],
    )
    assert torch.allclose(chart.H, _finite_difference(forward, x0), atol=1e-5)
    assert torch.allclose(chart.B, _finite_difference(lift, forward(x0)), atol=1e-5)


@pytest.mark.parametrize("end_param", [1e-9, 2 * math.pi - 1e-9])
def test_nearly_collinear_three_points_are_skipped(end_param):
    entity = arc("0", 0.0, 0.0, 1.0, start_param=0.0, end_param=end_param)
    chart = arc_three_point_chart("0", entity, _arc_state(entity), SCALE)
    assert chart.skip_reason == "ill_conditioned_arc_three_point"


def test_zero_radius_arc_is_skipped():
    entity = arc("0", 0.0, 0.0, 1e-12, start_param=0.0, end_param=1.0)
    chart = arc_three_point_chart("0", entity, _arc_state(entity), SCALE)
    assert chart.skip_reason == "ill_conditioned_arc_three_point"


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def test_global_chart_changes_only_the_targeted_primitive_type():
    sketch = mixed_primitive_sketch()
    state = SketchState(sketch)
    n_x = state.num_dof()

    c1 = build_global_chart(sketch, state, C1, SCALE)
    c2 = build_global_chart(sketch, state, C2, SCALE)

    assert c1.n_x == n_x and c1.n_y == n_x - 2, "one line loses two variables"
    assert c2.n_x == n_x and c2.n_y == n_x + 1, "one arc gains one variable"
    assert c1.x_gauge_basis.shape == (n_x, 2)
    assert c2.x_gauge_basis.shape == (n_x, 0), "the arc chart discards no x-space direction"
    assert [b.entity_type for b in c1.blocks] == ["Line", "Circle", "Arc", "Point"]


def test_identity_chart_is_square_and_gaugeless():
    state = SketchState(mixed_primitive_sketch())
    chart = identity_chart(state)
    assert chart.n_x == chart.n_y == state.num_dof()
    assert torch.allclose(chart.B, torch.eye(chart.n_x, dtype=torch.float64))
    assert chart.x_gauge_basis.shape[1] == 0


def test_unexpected_state_layout_is_an_error_not_a_silent_mismatch():
    sketch = mixed_primitive_sketch()
    sketch.entities["e_extra"] = circle("e_extra", 9.0, 9.0, 1.0)
    state = SketchState(sketch)
    state.slices["e_extra"] = slice(0, 4)
    with pytest.raises(AssertionError):
        build_global_chart(sketch, state, C1, SCALE)


def test_relative_gauge_residual_is_scale_free():
    gauge = torch.eye(4, 2, dtype=torch.float64)
    orthogonal = torch.zeros(3, 4, dtype=torch.float64)
    orthogonal[:, 2:] = torch.randn(3, 2, dtype=torch.float64)
    assert relative_gauge_residual(orthogonal, gauge) == pytest.approx(0.0, abs=1e-12)
    assert relative_gauge_residual(10.0 * orthogonal, gauge) == pytest.approx(0.0, abs=1e-12)

    sees_gauge = torch.zeros(3, 4, dtype=torch.float64)
    sees_gauge[0, 0] = 1.0
    assert relative_gauge_residual(sees_gauge, gauge) == pytest.approx(1.0)
    assert relative_gauge_residual(None, gauge) == 0.0
