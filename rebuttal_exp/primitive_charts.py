"""Local charts between the native primitive state and alternative CAD representations.

The native state is fixed by ``SketchState``:

    Line   (6): [pntX, pntY, dirX, dirY, startParam, endParam]
    Circle (3): [xCenter, yCenter, radius]
    Arc    (5): [xCenter, yCenter, radius, startParam, endParam]
    Point  (2): [x, y]

Two alternative representations are reproduced at Jacobian level:

    C1  Line -> (start point, end point)                 6 -> 4
    C2  Arc  -> (start point, sweep midpoint, end point) 5 -> 6

Both change the number of variables, so ``B = dg/dy`` is rectangular and is applied by
right multiplication; no inverse of ``B`` is ever formed. ``h`` is the forward map and is
built alongside ``g`` purely so the rank / left-inverse / round-trip conditions can be
checked per primitive. Every Jacobian comes from autodiff so those checks are meaningful.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import sketchgraphs.data as datalib
import torch

from src.sketch_handlers.sketch_solver import SketchState

NATIVE = "native"
C0 = "C0"
C1 = "C1"
C2 = "C2"
SCENARIOS = (NATIVE, C0, C1, C2)
TRANSFORMED_SCENARIOS = (C0, C1, C2)

RANK_EPS = 1e-8
DEGENERATE_LINE_TOL = 1e-8
ARC_RADIUS_TOL = 1e-8
ARC_SIGMA_ABS_TOL = 1e-10
ARC_SIGMA_REL_TOL = 1e-8
ARC_RECONSTRUCTION_TOL = 1e-8
ARC_ANGLE_TOL = 1e-8
LEFT_INVERSE_TOL = 1e-8
GAUGE_RESIDUAL_TOL = 1e-8


@dataclass(frozen=True)
class BlockChart:
    eid: str
    entity_type: str
    B: torch.Tensor
    H: torch.Tensor | None
    x_gauge_basis: torch.Tensor | None
    """ker(H) in native x-space: directions the alternative representation cannot see.
    Only a chart that *loses* variables has one (C1). The arc chart instead *gains* a
    variable, and its redundancy ker(B) lives in y-space; it is kept as a diagnostic and
    deliberately not squared away."""
    diagnostics: dict = field(default_factory=dict)
    skip_reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.skip_reason


@dataclass(frozen=True)
class GlobalChart:
    scenario: str
    B: torch.Tensor
    blocks: tuple[BlockChart, ...]
    x_gauge_basis: torch.Tensor
    n_x: int
    n_y: int
    skip_reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.skip_reason

    @property
    def is_identity(self) -> bool:
        return self.scenario == C0


def identity_block(eid: str, entity_type: str, size: int, dtype=torch.float64) -> BlockChart:
    eye = torch.eye(size, dtype=dtype)
    return BlockChart(eid=eid, entity_type=entity_type, B=eye, H=eye, x_gauge_basis=None)


# ---------------------------------------------------------------------------
# C1: line endpoints
# ---------------------------------------------------------------------------

def line_forward(x: torch.Tensor) -> torch.Tensor:
    """(pnt, dir, s, t) -> (start, end)."""
    pnt, direction, s, t = x[0:2], x[2:4], x[4], x[5]
    return torch.cat([pnt + s * direction, pnt + t * direction])


def line_lift(y: torch.Tensor, s0: float, t0: float) -> torch.Tensor:
    """(start, end) -> (pnt, dir, s0, t0) with the baseline parameter range frozen."""
    start, end = y[0:2], y[2:4]
    direction = (end - start) / (t0 - s0)
    pnt = start - s0 * direction
    span = torch.stack([torch.as_tensor(s0, dtype=y.dtype), torch.as_tensor(t0, dtype=y.dtype)])
    return torch.cat([pnt, direction, span])


def line_endpoint_chart(eid: str, x0: torch.Tensor, scale: float) -> BlockChart:
    s0, t0 = float(x0[4]), float(x0[5])
    if abs(t0 - s0) <= DEGENERATE_LINE_TOL * max(scale, 1.0):
        return BlockChart(eid, "Line", torch.zeros(6, 4, dtype=x0.dtype), None, None,
                          dict(span=abs(t0 - s0)), "degenerate_line_span")

    y0 = line_forward(x0)
    H = torch.func.jacfwd(line_forward)(x0)
    B = torch.func.jacfwd(lambda y: line_lift(y, s0, t0))(y0)
    diagnostics, skip = _chart_diagnostics(
        H, B, x0, y0, lambda y: line_lift(y, s0, t0), expected_rank=4, left_inverse="HB"
    )
    diagnostics["span"] = abs(t0 - s0)
    gauge = _null_space(H) if not skip else None
    return BlockChart(eid, "Line", B, H, gauge, diagnostics, skip)


# ---------------------------------------------------------------------------
# C2: arc three-point representation
# ---------------------------------------------------------------------------

def _arc_point(center: torch.Tensor, radius, theta_base: float, sign: float, angle):
    theta = theta_base + sign * angle
    return center + radius * torch.stack([torch.cos(theta), torch.sin(theta)])


def arc_forward(x: torch.Tensor, theta_base: float, sign: float) -> torch.Tensor:
    """(center, r, alpha_s, alpha_e) -> (start, sweep midpoint, end)."""
    center, radius, a_s, a_e = x[0:2], x[2], x[3], x[4]
    return torch.cat([
        _arc_point(center, radius, theta_base, sign, a_s),
        _arc_point(center, radius, theta_base, sign, 0.5 * (a_s + a_e)),
        _arc_point(center, radius, theta_base, sign, a_e),
    ])


def arc_lift(
    y: torch.Tensor, theta_base: float, sign: float, turns_start: float, turns_end: float
) -> torch.Tensor:
    """(start, midpoint, end) -> (center, r, alpha_s, alpha_e) via the circumcircle.

    The 2*pi branch integers are chosen once at the baseline and passed in as constants,
    so the branch selection itself is never differentiated.
    """
    q_s, q_m, q_e = y[0:2], y[2:4], y[4:6]
    M = 2.0 * torch.stack([q_m - q_s, q_e - q_s])
    rhs = torch.stack([q_m @ q_m - q_s @ q_s, q_e @ q_e - q_s @ q_s])
    center = torch.linalg.solve(M, rhs)
    radius = torch.linalg.norm(q_s - center)
    a_s = _angle_of(q_s, center, theta_base, sign, turns_start)
    a_e = _angle_of(q_e, center, theta_base, sign, turns_end)
    return torch.cat([center, radius.reshape(1), a_s.reshape(1), a_e.reshape(1)])


def _angle_of(point: torch.Tensor, center: torch.Tensor, theta_base: float, sign: float, turns: float):
    offset = point - center
    theta = torch.atan2(offset[1], offset[0])
    return sign * (theta - theta_base) + 2.0 * math.pi * turns


def _branch_turns(angle: float, recovered_angle: float) -> float:
    """Choose the 2*pi representative relative to the *recovered* circle.

    Using the input center here is unstable at the atan2 branch cut: the circumcircle
    reconstructed from the three points can differ by roundoff, turning +pi into -pi.
    The branch must therefore be selected after that reconstruction has taken place.
    """
    return round((angle - recovered_angle) / (2.0 * math.pi))


def arc_three_point_chart(eid: str, entity, x0: torch.Tensor, scale: float) -> BlockChart:
    theta_base = math.atan2(entity.yDir, entity.xDir)
    sign = -1.0 if getattr(entity, "clockwise", False) else 1.0
    center, radius = x0[0:2], float(x0[2])
    if radius <= ARC_RADIUS_TOL * max(scale, 1.0):
        return BlockChart(eid, "Arc", torch.zeros(5, 6, dtype=x0.dtype), None, None,
                          dict(radius=radius), "ill_conditioned_arc_three_point")

    y0 = arc_forward(x0, theta_base, sign)
    q_s, q_m, q_e = y0[0:2], y0[2:4], y0[4:6]
    M = 2.0 * torch.stack([q_m - q_s, q_e - q_s])
    sigma = torch.linalg.svdvals(M)
    sigma_min, sigma_max = float(sigma[-1]), float(sigma[0])
    condition = sigma_min / max(sigma_max, 1e-30)
    if sigma_min < max(ARC_SIGMA_ABS_TOL * max(scale, 1.0), ARC_SIGMA_REL_TOL * sigma_max):
        return BlockChart(eid, "Arc", torch.zeros(5, 6, dtype=x0.dtype), None, None,
                          dict(radius=radius, circumcircle_condition=condition),
                          "ill_conditioned_arc_three_point")

    provisional = arc_lift(y0, theta_base, sign, 0.0, 0.0)
    turns_start = _branch_turns(float(x0[3]), float(provisional[3]))
    turns_end = _branch_turns(float(x0[4]), float(provisional[4]))
    lift = lambda y: arc_lift(y, theta_base, sign, turns_start, turns_end)  # noqa: E731

    H = torch.func.jacfwd(lambda x: arc_forward(x, theta_base, sign))(x0)
    B = torch.func.jacfwd(lift)(y0)
    diagnostics, skip = _chart_diagnostics(H, B, x0, y0, lift, expected_rank=5, left_inverse="BH")
    diagnostics.update(radius=radius, circumcircle_condition=condition,
                       angle_branch_start=turns_start, angle_branch_end=turns_end)
    if skip:
        return BlockChart(eid, "Arc", B, H, None, diagnostics, skip)

    # Arc angles are periodic.  Comparing ``lift(y0)`` and ``x0`` with a raw
    # Euclidean norm incorrectly rejects geometrically identical states whose
    # angle representatives differ by exactly 2*pi.  Validate dimensional and
    # angular coordinates separately, with the latter wrapped to [-pi, pi].
    recovered = lift(y0)
    dimensional_error = float(torch.linalg.norm(recovered[:3] - x0[:3]))
    angle_delta = recovered[3:5] - x0[3:5]
    wrapped_angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
    angle_error = float(torch.max(torch.abs(wrapped_angle_delta)))
    recovered_points = arc_forward(recovered, theta_base, sign).reshape(3, 2)
    baseline_points = y0.reshape(3, 2)
    point_error = float(torch.max(torch.linalg.norm(recovered_points - baseline_points, dim=1)))
    diagnostics.update(
        raw_state_reconstruction_error=diagnostics["reconstruction_error"],
        reconstruction_error=dimensional_error,
        angle_reconstruction_error=angle_error,
        point_reconstruction_error=point_error,
    )
    if (
        dimensional_error > ARC_RECONSTRUCTION_TOL * max(scale, 1.0)
        or point_error > ARC_RECONSTRUCTION_TOL * max(scale, 1.0)
        or angle_error > ARC_ANGLE_TOL
    ):
        return BlockChart(eid, "Arc", B, H, None, diagnostics, "arc_reconstruction_failed")
    diagnostics["y_redundancy_dim"] = _null_space(B).shape[1]
    return BlockChart(eid, "Arc", B, H, None, diagnostics, "")


# ---------------------------------------------------------------------------
# Shared chart validation
# ---------------------------------------------------------------------------

def _chart_diagnostics(H, B, x0, y0, lift, *, expected_rank: int, left_inverse: str):
    """Rank, left-inverse and round-trip checks required before a chart may be used."""
    rank_h = int(torch.linalg.matrix_rank(H, rtol=RANK_EPS))
    rank_b = int(torch.linalg.matrix_rank(B, rtol=RANK_EPS))
    product = H @ B if left_inverse == "HB" else B @ H
    identity_error = float(torch.linalg.norm(product - torch.eye(expected_rank, dtype=B.dtype)))
    reconstruction_error = float(torch.linalg.norm(lift(y0) - x0))
    diagnostics = dict(
        rank_H=rank_h, rank_B=rank_b, expected_rank=expected_rank,
        left_inverse=left_inverse, identity_error=identity_error,
        reconstruction_error=reconstruction_error,
    )
    if rank_h != expected_rank or rank_b != expected_rank:
        return diagnostics, "chart_rank_deficient"
    if identity_error > LEFT_INVERSE_TOL:
        return diagnostics, "chart_left_inverse_failed"
    return diagnostics, ""


def _null_space(J: torch.Tensor) -> torch.Tensor:
    _U, S, Vh = torch.linalg.svd(J, full_matrices=True)
    threshold = RANK_EPS * (float(S[0]) if S.numel() else 1.0)
    kept = int((S >= threshold).sum())
    return Vh[kept:].T.contiguous()


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

_NATIVE_BLOCK_SIZE = {"Line": 6, "Circle": 3, "Arc": 5, "Point": 2}
_TARGET_TYPE = {C1: "Line", C2: "Arc"}


def build_global_chart(
    sketch: datalib.Sketch, state: SketchState, scenario: str, scale: float
) -> GlobalChart:
    """Block-diagonal rectangular B over the entities in SketchState order."""
    if scenario not in TRANSFORMED_SCENARIOS:
        raise ValueError(f"not a transformed scenario: {scenario}")
    target = _TARGET_TYPE.get(scenario)
    dtype = state.q.dtype
    blocks: list[BlockChart] = []

    for eid in state.entity_ids:
        entity = sketch.entities[eid]
        entity_type = type(entity).__name__
        block_slice = state.slices[eid]
        size = block_slice.stop - block_slice.start
        if _NATIVE_BLOCK_SIZE.get(entity_type) != size:
            raise AssertionError(
                f"{eid}: {entity_type} occupies {size} state variables, expected "
                f"{_NATIVE_BLOCK_SIZE.get(entity_type)}"
            )
        x0 = state.q[block_slice].detach().clone()
        if entity_type != target:
            blocks.append(identity_block(eid, entity_type, size, dtype))
        elif entity_type == "Line":
            blocks.append(line_endpoint_chart(eid, x0, scale))
        else:
            blocks.append(arc_three_point_chart(eid, entity, x0, scale))

    failed = next((b for b in blocks if not b.usable), None)
    n_x = state.num_dof()
    if failed is not None:
        return GlobalChart(scenario, torch.zeros(n_x, 0, dtype=dtype), tuple(blocks),
                           torch.zeros(n_x, 0, dtype=dtype), n_x, 0, failed.skip_reason)

    n_y = sum(int(b.B.shape[1]) for b in blocks)
    B = torch.zeros(n_x, n_y, dtype=dtype)
    gauge_columns: list[torch.Tensor] = []
    row = col = 0
    for block in blocks:
        rows, cols = int(block.B.shape[0]), int(block.B.shape[1])
        B[row:row + rows, col:col + cols] = block.B
        if block.x_gauge_basis is not None and block.x_gauge_basis.shape[1]:
            embedded = torch.zeros(n_x, block.x_gauge_basis.shape[1], dtype=dtype)
            embedded[row:row + rows, :] = block.x_gauge_basis
            gauge_columns.append(embedded)
        row += rows
        col += cols

    gauge = torch.cat(gauge_columns, dim=1) if gauge_columns else torch.zeros(n_x, 0, dtype=dtype)
    return GlobalChart(scenario, B, tuple(blocks), gauge, n_x, n_y)


def identity_chart(state: SketchState) -> GlobalChart:
    n_x = state.num_dof()
    dtype = state.q.dtype
    return GlobalChart(C0, torch.eye(n_x, dtype=dtype), (), torch.zeros(n_x, 0, dtype=dtype),
                       n_x, n_x)


# ---------------------------------------------------------------------------
# Gauge diagnostics
# ---------------------------------------------------------------------------

def relative_gauge_residual(J: torch.Tensor | None, gauge: torch.Tensor) -> float:
    """How much of the chart's discarded gauge a Jacobian still sees."""
    if J is None or J.shape[0] == 0 or gauge.shape[1] == 0:
        return 0.0
    norm = float(torch.linalg.norm(J))
    return float(torch.linalg.norm(J @ gauge)) / max(norm, 1e-12)


def gauge_residuals(chart: GlobalChart, *, J_x, J_R, J_phi) -> dict:
    return dict(
        gauge_residual_J_x=relative_gauge_residual(J_x, chart.x_gauge_basis),
        gauge_residual_J_R=relative_gauge_residual(J_R, chart.x_gauge_basis),
        gauge_residual_J_phi=relative_gauge_residual(J_phi, chart.x_gauge_basis),
    )


def gauge_is_annihilated(residuals: dict) -> bool:
    return max(residuals.values()) <= GAUGE_RESIDUAL_TOL
