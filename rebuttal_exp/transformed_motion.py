"""The four paper metrics recomputed in an alternative primitive chart.

Mirrors the assembly of ``src.eval.motion_metrics.SketchMotion`` with a rectangular
``B`` right-multiplied into every Jacobian. All scalar formulas are imported from the
existing implementation so the transformed numbers and the paper numbers cannot drift;
only the assembly is duplicated, because the existing class has no injection point.

The response is recomputed from scratch as ``R~ = -pinv(J_x B) J_p`` with the same
singular-value cutoff the native path uses. Multiplying the existing ``R`` by ``B`` would
hide exactly what this experiment measures: the parameterization dependence of the
Moore-Penrose minimum-norm solution.
"""
from __future__ import annotations

import torch

from rebuttal_exp.primitive_charts import (
    GlobalChart,
    gauge_is_annihilated,
    gauge_residuals,
)
from src.metrics.edit_response_distance import (
    _is_relational,
    compute_native_response,
    pairwise_distance_observable,
)
from src.metrics.motion_subspace import (
    leak_angle_from_overlap,
    orthonormal_range,
    padded_subspace_angle,
    placement_sims,
    principal_cos2_sum,
    stack_relation_jacobians_at,
)
from src.sketch_handlers.sketch_solver import compute_null_space

EPS_DEFAULT = 1e-9
RCOND_DEFAULT = 1e-6
DIRECTION_NORM_TOL = 1e-8


def cutoff_pinv(J: torch.Tensor, eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT):
    """Pseudo-inverse under ``compute_native_response``'s combined absolute/relative cutoff."""
    if J.numel() == 0:
        return torch.zeros(J.shape[1], J.shape[0], dtype=J.dtype)
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    eps_eff = max(eps, rcond * float(S[0])) if S.numel() else eps
    keep = S >= eps_eff
    if not bool(keep.any()):
        return torch.zeros(J.shape[1], J.shape[0], dtype=J.dtype)
    return Vh.T[:, keep] @ torch.diag(1.0 / S[keep]) @ U[:, keep].T


class TransformedMotion:
    """Cached full/relational motion objects for one sketch in one chart."""

    def __init__(self, sketch, context, base_eids, chart: GlobalChart,
                 eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT):
        self.sketch = sketch
        self.context = context
        self.chart = chart
        self.eps = eps
        self.skip_reason = chart.skip_reason

        native = compute_native_response(sketch, context, eps=eps)
        self.native = native
        dtype = native.R.dtype
        if chart.n_x != native.J_x.shape[1]:
            raise AssertionError(
                f"chart built for n_x={chart.n_x} but this sketch has {native.J_x.shape[1]}"
            )

        j_phi = pairwise_distance_observable(sketch, base_eids=base_eids).to(dtype=dtype)
        self.m_phi = int(j_phi.shape[0])
        rels = [c for c in sketch.constraints.values() if _is_relational(c)]
        j_rel, _n_rel = stack_relation_jacobians_at(rels, native.state, context)

        self.gauge_residuals = gauge_residuals(
            chart, J_x=native.J_x, J_R=j_rel, J_phi=j_phi
        )
        if not self.skip_reason and not gauge_is_annihilated(self.gauge_residuals):
            self.skip_reason = "line_gauge_not_annihilated"
        if self.skip_reason:
            return

        B = chart.B
        self.n_y = chart.n_y
        j_phi_t = j_phi @ B
        j_x_t = native.J_x @ B
        self.R_t = (
            -cutoff_pinv(j_x_t, eps, rcond) @ native.J_p
            if native.J_p.shape[1] > 0
            else torch.zeros(self.n_y, 0, dtype=dtype)
        )
        self.U_response, self.k_response = orthonormal_range(self.R_t, eps)

        full = j_phi_t @ self.R_t if self.R_t.shape[1] > 0 else torch.zeros(
            self.m_phi, 0, dtype=dtype
        )
        self.U_full, self.k_full = orthonormal_range(full, eps)

        j_rel_t = (j_rel @ B) if j_rel is not None else torch.zeros(0, self.n_y, dtype=dtype)
        self.U_rel, self.k_rel = orthonormal_range(
            j_phi_t @ compute_null_space(j_rel_t, eps=eps), eps
        )

        norms = torch.linalg.norm(full, dim=0)
        keep = norms > DIRECTION_NORM_TOL
        self.dirs = (full[:, keep] / norms[keep]) if bool(keep.any()) else torch.zeros(
            self.m_phi, 0, dtype=dtype
        )
        self.n_dim_cols = int(self.dirs.shape[1])

    @property
    def usable(self) -> bool:
        return not self.skip_reason


def subspace_angle(U_a, k_a, U_b, k_b) -> float:
    cos2 = principal_cos2_sum(U_a, U_b) if (k_a > 0 and k_b > 0) else 0.0
    return padded_subspace_angle(cos2, k_a, k_b)


def leak_dimensional_angle(pred: TransformedMotion, gt: TransformedMotion) -> float:
    """theta_leak in the alternative raw y-space, never moved into observable space."""
    U_P, dim_P = pred.U_response, pred.k_response
    if dim_P == 0:
        return 0.0
    gt_rels = [c for c in gt.sketch.constraints.values() if _is_relational(c)]
    j_rel, n_rel = stack_relation_jacobians_at(gt_rels, pred.native.state, pred.context)
    if j_rel is None or n_rel == 0:
        return 0.0
    N_G, dim_N = orthonormal_range((j_rel @ pred.chart.B).T, pred.eps)
    if dim_N == 0:
        return 0.0
    return leak_angle_from_overlap(N_G.T @ U_P, dim_P)


def transformed_pair_metrics(pred: TransformedMotion, gt: TransformedMotion) -> dict | None:
    if not (pred.usable and gt.usable):
        return None
    if pred.m_phi != gt.m_phi or pred.m_phi == 0:
        return None
    out = dict(
        theta_full=subspace_angle(pred.U_full, pred.k_full, gt.U_full, gt.k_full),
        theta_rel=subspace_angle(pred.U_rel, pred.k_rel, gt.U_rel, gt.k_rel),
        theta_leak=leak_dimensional_angle(pred, gt),
        k_full_pred=pred.k_full,
        k_full_gt=gt.k_full,
        k_rel_pred=pred.k_rel,
        k_rel_gt=gt.k_rel,
        n_dim_pred=pred.n_dim_cols,
        n_dim_gt=gt.n_dim_cols,
        n_y_pred=pred.n_y,
        n_y_gt=gt.n_y,
    )
    out.update(placement_sims(pred.dirs, gt.dirs))
    return out
