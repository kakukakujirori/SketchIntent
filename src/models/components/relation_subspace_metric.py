"""Relation-only observable subspace distance used by RL rewards."""
from __future__ import annotations

import math

import torch
import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _is_relational,
    constraint_jacobian,
    pairwise_distance_observable,
)
from src.sketch_handlers.sketch_solver import compute_null_space


RELATION_SUBSPACE_DISTANCE_MODES = {
    "symmetric",
    "pred_subset_gt",
    "mixed_subset",
}


def orthonormal_range(R: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    if R.numel() == 0 or R.shape[1] == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    U, S, _Vh = torch.linalg.svd(R, full_matrices=False)
    keep = int((S >= eps).sum().item())
    if keep == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    return U[:, :keep].contiguous(), keep


def stack_relation_jacobians_at(
    rels,
    state,
    context,
) -> tuple[torch.Tensor | None, int]:
    rows: list[torch.Tensor] = []
    for rel in rels:
        j = constraint_jacobian(rel, state, context)
        if j is None or j.shape[0] == 0:
            continue
        rows.append(j)
    if not rows:
        return None, 0
    Jr = torch.cat(rows, dim=0)
    return Jr, int(Jr.shape[0])


def compute_relation_subspace_distance(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    context,
    eps: float,
    distance_mode: str = "symmetric",
    reverse_weight: float = 0.5,
) -> dict[str, float | int]:
    """Compare relation-only null-space structure in observable space.

    This is the reward-oriented replacement for the full ``d_sub`` term of
    H/Hθ: it ignores dimensional constraint placement and compares only the
    primitive motions allowed by relational constraints.
    """
    if distance_mode not in RELATION_SUBSPACE_DISTANCE_MODES:
        raise ValueError(f"unknown relation subspace distance mode: {distance_mode!r}")

    state = pred_native.state
    n_x = state.num_dof()
    dtype = state.q.dtype

    pred_rels = [c for c in pred_sketch.constraints.values() if _is_relational(c)]
    gt_rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]

    J_R_pred, _ = stack_relation_jacobians_at(pred_rels, state, context)
    J_R_gt, _ = stack_relation_jacobians_at(gt_rels, state, context)

    if J_R_pred is None:
        J_R_pred = torch.zeros(0, n_x, dtype=dtype)
    if J_R_gt is None:
        J_R_gt = torch.zeros(0, n_x, dtype=dtype)

    N_pred = compute_null_space(J_R_pred, eps=eps)
    N_gt = compute_null_space(J_R_gt, eps=eps)

    J_phi = pairwise_distance_observable(pred_sketch).to(dtype=dtype)
    if J_phi.shape[0] == 0:
        return {
            "d_relstruct": 0.0,
            "d_relstruct_pred_subset_gt": 0.0,
            "d_relstruct_gt_subset_pred": 0.0,
            "relstruct_dim_pred": 0,
            "relstruct_dim_gt": 0,
        }

    W_pred = J_phi @ N_pred
    W_gt = J_phi @ N_gt

    U_pred, dim_pred = orthonormal_range(W_pred, eps)
    U_gt, dim_gt = orthonormal_range(W_gt, eps)

    if dim_pred > 0 and dim_gt > 0:
        align = U_gt.T @ U_pred
        sigmas = torch.linalg.svdvals(align).clamp(0.0, 1.0)
        cos2_sum = float((sigmas ** 2).sum())
    else:
        cos2_sum = 0.0

    pred_subset_gt_sq = 0.0
    if dim_pred > 0:
        pred_subset_gt_sq = (dim_pred - cos2_sum) / float(dim_pred)

    gt_subset_pred_sq = 0.0
    if dim_gt > 0:
        gt_subset_pred_sq = (dim_gt - cos2_sum) / float(dim_gt)

    d_norm = float(max(dim_pred, dim_gt, 1))
    if distance_mode == "symmetric":
        d_relstruct_sq = (dim_pred + dim_gt - 2.0 * cos2_sum) / d_norm
    elif distance_mode == "pred_subset_gt":
        d_relstruct_sq = pred_subset_gt_sq
    else:
        d_relstruct_sq = pred_subset_gt_sq + reverse_weight * gt_subset_pred_sq

    return {
        "d_relstruct": math.sqrt(max(d_relstruct_sq, 0.0)),
        "d_relstruct_pred_subset_gt": math.sqrt(max(pred_subset_gt_sq, 0.0)),
        "d_relstruct_gt_subset_pred": math.sqrt(max(gt_subset_pred_sq, 0.0)),
        "relstruct_dim_pred": int(dim_pred),
        "relstruct_dim_gt": int(dim_gt),
    }
