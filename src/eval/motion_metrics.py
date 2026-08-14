"""Motion-subspace metrics reported in the paper."""
from __future__ import annotations

import sys

import torch

from .config import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.edit_response_distance import (  # noqa: E402
    _is_relational,
    compute_native_response,
    pairwise_distance_observable,
)
from src.metrics.motion_subspace import (  # noqa: E402
    leak_angle_from_overlap,
    motion_relational_subspace,
    orthonormal_range,
    padded_subspace_angle,
    placement_sims,
    principal_cos2_sum,
    stack_relation_jacobians_at,
)
from src.sketch_handlers.context import SketchContext  # noqa: E402

EPS_DEFAULT = 1e-9


class SketchMotion:
    """Cached full/relational motion objects for one sketch."""

    def __init__(self, sketch, context: SketchContext, base_eids, eps: float = EPS_DEFAULT):
        self.eps = eps
        # Kept for the relation-leak metric, which needs this sketch's native
        # response (R, state) and, for the GT side, its relational constraints.
        self.sketch = sketch
        self.context = context
        native = compute_native_response(sketch, context, eps=eps)
        self.native = native
        j_phi = pairwise_distance_observable(sketch, base_eids=base_eids).to(dtype=native.R.dtype)
        self.m_phi = int(j_phi.shape[0])

        full = j_phi @ native.R if native.R.shape[1] > 0 else torch.zeros(
            self.m_phi, 0, dtype=native.R.dtype
        )
        self.U_full, self.k_full = orthonormal_range(full, eps)
        self.U_rel, self.k_rel = motion_relational_subspace(
            sketch, native.state, context, eps, J_phi=j_phi
        )

        norms = torch.linalg.norm(full, dim=0)
        keep = norms > 1e-8
        self.dirs = (full[:, keep] / norms[keep]) if bool(keep.any()) else torch.zeros(
            self.m_phi, 0, dtype=full.dtype
        )
        self.n_dim_cols = int(self.dirs.shape[1])


def subspace_angle(U_a: torch.Tensor, k_a: int, U_b: torch.Tensor, k_b: int) -> float:
    cos2 = principal_cos2_sum(U_a, U_b) if (k_a > 0 and k_b > 0) else 0.0
    return padded_subspace_angle(cos2, k_a, k_b)


def leak_dimensional_angle(pred: SketchMotion, gt: SketchMotion, eps: float = EPS_DEFAULT) -> float:
    r"""theta_leak^dim: dimensional-intent leakage of Pred against GT relations.

    RMS cosine between the directions Pred allows through its (dimensional)
    response map and the GT relation-forbidden directions (the row space of the
    GT relation Jacobians):

        theta_leak^dim ** 2 = || orth(row(J_R^G))^T U_P ||_F^2 / dim(U_P)

    Jacobians are evaluated at Pred's primitive state, with GT's relational
    constraints. Shares the scalar formula with the reward's
    ``_leak_relational_angle`` via ``src.metrics.motion_subspace.leak_angle_from_overlap``,
    so the held-out-human reference matches the model-side leak angle exactly.
    """
    U_P, dim_P = orthonormal_range(pred.native.R, eps)
    if dim_P == 0:
        return 0.0
    gt_rels = [c for c in gt.sketch.constraints.values() if _is_relational(c)]
    J_R, n_R = stack_relation_jacobians_at(gt_rels, pred.native.state, pred.context)
    if J_R is None or n_R == 0:
        return 0.0
    N_G, dim_N = orthonormal_range(J_R.T, eps)
    if dim_N == 0:
        return 0.0
    overlap = N_G.T @ U_P
    return leak_angle_from_overlap(overlap, dim_P)


def pair_paper_metrics(pred: SketchMotion, gt: SketchMotion) -> dict | None:
    if pred.m_phi != gt.m_phi or pred.m_phi == 0:
        return None
    out = dict(
        theta_full=subspace_angle(pred.U_full, pred.k_full, gt.U_full, gt.k_full),
        theta_rel=subspace_angle(pred.U_rel, pred.k_rel, gt.U_rel, gt.k_rel),
        theta_leak=leak_dimensional_angle(pred, gt, pred.eps),
        k_full_pred=pred.k_full,
        k_full_gt=gt.k_full,
        k_rel_pred=pred.k_rel,
        k_rel_gt=gt.k_rel,
        n_dim_pred=pred.n_dim_cols,
        n_dim_gt=gt.n_dim_cols,
    )
    out.update(placement_sims(pred.dirs, gt.dirs))
    return out
