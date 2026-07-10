"""Motion-subspace metrics reported in the paper."""
from __future__ import annotations

import math
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
    padded_subspace_angle,
)
from src.models.rewards.utils.geometry import (  # noqa: E402
    motion_relational_subspace,
    orthonormal_range,
    principal_cos2_sum,
    stack_relation_jacobians_at,
)
from src.sketch_handlers.context import SketchContext  # noqa: E402

EPS_DEFAULT = 1e-9
_TINY = 1e-12


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


def _hungarian_matched(C: torch.Tensor) -> torch.Tensor:
    from scipy.optimize import linear_sum_assignment

    ri, ci = linear_sum_assignment((1.0 - C).numpy())
    return C[ri, ci]


def placement_sims(A: torch.Tensor, B: torch.Tensor) -> dict:
    ka, kb = int(A.shape[1]), int(B.shape[1])
    if ka == 0 and kb == 0:
        return dict(placement_sim=1.0, asgn_min=None, set_sim=1.0, omp_sim=1.0)
    if ka == 0 or kb == 0:
        return dict(placement_sim=0.0, asgn_min=None, set_sim=0.0, omp_sim=0.0)
    C = (A.T @ B).abs()
    matched = _hungarian_matched(C)
    asgn = float(matched.sum())
    set_sim = float((C.max(dim=0).values.mean() + C.max(dim=1).values.mean()) / 2)
    return dict(
        placement_sim=asgn / max(ka, kb),
        asgn_min=asgn / min(ka, kb),
        set_sim=set_sim,
        omp_sim=_omp_sim(A, B),
    )


def _omp_sim(Ua: torch.Tensor, Ub: torch.Tensor, tol: float = 1e-9) -> float:
    A, B = Ua.clone(), Ub.clone()
    avail_a = list(range(A.shape[1]))
    avail_b = list(range(B.shape[1]))
    num = ea = eb = 0.0
    while avail_a and avail_b:
        C = (A[:, avail_a].T @ B[:, avail_b]).abs()
        flat = int(torch.argmax(C))
        i, j = flat // C.shape[1], flat % C.shape[1]
        best = float(C[i, j])
        if best < tol:
            break
        ia, jb = avail_a.pop(i), avail_b.pop(j)
        num += best
        ea += float(A[:, ia] @ A[:, ia])
        eb += float(B[:, jb] @ B[:, jb])
        da = A[:, ia] / max(float(A[:, ia].norm()), _TINY)
        db = B[:, jb] / max(float(B[:, jb].norm()), _TINY)
        if avail_a:
            Ra = A[:, avail_a]
            A[:, avail_a] = Ra - torch.outer(da, da @ Ra)
        if avail_b:
            Rb = B[:, avail_b]
            B[:, avail_b] = Rb - torch.outer(db, db @ Rb)
    ea_left = float(sum(float(A[:, i] @ A[:, i]) for i in avail_a))
    eb_left = float(sum(float(B[:, i] @ B[:, i]) for i in avail_b))
    den = math.sqrt(max((ea + ea_left) * (eb + eb_left), _TINY))
    return num / den


def leak_dimensional_angle(pred: SketchMotion, gt: SketchMotion, eps: float = EPS_DEFAULT) -> float:
    r"""theta_leak^dim: dimensional-intent leakage of Pred against GT relations.

    RMS cosine between the directions Pred allows through its (dimensional)
    response map and the GT relation-forbidden directions (the row space of the
    GT relation Jacobians):

        theta_leak^dim ** 2 = || orth(row(J_R^G))^T U_P ||_F^2 / dim(U_P)

    Jacobians are evaluated at Pred's primitive state, with GT's relational
    constraints. This is a direct port of
    ``src.models.rewards.response_metric_reward._leak_relational_angle`` (the
    reward keeps its historical name) so the held-out-human reference matches the
    model-side leak angle.
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
