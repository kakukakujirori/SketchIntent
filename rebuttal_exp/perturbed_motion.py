"""The paper metrics evaluated against a substituted response, without re-deriving them.

Experiment C had to duplicate ``SketchMotion``'s assembly because injecting a rectangular
``B`` changes every Jacobian. Experiment D only swaps ``R``, so the existing classes and
scalar formulas are reused as they are; ``motion_with_response`` rebuilds exactly the
fields that depend on ``R`` and shares the rest with the unperturbed motion.

Calling ``pair_paper_metrics`` once per sample is not viable: ``leak_dimensional_angle``
rebuilds the GT relation Jacobian at the Pred state by autodiff every call, and that
quantity does not depend on ``R``. ``PairEvaluator`` caches it, and ``PredResponseState``
caches the Pred-side subspaces so both GT references reuse one set of SVDs. Agreement with
``pair_paper_metrics`` at eta = 0 is asserted by unit tests and by the runner's gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from src.eval.motion_metrics import SketchMotion, subspace_angle
from src.metrics.edit_response_distance import _is_relational
from src.metrics.motion_subspace import (
    leak_angle_from_overlap,
    orthonormal_range,
    placement_sims,
    stack_relation_jacobians_at,
)

DIRECTION_NORM_TOL = 1e-8


@dataclass(frozen=True)
class PredResponseState:
    """The Pred-side quantities that change when the response changes."""

    U_full: torch.Tensor
    k_full: int
    dirs: torch.Tensor
    n_dim_cols: int
    U_response: torch.Tensor
    k_response: int
    m_phi: int


def pred_response_state(R: torch.Tensor, j_phi: torch.Tensor, eps: float) -> PredResponseState:
    m_phi = int(j_phi.shape[0])
    full = j_phi @ R if R.shape[1] > 0 else torch.zeros(m_phi, 0, dtype=R.dtype)
    U_full, k_full = orthonormal_range(full, eps)
    norms = torch.linalg.norm(full, dim=0)
    keep = norms > DIRECTION_NORM_TOL
    dirs = (full[:, keep] / norms[keep]) if bool(keep.any()) else torch.zeros(
        m_phi, 0, dtype=full.dtype
    )
    U_response, k_response = orthonormal_range(R, eps)
    return PredResponseState(
        U_full=U_full, k_full=k_full, dirs=dirs, n_dim_cols=int(dirs.shape[1]),
        U_response=U_response, k_response=k_response, m_phi=m_phi,
    )


def motion_with_response(base: SketchMotion, R: torch.Tensor, j_phi: torch.Tensor) -> SketchMotion:
    """A SketchMotion identical to ``base`` except for the response and what it induces."""
    state = pred_response_state(R, j_phi, base.eps)
    motion = SketchMotion.__new__(SketchMotion)
    motion.sketch = base.sketch
    motion.context = base.context
    motion.eps = base.eps
    motion.m_phi = base.m_phi
    motion.U_rel = base.U_rel
    motion.k_rel = base.k_rel
    motion.native = replace(base.native, R=R)
    motion.U_full = state.U_full
    motion.k_full = state.k_full
    motion.dirs = state.dirs
    motion.n_dim_cols = state.n_dim_cols
    return motion


class PairEvaluator:
    """One directed (Pred, GT) pair, with everything independent of the response cached."""

    def __init__(self, pred: SketchMotion, gt: SketchMotion):
        self.pred = pred
        self.gt = gt
        self.eps = pred.eps
        self.comparable = pred.m_phi == gt.m_phi and pred.m_phi != 0
        self.theta_rel = float("nan")
        self.J_R_at_pred, self.N_G, self.dim_N = None, None, 0
        if not self.comparable:
            return
        self.theta_rel = subspace_angle(pred.U_rel, pred.k_rel, gt.U_rel, gt.k_rel)
        gt_rels = [c for c in gt.sketch.constraints.values() if _is_relational(c)]
        self.J_R_at_pred, n_rel = stack_relation_jacobians_at(
            gt_rels, pred.native.state, pred.context
        )
        if self.J_R_at_pred is None or n_rel == 0:
            self.N_G, self.dim_N = None, 0
        else:
            self.N_G, self.dim_N = orthonormal_range(self.J_R_at_pred.T, pred.eps)

    def leak(self, state: PredResponseState) -> float:
        if state.k_response == 0 or self.dim_N == 0:
            return 0.0
        return leak_angle_from_overlap(self.N_G.T @ state.U_response, state.k_response)

    def metrics(self, state: PredResponseState) -> dict | None:
        if not self.comparable:
            return None
        out = dict(
            theta_full=subspace_angle(state.U_full, state.k_full, self.gt.U_full, self.gt.k_full),
            theta_rel=self.theta_rel,
            theta_leak=self.leak(state),
            k_full_pred=state.k_full,
            k_full_gt=self.gt.k_full,
            k_rel_pred=self.pred.k_rel,
            k_rel_gt=self.gt.k_rel,
            k_response_pred=state.k_response,
            n_dim_pred=state.n_dim_cols,
            n_dim_gt=self.gt.n_dim_cols,
        )
        out.update(placement_sims(state.dirs, self.gt.dirs))
        return out
