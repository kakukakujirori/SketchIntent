"""Shared geometry helpers for reward computations.

Single home for the SVD / Jacobian / motion-subspace primitives used by the
Htheta family of rewards.  The intent is that ``response_metric_reward.py``
composes thin scoring wrappers from these primitives; every "build a motion
subspace" or "compare two subspaces" piece of math should live here.

Current consumers include ResponseIntentReward and ProbeResponseSVDReward.
"""
from __future__ import annotations

import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _is_relational,
    constraint_jacobian,
    pairwise_distance_observable,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_solver import SketchState, compute_null_space


def orthonormal_range(R: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    """SVD-based orthonormal basis of range(R), keeping singular values >= eps.

    Returns (U, k) where U is (R.shape[0], k) and the columns span range(R).
    """
    if R.numel() == 0 or R.shape[1] == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    U, S, _Vh = torch.linalg.svd(R, full_matrices=False)
    keep = int((S >= eps).sum().item())
    if keep == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    return U[:, :keep].contiguous(), keep


def stack_relation_jacobians_at(
    rels,
    state: SketchState,
    context: SketchContext,
) -> tuple[torch.Tensor | None, int]:
    """Concatenate relation-constraint Jacobians evaluated at the given state."""
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


def motion_full_subspace(
    sketch: datalib.Sketch,
    native: NativeResponse,
    eps: float,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of the *full* motion subspace in observable space.

    "Full" here means the motion subspace driven by the full first-order
    response map ``R_C`` (dimensional + relational null-space columns
    together), projected through the pairwise-distance Jacobian ``J_phi`` so
    that Pred and GT can be compared in a common, sketch-topology-independent
    observable coordinate system:

        W_C := range(J_phi @ R_C)

    Contrast this with ``motion_relational_subspace``, which uses only
    ``null(J_R)`` and ignores the placement of dimensional drivers.

    Returns the orthonormal basis ``U`` of ``W_C`` and its dimension ``k``.
    """
    J_phi = pairwise_distance_observable(
        sketch,
        dtype=native.R.dtype,
    ).to(dtype=native.R.dtype)
    if J_phi.shape[0] == 0 or native.R.shape[1] == 0:
        return torch.zeros(J_phi.shape[0], 0, dtype=native.R.dtype), 0
    return orthonormal_range(J_phi @ native.R, eps)


def motion_relational_subspace(
    sketch: datalib.Sketch,
    state: SketchState,
    context: SketchContext,
    eps: float,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of the *relational* motion subspace in observable space.

    "Relational" means: only the primitive motions allowed by the sketch's
    relational constraints, with dimensional drivers deliberately dropped:

        N_C := null(J_R)               # motions allowed by relations alone
        W_C := range(J_phi @ N_C)      # same motions, in pairwise-distance
                                       # observable space

    Jacobians are evaluated at ``state``.  When comparing two sketches at the
    same geometric configuration (e.g. Pred and GT, where Pred's primitive
    geometry has been overwritten with the original), the caller should pass
    the same ``state`` for both sketches so that the two subspaces live in
    coordinates that can be compared with a principal-angle SVD.

    Contrast with ``motion_full_subspace``, which uses the full first-order
    response map ``R_C`` and therefore mixes dimensional-driver placement into
    the comparison.
    """
    n_x = state.num_dof()
    dtype = state.q.dtype

    rels = [c for c in sketch.constraints.values() if _is_relational(c)]
    J_R, _ = stack_relation_jacobians_at(rels, state, context)
    if J_R is None:
        J_R = torch.zeros(0, n_x, dtype=dtype)
    N_C = compute_null_space(J_R, eps=eps)

    J_phi = pairwise_distance_observable(sketch).to(dtype=dtype)
    if J_phi.shape[0] == 0:
        return torch.zeros(0, 0, dtype=dtype), 0
    return orthonormal_range(J_phi @ N_C, eps)


def principal_cos2_sum(U_P: torch.Tensor, U_G: torch.Tensor) -> float:
    """Σ cos²θ_i between two orthonormal subspaces.

    Computes the sum of squared principal cosines between ``range(U_P)`` and
    ``range(U_G)`` via the SVD of ``U_G^T @ U_P``.  Returns 0 if either
    subspace is empty.

    This is the shared building block for the angle scalars used by Hθ:

    - padded principal-angle distance: ``(d_norm - cos2_sum) / d_norm``
    - symmetric projection distance:   ``(dim_P + dim_G - 2 cos2_sum) / d_norm``
    - asymmetric subset distances:     ``(dim_? - cos2_sum) / dim_?``

    Callers handle normalization by the relevant dimensions; this helper
    intentionally returns the raw quantity.
    """
    if U_P.shape[1] == 0 or U_G.shape[1] == 0:
        return 0.0
    align = U_G.T @ U_P
    sigmas = torch.linalg.svdvals(align).clamp(0.0, 1.0)
    return float((sigmas ** 2).sum())
