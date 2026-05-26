"""Shared geometry helpers for reward computations.

Single home for the SVD / Jacobian / observable-subspace primitives that were duplicated.
Current consumers include ResponseIntentReward, ProbeResponseSVDReward, and DFeasReward.
"""
from __future__ import annotations

import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    constraint_jacobian,
    pairwise_distance_observable,
)


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
    state,
    context,
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


def observable_subspace(
    sketch: datalib.Sketch,
    native: NativeResponse,
    eps: float,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of range(J_phi R) in pairwise-distance space."""
    J_phi = pairwise_distance_observable(
        sketch,
        dtype=native.R.dtype,
    ).to(dtype=native.R.dtype)
    if J_phi.shape[0] == 0 or native.R.shape[1] == 0:
        return torch.zeros(J_phi.shape[0], 0, dtype=native.R.dtype), 0
    return orthonormal_range(J_phi @ native.R, eps)
