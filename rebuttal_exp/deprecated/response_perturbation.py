"""Admissible alternatives to the minimum-norm response R0 = -pinv(J_x) J_p.

The linearised system ``J_x R = -J_p`` is under-determined whenever ``ker(J_x)`` is
non-trivial, so every ``R0 + N Z`` with ``N = ker(J_x)`` is an equally valid first-order
response. Two arms sample that freedom:

    main            (paper section 4.5) the free motion is sized in *observable* space,
                    so ``eta`` means the same physical thing across sketches. By
                    construction E lies in range(J_phi N), which means the arm can never
                    excite a direction the observable cannot see.

    observable_null (auxiliary) the free motion is drawn from G = ker([J_x; J_phi]) and
                    sized in *raw* space. Both J_x R and J_phi R are then preserved
                    exactly, so theta_full / theta_rel / S_place cannot move and only
                    theta_leak -- which compares range(R) in raw primitive tangent space --
                    is exposed. The two arms' eta are not comparable and must not be
                    plotted on one axis.

G is the null space of the *stacked* Jacobian rather than a numerical intersection of two
separately computed null spaces, and it is not called a gauge basis: ker(J_phi) also holds
directions such as global rotation, which the pairwise-distance observable cannot see but
GT relations can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MAIN_ARM = "main"
OBSERVABLE_NULL_ARM = "observable_null"
ARMS = (MAIN_ARM, OBSERVABLE_NULL_ARM)

EPS_DEFAULT = 1e-9
RCOND_DEFAULT = 1e-6
ZERO_COLUMN_TOL = 1e-12
CONSISTENCY_TOL = 1e-6

NOT_UNDERCONSTRAINED = "not_underconstrained"
INCONSISTENT_LINEAR_SYSTEM = "inconsistent_linear_system"
NO_OBSERVABLE_FREE_DIRECTION = "no_observable_free_direction"
NO_OBSERVABLE_NULL_DIRECTION = "no_observable_null_direction"
NO_NONZERO_OBSERVABLE_RESPONSE = "no_nonzero_observable_response"
NO_NONZERO_RAW_RESPONSE = "no_nonzero_raw_response"


def effective_cutoff(singular_values: torch.Tensor, eps: float, rcond: float) -> float:
    """``compute_native_response``'s combined absolute/relative singular-value threshold."""
    if singular_values.numel() == 0:
        return eps
    return max(eps, rcond * float(singular_values[0]))


def cutoff_null_space(
    J: torch.Tensor, eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT
) -> torch.Tensor:
    """Orthonormal basis of the right null space, under the native response's cutoff.

    ``full_matrices=True`` is required: the economy SVD of a wide matrix returns only
    ``m`` right singular vectors and silently drops the rest of the null space.
    """
    n = int(J.shape[1])
    if J.shape[0] == 0:
        return torch.eye(n, dtype=J.dtype)
    _U, S, Vh = torch.linalg.svd(J, full_matrices=True)
    cutoff = effective_cutoff(S, eps, rcond)
    kept = torch.cat([S < cutoff, torch.ones(Vh.shape[0] - S.numel(), dtype=torch.bool)])
    return Vh[kept].T.contiguous()


def cutoff_range(
    A: torch.Tensor, eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT
) -> torch.Tensor:
    """Orthonormal basis of range(A) under the same cutoff convention."""
    if A.numel() == 0 or A.shape[1] == 0:
        return torch.zeros(A.shape[0], 0, dtype=A.dtype)
    U, S, _Vh = torch.linalg.svd(A, full_matrices=False)
    keep = int((S >= effective_cutoff(S, eps, rcond)).sum())
    return U[:, :keep].contiguous() if keep else torch.zeros(A.shape[0], 0, dtype=A.dtype)


def cutoff_pinv(
    A: torch.Tensor, eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT
) -> torch.Tensor:
    if A.numel() == 0:
        return torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = S >= effective_cutoff(S, eps, rcond)
    if not bool(keep.any()):
        return torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype)
    return Vh.T[:, keep] @ torch.diag(1.0 / S[keep]) @ U[:, keep].T


def stacked_observable_null(
    J_x: torch.Tensor, J_phi: torch.Tensor,
    eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT,
) -> torch.Tensor:
    """G = ker([J_x; J_phi]), taken from one SVD of the stacked matrix."""
    stacked = torch.cat([J_x, J_phi.to(dtype=J_x.dtype)], dim=0)
    return cutoff_null_space(stacked, eps, rcond)


@dataclass(frozen=True)
class ResponseGeometry:
    """Everything about one annotation's response freedom that no sample changes."""

    R0: torch.Tensor
    J_x: torch.Tensor
    J_p: torch.Tensor
    J_phi: torch.Tensor
    N: torch.Tensor
    U_free: torch.Tensor
    K_pinv: torch.Tensor
    G: torch.Tensor
    observable_column_norms: torch.Tensor
    raw_column_norms: torch.Tensor
    r0_frobenius: float
    raw_null_dim: int
    observable_free_rank: int
    observable_null_dim: int
    consistency_residual: float
    n_zero_observable_columns: int
    n_zero_raw_columns: int
    skip_reason: str

    def arm_skip_reason(self, arm: str) -> str:
        if self.skip_reason:
            return self.skip_reason
        if arm == MAIN_ARM and self.observable_free_rank == 0:
            return NO_OBSERVABLE_FREE_DIRECTION
        if arm == MAIN_ARM and (
            self.R0.shape[1] == 0
            or self.n_zero_observable_columns == self.R0.shape[1]
        ):
            return NO_NONZERO_OBSERVABLE_RESPONSE
        if arm == OBSERVABLE_NULL_ARM and self.observable_null_dim == 0:
            return NO_OBSERVABLE_NULL_DIRECTION
        if arm == OBSERVABLE_NULL_ARM and (
            self.R0.shape[1] == 0 or self.n_zero_raw_columns == self.R0.shape[1]
        ):
            return NO_NONZERO_RAW_RESPONSE
        return ""

    def as_record(self) -> dict:
        return dict(
            raw_null_dim=self.raw_null_dim,
            observable_free_rank=self.observable_free_rank,
            observable_null_dim=self.observable_null_dim,
            consistency_residual=self.consistency_residual,
            r0_frobenius=self.r0_frobenius,
            n_zero_observable_columns=self.n_zero_observable_columns,
            n_zero_raw_columns=self.n_zero_raw_columns,
            n_drivers=int(self.R0.shape[1]),
            skip_reason=self.skip_reason,
        )


def build_response_geometry(
    native, J_phi: torch.Tensor,
    eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT,
    consistency_tol: float = CONSISTENCY_TOL,
) -> ResponseGeometry:
    dtype = native.R.dtype
    J_phi = J_phi.to(dtype=dtype)
    R0, J_x, J_p = native.R, native.J_x, native.J_p

    residual = torch.linalg.norm(J_x @ R0 + J_p) if J_p.numel() else torch.zeros((), dtype=dtype)
    denominator = max(float(torch.linalg.norm(J_p)) if J_p.numel() else 0.0, 1.0)
    consistency_residual = float(residual) / denominator

    N = cutoff_null_space(J_x, eps, rcond)
    K = J_phi @ N
    U_free = cutoff_range(K, eps, rcond)
    G = stacked_observable_null(J_x, J_phi, eps, rcond)
    observable_columns = J_phi @ R0
    observable_column_norms = torch.linalg.norm(observable_columns, dim=0)
    raw_column_norms = torch.linalg.norm(R0, dim=0)

    skip_reason = ""
    if int(N.shape[1]) == 0:
        skip_reason = NOT_UNDERCONSTRAINED
    elif consistency_residual > consistency_tol:
        skip_reason = INCONSISTENT_LINEAR_SYSTEM

    return ResponseGeometry(
        R0=R0,
        J_x=J_x,
        J_p=J_p,
        J_phi=J_phi,
        N=N,
        U_free=U_free,
        K_pinv=cutoff_pinv(K, eps, rcond),
        G=G,
        observable_column_norms=observable_column_norms,
        raw_column_norms=raw_column_norms,
        r0_frobenius=float(torch.linalg.norm(R0)) if R0.numel() else 0.0,
        raw_null_dim=int(N.shape[1]),
        observable_free_rank=int(U_free.shape[1]),
        observable_null_dim=int(G.shape[1]),
        consistency_residual=consistency_residual,
        n_zero_observable_columns=int((observable_column_norms < ZERO_COLUMN_TOL).sum()),
        n_zero_raw_columns=int((raw_column_norms < ZERO_COLUMN_TOL).sum()),
        skip_reason=skip_reason,
    )


def _unit_columns(basis: torch.Tensor, n_columns: int, seed: int) -> torch.Tensor:
    """One isotropic unit vector per driving column, drawn in the basis' coordinates."""
    generator = np.random.default_rng(seed)
    coefficients = torch.as_tensor(
        generator.standard_normal((int(basis.shape[1]), n_columns)), dtype=basis.dtype
    )
    norms = torch.linalg.norm(coefficients, dim=0)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    return basis @ (coefficients / norms)


def perturbed_response(
    geometry: ResponseGeometry, arm: str, eta: float, seed: int,
    eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT,
) -> tuple[torch.Tensor, dict]:
    if arm == MAIN_ARM:
        return _observable_normalized_response(geometry, eta, seed, eps, rcond)
    if arm == OBSERVABLE_NULL_ARM:
        return _observable_null_response(geometry, eta, seed)
    raise ValueError(f"unknown arm: {arm}")


def _observable_normalized_response(
    geometry: ResponseGeometry, eta: float, seed: int, eps: float, rcond: float
) -> tuple[torch.Tensor, dict]:
    """R0 + N K^+ E with E_j = eta * ||(J_phi R0)_j|| * unit vector in range(J_phi N)."""
    R0 = geometry.R0
    if eta == 0.0 or geometry.observable_free_rank == 0 or R0.shape[1] == 0:
        return R0, _sample_diagnostics(geometry, R0)
    directions = _unit_columns(geometry.U_free, int(R0.shape[1]), seed)
    scales = geometry.observable_column_norms.clone()
    scales[scales < ZERO_COLUMN_TOL] = 0.0
    E = directions * (eta * scales)
    R_eta = R0 + geometry.N @ (geometry.K_pinv @ E)
    return R_eta, _sample_diagnostics(geometry, R_eta)


def _observable_null_response(
    geometry: ResponseGeometry, eta: float, seed: int
) -> tuple[torch.Tensor, dict]:
    """R0 + eta * ||R0_j|| * unit vector in ker([J_x; J_phi]), column by column."""
    R0 = geometry.R0
    if eta == 0.0 or geometry.observable_null_dim == 0 or R0.shape[1] == 0:
        return R0, _sample_diagnostics(geometry, R0)
    directions = _unit_columns(geometry.G, int(R0.shape[1]), seed)
    scales = geometry.raw_column_norms.clone()
    scales[scales < ZERO_COLUMN_TOL] = 0.0
    R_eta = R0 + directions * (eta * scales)
    return R_eta, _sample_diagnostics(geometry, R_eta)


def _sample_diagnostics(geometry: ResponseGeometry, R_eta: torch.Tensor) -> dict:
    delta = R_eta - geometry.R0
    denominator = max(
        float(torch.linalg.norm(geometry.J_p)) if geometry.J_p.numel() else 0.0,
        1.0,
    )
    rhs = geometry.J_p
    total_residual = geometry.J_x @ R_eta + rhs
    nullspace_residual = geometry.J_x @ delta
    return dict(
        raw_perturbation_ratio=float(torch.linalg.norm(delta))
        / max(geometry.r0_frobenius, ZERO_COLUMN_TOL),
        constraint_residual=float(torch.linalg.norm(total_residual)) / denominator,
        nullspace_residual=float(torch.linalg.norm(nullspace_residual)) / denominator,
    )


def relation_visibility(J_R: torch.Tensor | None, G: torch.Tensor,
                        eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT) -> dict:
    """How much of the observable-null freedom the GT relations can still see.

    ``rank == 0`` marks directions invisible to both this GT relation Jacobian and the
    observable.  This is a visibility label, not a claim that every such direction is a
    primitive-parameterization gauge (global rigid motion can also be invisible).
    """
    if J_R is None or J_R.numel() == 0 or G.shape[1] == 0:
        return dict(relation_null_overlap_norm=0.0, relation_null_overlap_rank=0)
    overlap = J_R.to(dtype=G.dtype) @ G
    U = cutoff_range(overlap, eps, rcond)
    return dict(
        relation_null_overlap_norm=float(torch.linalg.norm(overlap)),
        relation_null_overlap_rank=int(U.shape[1]),
    )
