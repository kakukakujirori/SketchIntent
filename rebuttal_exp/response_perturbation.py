"""Admissible alternatives to the minimum-norm response R0 = -pinv(J_x) J_p.

The linearised system ``J_x R = -J_p`` is under-determined whenever ``ker(J_x)`` is
non-trivial, so every ``R0 + N Z`` with ``N = ker(J_x)`` is an equally valid first-order
response. We sample that freedom one driver column at a time:

    n_j  uniform on the unit sphere of ker(J_x)
    Delta_j = eta * ||(J_phi R0)_j|| * n_j / ||J_phi n_j||

so ``||J_phi Delta_j|| = eta ||(J_phi R0)_j||`` exactly: eta is an *observable-space* size
and therefore means the same physical thing across sketches, which a Gaussian coefficient in
raw state coordinates (mixing mm, radians and Line direction components) would not.

The division is the scheme's only fragile step. ``ker(J_x)`` is mostly invisible to J_phi --
it contains global rigid motion (phi is a complete rigid invariant), the two-per-Line
parameterisation gauge of ``(pnt, dir, startParam, endParam)``, and every coordinate of an
entity outside ``base_eids`` -- so a draw can land where ``||J_phi n_j||`` is numerically
zero. Such a draw is discarded, not rescaled: below ``OBSERVABLE_GAIN_FLOOR`` the quotient
is roundoff, not a direction.

How much the floor matters is measured rather than assumed. There is no clean gap to hide
in: over the full run the draw gains thin out continuously, and each decade of floor from
1e-10 to 1e-5 discards a few hundred more draws of 1.29 million. What makes the choice
harmless is the consequence, not a gap -- ``analyze_pseudoinverse.floor_sensitivity``
re-derives the reported statistics under each candidate floor, and moving it four orders up
to 1e-5 shifts every reported median and Q90 by under 0.5% relative.

A response for which *every* direction is discarded is not a sampling failure. It means
``J_phi N = 0``, hence ``J_phi R = J_phi R0`` for every admissible R, so theta_full,
theta_rel and S_place cannot depend on the minimum-norm choice at all -- see
``ResponseGeometry.exactly_invariant``. Only theta_leak, which compares ``range(R)`` in raw
state space, stays response-dependent there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

EPS_DEFAULT = 1e-9
RCOND_DEFAULT = 1e-6
OBSERVABLE_GAIN_FLOOR = 1e-9
"""Draw guard on ``||J_phi n||`` for a unit ``n``. This is the absolute floor of
``effective_cutoff`` -- the same number the evaluator already uses to decide that a
singular value is zero -- reused at the level of a single sampled direction."""
ZERO_COLUMN_TOL = 1e-12
CONSISTENCY_TOL = 1e-6

NOT_UNDERCONSTRAINED = "not_underconstrained"
INCONSISTENT_LINEAR_SYSTEM = "inconsistent_linear_system"
NO_NONZERO_OBSERVABLE_RESPONSE = "no_nonzero_observable_response"


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


@dataclass(frozen=True)
class ResponseGeometry:
    """Everything about one annotation's response freedom that no sample changes."""

    R0: torch.Tensor
    J_x: torch.Tensor
    J_p: torch.Tensor
    J_phi: torch.Tensor
    N: torch.Tensor
    observable_column_norms: torch.Tensor
    observable_gain_max: float
    """sigma_max(J_phi N): the largest ``||J_phi n||`` any unit ``n`` in ker(J_x) can reach,
    i.e. the supremum of the quantity each draw is guarded on."""
    observable_free_rank: int
    """rank(J_phi N). Reported for stratification only -- the sampler never branches on it."""
    raw_null_dim: int
    r0_frobenius: float
    consistency_residual: float
    n_zero_observable_columns: int
    gain_floor: float
    """The draw guard, carried here so the response-level verdict below and the per-draw test
    in ``perturbed_response`` can never be judged against two different thresholds."""
    skip_reason: str

    @property
    def exactly_invariant(self) -> bool:
        """J_phi N is numerically zero, so J_phi R is the same for every admissible R.

        Equivalently: no unit direction in ker(J_x) can pass the draw guard, since
        ``observable_gain_max`` is the supremum of the quantity each draw is tested on.
        """
        return self.observable_gain_max < self.gain_floor

    def as_record(self) -> dict:
        return dict(
            raw_null_dim=self.raw_null_dim,
            observable_free_rank=self.observable_free_rank,
            observable_gain_max=self.observable_gain_max,
            exactly_invariant=self.exactly_invariant,
            consistency_residual=self.consistency_residual,
            r0_frobenius=self.r0_frobenius,
            n_zero_observable_columns=self.n_zero_observable_columns,
            n_drivers=int(self.R0.shape[1]),
            skip_reason=self.skip_reason,
        )


def build_response_geometry(
    native, J_phi: torch.Tensor,
    eps: float = EPS_DEFAULT, rcond: float = RCOND_DEFAULT,
    consistency_tol: float = CONSISTENCY_TOL,
    gain_floor: float = OBSERVABLE_GAIN_FLOOR,
) -> ResponseGeometry:
    dtype = native.R.dtype
    J_phi = J_phi.to(dtype=dtype)
    R0, J_x, J_p = native.R, native.J_x, native.J_p

    residual = torch.linalg.norm(J_x @ R0 + J_p) if J_p.numel() else torch.zeros((), dtype=dtype)
    denominator = max(float(torch.linalg.norm(J_p)) if J_p.numel() else 0.0, 1.0)
    consistency_residual = float(residual) / denominator

    N = cutoff_null_space(J_x, eps, rcond)
    gain_max, free_rank = _observable_gain(J_phi @ N, eps, rcond)
    observable_column_norms = torch.linalg.norm(J_phi @ R0, dim=0)

    skip_reason = ""
    if int(N.shape[1]) == 0:
        skip_reason = NOT_UNDERCONSTRAINED
    elif consistency_residual > consistency_tol:
        skip_reason = INCONSISTENT_LINEAR_SYSTEM
    elif int(R0.shape[1]) == 0 or bool((observable_column_norms < ZERO_COLUMN_TOL).all()):
        skip_reason = NO_NONZERO_OBSERVABLE_RESPONSE

    return ResponseGeometry(
        R0=R0,
        J_x=J_x,
        J_p=J_p,
        J_phi=J_phi,
        N=N,
        observable_column_norms=observable_column_norms,
        observable_gain_max=gain_max,
        observable_free_rank=free_rank,
        raw_null_dim=int(N.shape[1]),
        r0_frobenius=float(torch.linalg.norm(R0)) if R0.numel() else 0.0,
        consistency_residual=consistency_residual,
        n_zero_observable_columns=int((observable_column_norms < ZERO_COLUMN_TOL).sum()),
        gain_floor=gain_floor,
        skip_reason=skip_reason,
    )


def _observable_gain(K: torch.Tensor, eps: float, rcond: float) -> tuple[float, int]:
    """(sigma_max, rank) of ``K = J_phi N`` from one SVD; no basis is retained."""
    if K.numel() == 0 or K.shape[1] == 0:
        return 0.0, 0
    S = torch.linalg.svdvals(K)
    return float(S[0]), int((S >= effective_cutoff(S, eps, rcond)).sum())


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
    geometry: ResponseGeometry, eta: float, seed: int,
    *, gain_floor: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """``R0 + Delta`` with ``Delta_j = eta ||M_j|| n_j / ||J_phi n_j||``.

    A column is left untouched when its observable response is zero (there is no size to
    scale to) or when its draw falls below ``gain_floor``; both are counted rather than
    silently rescaled or resampled. No retry loop: resampling would replace the uniform
    distribution on ker(J_x) with a conditioned one, and the measured rejection rate does not
    justify that -- once the exactly-invariant responses are set aside, the full run guarded
    30 columns out of 1.29 million draws.
    """
    gain_floor = geometry.gain_floor if gain_floor is None else gain_floor
    R0 = geometry.R0
    n_columns = int(R0.shape[1])
    if eta == 0.0 or n_columns == 0:
        return R0, _sample_diagnostics(geometry, R0, n_guarded=0, gains=None)

    directions = _unit_columns(geometry.N, n_columns, seed)
    gains = torch.linalg.norm(geometry.J_phi @ directions, dim=0)
    sizeable = geometry.observable_column_norms >= ZERO_COLUMN_TOL
    guarded = sizeable & (gains < gain_floor)
    usable = sizeable & ~guarded

    factors = torch.zeros_like(gains)
    factors[usable] = eta * geometry.observable_column_norms[usable] / gains[usable]
    R_eta = R0 + directions * factors if bool(usable.any()) else R0
    return R_eta, _sample_diagnostics(
        geometry, R_eta, n_guarded=int(guarded.sum()), gains=gains[sizeable]
    )


def _sample_diagnostics(
    geometry: ResponseGeometry, R_eta: torch.Tensor, *, n_guarded: int,
    gains: torch.Tensor | None,
) -> dict:
    """``gains`` holds the draw gains of the columns the guard actually judged, so
    ``min_observable_gain`` is what a different floor would have to cross to change this
    draw, which is what makes the floor's arbitrariness measurable after the fact.
    """
    delta = R_eta - geometry.R0
    denominator = max(
        float(torch.linalg.norm(geometry.J_p)) if geometry.J_p.numel() else 0.0,
        1.0,
    )
    total_residual = geometry.J_x @ R_eta + geometry.J_p
    nullspace_residual = geometry.J_x @ delta
    return dict(
        raw_perturbation_ratio=float(torch.linalg.norm(delta))
        / max(geometry.r0_frobenius, ZERO_COLUMN_TOL),
        constraint_residual=float(torch.linalg.norm(total_residual)) / denominator,
        nullspace_residual=float(torch.linalg.norm(nullspace_residual)) / denominator,
        n_columns_guarded=n_guarded,
        min_observable_gain=float(gains.min()) if gains is not None and gains.numel() else float("nan"),
    )
