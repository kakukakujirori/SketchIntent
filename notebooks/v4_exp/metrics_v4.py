"""metrics_v4.py — alternative response-feasibility metrics for v4 evaluation.

Implements the top-priority candidates from
``notebooks/summary_histcad_response_metric_v4.md``:

- Case D  : probe-free subspace distance between range(R_P) and range(R_G)
- Case H  : Case D plus operator-leak  ``||J_{R_G} U_P||_F``
- Case I  : per-probe knob-effort  ``e_q = min_u ||u||  s.t. A_q u = I``
- Case J  : per-probe budget reach  ``reach_q = ||A_q||_2``
- Case K  : per-probe relation-preserving controllability
              ``min_u ||A_q u - I||^2 + mu ||A_R u||^2 + lambda ||u||^2``
- Case M  : Case K stacked over the whole probe set Q_G

The helpers reuse ``compute_native_response`` / ``constraint_jacobian`` from
``src.metrics.edit_response_distance`` and ``stack_relation_jacobians`` from
``exp_constrained_response_metric``.  This module computes only v4 metrics;
v3 ``compute_metrics`` should be called in parallel by the driver to get
``d_feas`` alongside for direct comparison.

Conventions
-----------
``D_p``  : column-norm of ``R_P`` with an ``eps`` floor.  After right-
            multiplying by ``D_p^{-1}`` each column of the reduced response
            ``R_tilde`` has unit norm.  This neutralises the angle-vs-distance
            scale mismatch of native dimensional parameters without requiring
            constraint-type metadata.

``D_R``  : row-norm of ``A_R = J_{R_G} R_tilde`` with an ``eps`` floor.
            Each relation residual row contributes unit Frobenius energy,
            matching the per-relation ``J_R`` normalisation used by
            ``d_feas_tilde`` in v3.

``D_q``  : kept at 1 — probe values are already commensurate after
            ``R_tilde``'s column normalisation and the gain residual
            ``A_q u - I`` is itself unitless.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import torch

# Sys-path setup so this module can be run from anywhere under the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_NOTEBOOKS = os.path.abspath(os.path.join(_HERE, ".."))
if _NOTEBOOKS not in sys.path:
    sys.path.insert(0, _NOTEBOOKS)

from src.metrics.edit_response_distance import (  # noqa: E402
    NativeResponse,
    compute_native_response,
    constraint_jacobian,
    pairwise_distance_observable,
)


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------

def _orthonormal_range(R: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    """Return ``(U, dim)`` where ``U`` is an orthonormal basis of ``range(R)``.

    ``R`` has shape ``(n_x, n_p)``.  Uses absolute-eps SVD truncation, the
    same convention as ``_abs_pinv``/``compute_native_response``.
    """
    if R.numel() == 0 or R.shape[1] == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    U, S, _Vh = torch.linalg.svd(R, full_matrices=False)
    keep = (S >= eps).sum().item()
    if keep == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    return U[:, :keep].contiguous(), int(keep)


def _column_norms(R: torch.Tensor, eps: float) -> torch.Tensor:
    """Column norms of ``R`` with an ``eps`` floor; shape ``(n_p,)``."""
    if R.shape[1] == 0:
        return torch.zeros(0, dtype=R.dtype, device=R.device)
    return torch.linalg.norm(R, dim=0).clamp_min(eps)


def _row_norms(A: torch.Tensor, eps: float) -> torch.Tensor:
    if A.shape[0] == 0:
        return torch.zeros(0, dtype=A.dtype, device=A.device)
    return torch.linalg.norm(A, dim=1).clamp_min(eps)


def _stack_relation_jacobians_at(
    rels,
    state,
    context,
) -> tuple[torch.Tensor | None, int]:
    """Plain stacked ``J_R`` at ``state`` (no per-relation normalisation).

    Returns ``(J_R, n_rows)``; ``J_R`` may be ``None`` if every relation
    handler failed.  This is the raw operator we project with ``U_P`` for
    Case H's ``leak'`` and use as the leak block for Cases K / M (which
    apply their own row-normalisation downstream).
    """
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


def _solve_regularised_ls(
    A_q: torch.Tensor,
    A_R: torch.Tensor | None,
    n_p: int,
    *,
    mu: float,
    lambda_: float,
) -> torch.Tensor:
    """Solve ``argmin_u ||A_q u - I||^2 + mu ||A_R u||^2 + lambda ||u||^2``.

    Returns ``u*`` of shape ``(n_p, k_q)`` where ``k_q = A_q.shape[0]``.
    Built via a stacked least-squares matrix; the ``lambda * I`` row block
    guarantees the system is solvable even when ``A_q`` is rank-deficient.
    """
    dtype = A_q.dtype
    device = A_q.device
    k_q = A_q.shape[0]
    eye_kq = torch.eye(k_q, dtype=dtype, device=device)

    stack_rows = [A_q]
    stack_rhs = [eye_kq]
    if A_R is not None and A_R.shape[0] > 0 and mu > 0:
        stack_rows.append(math.sqrt(mu) * A_R)
        stack_rhs.append(torch.zeros(A_R.shape[0], k_q, dtype=dtype, device=device))
    if lambda_ > 0 and n_p > 0:
        stack_rows.append(math.sqrt(lambda_) * torch.eye(n_p, dtype=dtype, device=device))
        stack_rhs.append(torch.zeros(n_p, k_q, dtype=dtype, device=device))
    A = torch.cat(stack_rows, dim=0)
    b = torch.cat(stack_rhs, dim=0)
    return torch.linalg.lstsq(A, b).solution


def _compute_case_p(
    *,
    R_P: torch.Tensor,
    R_G: torch.Tensor,
    J_phi_P: torch.Tensor,
    J_phi_G: torch.Tensor,
    pred_state,
    gt_state,
    edit_probes,
    context,
    tau_p: float,
) -> dict | None:
    """Case P : SVD-based response comparison in observable space.

    Builds ``T_C = J_phi @ R_C @ pinv(J_Q @ R_C)`` for Pred and GT, then
    compares their singular spectra (``d_SVD_sigma_rel``), truncated
    principal-angle subspaces (``d_SVD_sub_trunc``), and leading singular
    directions (``d_SVD_top1``).  Only probes for which BOTH Pred and GT
    can evaluate a Jacobian are retained, so the column counts of T_P / T_G
    match by construction.  Returns ``None`` when there is no common
    observable space or no shared probe.

    ``tau_p`` is the relative threshold for truncated subspace comparison;
    components with sigma < tau_p * max(sig_P[0], sig_G[0]) are excluded
    when computing ``d_SVD_sub_trunc``.
    """
    if J_phi_P.shape[0] != J_phi_G.shape[0] or J_phi_P.shape[0] == 0:
        return None
    if R_P.shape[1] == 0 or R_G.shape[1] == 0:
        return None

    J_Q_P_rows: list[torch.Tensor] = []
    J_Q_G_rows: list[torch.Tensor] = []
    for probe in edit_probes:
        Jq_P = constraint_jacobian(probe, pred_state, context)
        Jq_G = constraint_jacobian(probe, gt_state, context)
        if Jq_P is None or Jq_G is None:
            continue
        # First row only — matches ``test_dH.compute_d_svd`` scalar-probe
        # convention; multi-row dimensional residuals are unusual in our set.
        J_Q_P_rows.append(Jq_P[:1])
        J_Q_G_rows.append(Jq_G[:1])
    if not J_Q_P_rows:
        return None

    J_Q_P = torch.cat(J_Q_P_rows, dim=0)
    J_Q_G = torch.cat(J_Q_G_rows, dim=0)

    M_P = J_Q_P @ R_P
    M_G = J_Q_G @ R_G

    S_P = R_P @ torch.linalg.pinv(M_P)
    S_G = R_G @ torch.linalg.pinv(M_G)

    T_P = J_phi_P @ S_P
    T_G = J_phi_G @ S_G

    m_phi, k_Q = T_P.shape

    U_P, sig_P, _Vh_P = torch.linalg.svd(T_P, full_matrices=False)
    U_G, sig_G, _Vh_G = torch.linalg.svd(T_G, full_matrices=False)

    sigma_P_norm = float(torch.linalg.norm(sig_P))
    sigma_G_norm = float(torch.linalg.norm(sig_G))

    d_sigma = float(torch.linalg.norm(sig_P - sig_G))
    d_sigma_rel = d_sigma / (sigma_G_norm + 1e-12)

    # Truncated subspace distance: keep components with sigma >= tau_p * sig_max
    sig_max = max(float(sig_P[0]), float(sig_G[0]), 1e-30)
    thresh = tau_p * sig_max
    k_eff_P = int((sig_P >= thresh).sum().item())
    k_eff_G = int((sig_G >= thresh).sum().item())
    k_norm = max(k_eff_P, k_eff_G, 1)
    if k_eff_P > 0 and k_eff_G > 0:
        Ut_P = U_P[:, :k_eff_P]
        Ut_G = U_G[:, :k_eff_G]
        M_align_t = Ut_G.T @ Ut_P
        svals_t = torch.linalg.svdvals(M_align_t).clamp(0.0, 1.0)
        cos2_t = float((svals_t ** 2).sum())
        d_sub_trunc_sq = (k_eff_P + k_eff_G - 2.0 * cos2_t) / k_norm
        d_sub_trunc = math.sqrt(max(d_sub_trunc_sq, 0.0))
    elif k_eff_P == 0 and k_eff_G == 0:
        d_sub_trunc = 0.0
    else:
        d_sub_trunc = math.sqrt((k_eff_P + k_eff_G) / k_norm)

    # Top-1 leading direction (sign-invariant).
    cos1 = abs(float(U_P[:, 0] @ U_G[:, 0]))
    d_top1 = math.sqrt(max(2.0 - 2.0 * cos1, 0.0))

    return {
        "d_SVD_sigma_rel": d_sigma_rel,
        "d_SVD_sub_trunc": d_sub_trunc,
        "d_SVD_top1": d_top1,
        "sigma_P_norm": sigma_P_norm,
        "sigma_G_norm": sigma_G_norm,
        "k_eff_P": k_eff_P,
        "k_eff_G": k_eff_G,
        "k_Q_used": int(k_Q),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class V4Hyperparams:
    eps: float = 1e-9
    mu_k: float = 1.0       # relation-preservation weight in Case K / M
    lambda_k: float = 1e-3  # effort regulariser in Case K / M
    tau_j: float = 1.0      # reach scale for Case J
    tau_p: float = 0.01     # Case P SVD truncation threshold (relative)


METRIC_NAMES_V4: tuple[str, ...] = (
    "d_sub",
    "d_sub_angle",
    "d_H",
    "d_H_angle",
    "leak_angle",
    "d_K",
    "d_M",
    "gain_I",
    "gain_J",
    "d_SVD_sigma_rel",
    "d_SVD_sub_trunc",
    "d_SVD_top1",
)


def compute_metrics_v4(
    *,
    pred_sketch,
    gt_sketch,
    edit_probes,
    edit_weights,
    rels,
    context,
    base_eids=None,
    hp: V4Hyperparams | None = None,
) -> dict:
    """Compute v4 candidate metrics for a single ``(Pred, GT)`` pair.

    Parameters mirror ``exp_constrained_response_metric.compute_metrics``.
    The subspace metrics (Case D / H) use the observable pair-distance space
    ``J_phi C R_C`` so that Pred and GT — which may have different primitive
    counts when the model drops entities — can be compared in a common space.
    Case H's operator-leak term lives in Pred's primitive space, where it is
    well-defined without cross-sketch alignment.
    """

    if hp is None:
        hp = V4Hyperparams()
    eps = hp.eps

    pred_native: NativeResponse = compute_native_response(
        pred_sketch, context, eps=eps)
    gt_native: NativeResponse = compute_native_response(
        gt_sketch, context, eps=eps)

    R_P = pred_native.R
    R_G = gt_native.R
    n_p_pred = R_P.shape[1]
    n_p_gt = R_G.shape[1]

    # ---------- Observable Jacobians (shared n_phi space via base_eids) ----------
    if base_eids is None:
        # Fall back to whatever entities each sketch has; this is only safe when
        # Pred and GT share a primitive layout (e.g. heldout / source_self).
        J_phi_P = pairwise_distance_observable(pred_sketch)
        J_phi_G = pairwise_distance_observable(gt_sketch)
    else:
        J_phi_P = pairwise_distance_observable(pred_sketch, base_eids=base_eids)
        J_phi_G = pairwise_distance_observable(gt_sketch, base_eids=base_eids)

    # ---------- Case D / H subspace component (observable space) ----------
    if J_phi_P.shape[0] != J_phi_G.shape[0] or J_phi_P.shape[0] == 0:
        # Different pair counts (e.g. base_eids resolves differently): no
        # common observable space.  Mark metrics as missing.
        W_P = torch.zeros(0, n_p_pred, dtype=R_P.dtype)
        W_G = torch.zeros(0, n_p_gt, dtype=R_G.dtype)
    else:
        W_P = J_phi_P @ R_P if n_p_pred > 0 else torch.zeros(J_phi_P.shape[0], 0, dtype=R_P.dtype)
        W_G = J_phi_G @ R_G if n_p_gt > 0 else torch.zeros(J_phi_G.shape[0], 0, dtype=R_G.dtype)
    U_W_P, dim_P_obs = _orthonormal_range(W_P, eps=eps)
    U_W_G, dim_G_obs = _orthonormal_range(W_G, eps=eps)

    d_norm = float(max(dim_P_obs, dim_G_obs, 1))
    if dim_P_obs > 0 and dim_G_obs > 0:
        Mpa = U_W_G.T @ U_W_P                    # (dim_G_obs, dim_P_obs)
        sigmas = torch.linalg.svdvals(Mpa)
        sigmas = sigmas.clamp(min=0.0, max=1.0)
        cos2_sum = float((sigmas ** 2).sum())
        min_dim = min(dim_P_obs, dim_G_obs)
        sin2 = float((1.0 - sigmas[:min_dim] ** 2).sum())
        align_loss = sin2 / max(min_dim, 1)
    else:
        cos2_sum = 0.0
        align_loss = 0.0
    d_sub_sq = (dim_P_obs + dim_G_obs - 2.0 * cos2_sum) / d_norm
    d_sub_sq = max(d_sub_sq, 0.0)
    d_sub = math.sqrt(d_sub_sq)
    # Padded principal-angle view: compare over max(dim_P, dim_G) angles,
    # treating missing dimensions as cos(theta)=0.  This preserves the
    # existing nested-subspace dropout scale while making equal-dimension
    # orthogonal subspaces have ceiling 1 instead of sqrt(2).
    d_sub_angle_sq = (d_norm - cos2_sum) / d_norm
    d_sub_angle_sq = min(max(d_sub_angle_sq, 0.0), 1.0)
    d_sub_angle = math.sqrt(d_sub_angle_sq)
    rank_loss = abs(dim_P_obs - dim_G_obs) / d_norm

    # Operator leak  ||J_{R_G} U_P_prim||_F^2  in Pred's primitive space.
    # U_P_prim is the orthonormal basis of range(R_P) in R^{n_x_P}.  Both
    # J_R_pred_state and U_P_prim are evaluated at Pred's state, so no
    # cross-sketch alignment is needed for the leak operator.
    U_P_prim, dim_P_prim = _orthonormal_range(R_P, eps=eps)
    J_R_pred_state, n_R_rows_pred = _stack_relation_jacobians_at(
        rels, pred_native.state, context)
    if J_R_pred_state is not None and dim_P_prim > 0 and n_R_rows_pred > 0:
        leak_op = J_R_pred_state @ U_P_prim       # (n_R, dim_P_prim)
        leak_prime_sq = float((leak_op ** 2).sum()) / float(
            n_R_rows_pred * dim_P_prim)
    else:
        leak_prime_sq = 0.0
    leak_prime = math.sqrt(max(leak_prime_sq, 0.0))

    # Scale-free angular leak.  N_G spans the forbidden normal directions of
    # GT relations in primitive space; leak_angle is the RMS cosine between
    # Pred's allowed motion directions and that forbidden subspace.
    if J_R_pred_state is not None and dim_P_prim > 0 and n_R_rows_pred > 0:
        N_G_prim, dim_N_prim = _orthonormal_range(J_R_pred_state.T, eps=eps)
        if dim_N_prim > 0:
            forbidden_overlap = N_G_prim.T @ U_P_prim
            leak_angle_sq = float((forbidden_overlap ** 2).sum()) / float(dim_P_prim)
        else:
            leak_angle_sq = 0.0
    else:
        leak_angle_sq = 0.0
    leak_angle_sq = min(max(leak_angle_sq, 0.0), 1.0)
    leak_angle = math.sqrt(leak_angle_sq)

    d_H_sq = d_sub_sq + leak_prime_sq
    d_H = math.sqrt(max(d_H_sq, 0.0))
    d_H_angle_sq = d_sub_angle_sq + leak_angle_sq
    d_H_angle = math.sqrt(max(d_H_angle_sq, 0.0))

    # ---------- Case P (d_SVD family) ----------
    case_p = _compute_case_p(
        R_P=R_P, R_G=R_G,
        J_phi_P=J_phi_P, J_phi_G=J_phi_G,
        pred_state=pred_native.state,
        gt_state=gt_native.state,
        edit_probes=edit_probes,
        context=context,
        tau_p=hp.tau_p,
    ) or {}

    # ---------- Case K / M / I / J  (q-retaining) ----------
    # Column-normalise R_P so each "unit knob" produces unit-norm primitive
    # response.  D_p_pred has shape (n_p_pred,).
    D_p_pred = _column_norms(R_P, eps=eps)
    if n_p_pred > 0:
        R_tilde = R_P / D_p_pred.unsqueeze(0)   # (n_x, n_p)
    else:
        R_tilde = R_P

    # Relation block for K / M: A_R_full = J_{R_G} R_tilde, then row-normalise.
    if J_R_pred_state is not None and J_R_pred_state.shape[0] > 0 and n_p_pred > 0:
        A_R_full = J_R_pred_state @ R_tilde     # (n_R, n_p)
        D_R = _row_norms(A_R_full, eps=eps)
        A_R_norm = A_R_full / D_R.unsqueeze(1)  # row-normalised
    else:
        A_R_norm = None

    sum_w = 0.0
    sum_d_K_sq = 0.0
    sum_gain_K_sq = 0.0
    sum_leak_K_sq = 0.0
    sum_effort_K_sq = 0.0
    sum_gain_I = 0.0
    sum_gain_J = 0.0
    sum_attain_K_sq = 0.0  # gain residual without relation block (sanity)
    skipped_probes = 0

    # Stack accumulators for Case M
    M_Aq_rows: list[torch.Tensor] = []
    M_kq_total = 0

    for probe, weight in zip(edit_probes, edit_weights):
        Jq = constraint_jacobian(probe, pred_native.state, context)
        if Jq is None:
            skipped_probes += 1
            continue
        k_q = Jq.shape[0]

        if n_p_pred == 0:
            # Pred has no dimensional knobs at all → no controllability.
            # Gain residual ||A_q u - I|| collapses to ||I|| = sqrt(k_q),
            # mean-square per row = 1, so gain_K = 1.
            sum_d_K_sq += float(weight) * (1.0 + hp.lambda_k * 0.0)
            sum_gain_K_sq += float(weight) * 1.0
            sum_gain_I += float(weight) * 1.0
            sum_gain_J += float(weight) * 1.0
            sum_w += float(weight)
            M_Aq_rows.append(torch.zeros(k_q, 0, dtype=Jq.dtype))
            M_kq_total += k_q
            continue

        A_q = Jq @ R_tilde                       # (k_q, n_p)

        u_star = _solve_regularised_ls(
            A_q, A_R_norm, n_p_pred, mu=hp.mu_k, lambda_=hp.lambda_k)
        eye_kq = torch.eye(k_q, dtype=A_q.dtype, device=A_q.device)
        gain_res = A_q @ u_star - eye_kq
        gain_K_sq = float(torch.linalg.norm(gain_res) ** 2) / max(k_q, 1)
        if A_R_norm is not None and A_R_norm.shape[0] > 0:
            leak_K_sq = float(torch.linalg.norm(A_R_norm @ u_star) ** 2) / max(
                A_R_norm.shape[0] * k_q, 1)
        else:
            leak_K_sq = 0.0
        effort_K_sq = float(torch.linalg.norm(u_star) ** 2) / max(
            n_p_pred * k_q, 1)
        d_K_sq = gain_K_sq + hp.mu_k * leak_K_sq + hp.lambda_k * effort_K_sq

        # Case I: min ||u||_2  s.t. A_q u = I  (hard, min-norm LS)
        try:
            u_I = torch.linalg.lstsq(A_q, eye_kq).solution
            achieved = A_q @ u_I
            if float(torch.linalg.norm(achieved - eye_kq)) > 1e-6 * math.sqrt(k_q):
                # Hard constraint infeasible → effort is effectively infinite.
                gain_I_val = 1.0
            else:
                e_q_sq = float(torch.linalg.norm(u_I) ** 2) / max(k_q, 1)
                gain_I_val = e_q_sq / (1.0 + e_q_sq)
        except Exception:
            gain_I_val = 1.0

        # Case J: reach_q = ||A_q||_2 (largest sing val); gain_J = 1/(1+reach^2/tau^2)
        if A_q.numel() > 0:
            sigma_max = float(torch.linalg.svdvals(A_q)[0])
        else:
            sigma_max = 0.0
        gain_J_val = 1.0 / (1.0 + (sigma_max ** 2) / (hp.tau_j ** 2))

        sum_w += float(weight)
        sum_d_K_sq += float(weight) * d_K_sq
        sum_gain_K_sq += float(weight) * gain_K_sq
        sum_leak_K_sq += float(weight) * leak_K_sq
        sum_effort_K_sq += float(weight) * effort_K_sq
        sum_gain_I += float(weight) * gain_I_val
        sum_gain_J += float(weight) * gain_J_val
        sum_attain_K_sq += float(weight) * gain_K_sq

        M_Aq_rows.append(A_q)
        M_kq_total += k_q

    if sum_w > 0:
        d_K = math.sqrt(sum_d_K_sq / sum_w)
        gain_K = math.sqrt(sum_gain_K_sq / sum_w)
        leak_K = math.sqrt(sum_leak_K_sq / sum_w)
        effort_K = math.sqrt(sum_effort_K_sq / sum_w)
        gain_I_mean = sum_gain_I / sum_w
        gain_J_mean = sum_gain_J / sum_w
    else:
        d_K = gain_K = leak_K = effort_K = 0.0
        gain_I_mean = gain_J_mean = 0.0

    # ---------- Case M : stacked Q_G ----------
    if n_p_pred == 0:
        # Pred has no dimensional knobs at all — the stacked system M U = b
        # has no degrees of freedom, so ||A_Q U - I|| = ||I|| and the gain
        # collapses to its ceiling.  Matches the definitional ``d_feas=1`` at
        # 100% dimensional dropout.
        k_Q = M_kq_total
        gain_M = 1.0 if k_Q > 0 else 0.0
        leak_M = 0.0
        effort_M = 0.0
        d_M = math.sqrt(gain_M ** 2)  # = gain_M
    elif M_Aq_rows:
        A_Q = torch.cat([t for t in M_Aq_rows if t.shape[1] == n_p_pred], dim=0)
        if A_Q.numel() == 0:
            k_Q = 0
            d_M = gain_M = leak_M = effort_M = 0.0
        else:
            k_Q = A_Q.shape[0]
            # Row-normalise A_Q so individual probes contribute unit-energy
            # rows; otherwise probes with large native gains dominate.
            D_Q = _row_norms(A_Q, eps=eps)
            A_Q_norm = A_Q / D_Q.unsqueeze(1)
            U_M = _solve_regularised_ls(
                A_Q_norm, A_R_norm, n_p_pred, mu=hp.mu_k, lambda_=hp.lambda_k)
            eye_kQ = torch.eye(k_Q, dtype=A_Q.dtype, device=A_Q.device)
            gain_M_sq = float(torch.linalg.norm(A_Q_norm @ U_M - eye_kQ) ** 2) / max(k_Q, 1)
            if A_R_norm is not None and A_R_norm.shape[0] > 0:
                leak_M_sq = float(
                    torch.linalg.norm(A_R_norm @ U_M) ** 2) / max(
                        A_R_norm.shape[0] * k_Q, 1)
            else:
                leak_M_sq = 0.0
            effort_M_sq = float(
                torch.linalg.norm(U_M) ** 2) / max(n_p_pred * k_Q, 1)
            d_M_sq = gain_M_sq + hp.mu_k * leak_M_sq + hp.lambda_k * effort_M_sq
            d_M = math.sqrt(max(d_M_sq, 0.0))
            gain_M = math.sqrt(max(gain_M_sq, 0.0))
            leak_M = math.sqrt(max(leak_M_sq, 0.0))
            effort_M = math.sqrt(max(effort_M_sq, 0.0))
    else:
        # No probes evaluated at all (every probe Jacobian failed).
        k_Q = 0
        d_M = gain_M = leak_M = effort_M = 0.0

    return {
        # Case D (observable subspace)
        "d_sub": d_sub,
        "d_sub_angle": d_sub_angle,
        "align_loss": align_loss,
        "rank_loss": rank_loss,
        "dim_P": int(dim_P_obs),
        "dim_G": int(dim_G_obs),
        "dim_P_prim": int(dim_P_prim),
        # Case H
        "d_H": d_H,
        "leak_prime": leak_prime,
        "d_H_angle": d_H_angle,
        "leak_angle": leak_angle,
        # Case K (per-probe mean)
        "d_K": d_K,
        "gain_K": gain_K,
        "leak_K": leak_K,
        "effort_K": effort_K,
        # Case M (stacked Q_G)
        "d_M": d_M,
        "gain_M": gain_M,
        "leak_M": leak_M,
        "effort_M": effort_M,
        "k_Q_stack": int(k_Q),
        # Case I / J
        "gain_I": gain_I_mean,
        "gain_J": gain_J_mean,
        # Case P (d_SVD family)
        "d_SVD_sigma_rel": case_p.get("d_SVD_sigma_rel"),
        "d_SVD_sub_trunc": case_p.get("d_SVD_sub_trunc"),
        "d_SVD_top1": case_p.get("d_SVD_top1"),
        "sigma_P_norm": case_p.get("sigma_P_norm"),
        "sigma_G_norm": case_p.get("sigma_G_norm"),
        "k_eff_P": case_p.get("k_eff_P"),
        "k_eff_G": case_p.get("k_eff_G"),
        "k_Q_used": case_p.get("k_Q_used"),
        # Diagnostics
        "n_R_rows": int(n_R_rows_pred),
        "n_p_pred": int(n_p_pred),
        "n_p_gt": int(n_p_gt),
        "n_relations": int(len(rels)),
        "skipped_probes": int(skipped_probes),
    }
