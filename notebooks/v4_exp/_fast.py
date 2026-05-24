"""Caching layer for v4_exp scripts.

Mirrors ``notebooks/v3_exp/_fast.py``: ``compute_metrics_v4`` recomputes
``compute_native_response`` / ``pairwise_distance_observable`` /
``constraint_jacobian`` for every ``(pred, gt, variant, probe)`` even though
within one rotation only the probe (or only the GT) actually changes between
many of those calls. This module wires :data:`notebooks.v3_exp._fast.CACHE`
into the v4 metric so all per-rotation work is shared with the v3 path that
``exp_v4_controls`` also runs.

Public surface mirrors v3:

* :func:`compute_metrics_v4_cached` — drop-in replacement for
  ``metrics_v4.compute_metrics_v4``.
* :func:`install` / re-exported ``clear`` — patch the experiment driver and
  reset the cache between rotations.

Numerical equivalence with the original v4 metric is preserved: the cache
stores returned tensors as-is and the per-probe loop performs exactly the
same operations in the same order. The relation Jacobian stack used here is
the raw concat (no per-relation normalisation, no skip-on-no-row) so it
matches ``metrics_v4._stack_relation_jacobians_at`` byte-for-byte.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS = os.path.abspath(os.path.join(HERE, ".."))
V3_DIR = os.path.abspath(os.path.join(NOTEBOOKS, "v3_exp"))
REPO_ROOT = os.path.abspath(os.path.join(NOTEBOOKS, ".."))
for _p in (REPO_ROOT, NOTEBOOKS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_v3_fast():
    """Load ``notebooks/v3_exp/_fast.py`` under a unique module name.

    Importing it as ``_fast`` would collide with this module (both files
    are named ``_fast.py``) and trigger a circular import; using
    ``importlib`` against an absolute path sidesteps ``sys.modules``
    ambiguity. We deliberately *do not* permanently add ``V3_DIR`` to
    ``sys.path`` so callers can still ``import _fast`` and get this v4
    module.
    """
    name = "_v4_fast__v3_backend"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(V3_DIR, "_fast.py"))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not locate v3 _fast at {V3_DIR}/_fast.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_v3_fast = _load_v3_fast()
CACHE = _v3_fast.CACHE
clear = _v3_fast.clear
_install_v3 = _v3_fast.install

from metrics_v4 import (  # noqa: E402
    V4Hyperparams,
    _column_norms,
    _orthonormal_range,
    _row_norms,
    _solve_regularised_ls,
)
from src.metrics.edit_response_distance import (  # noqa: E402
    pairwise_distance_observable,
)


# ---------------------------------------------------------------------------
# Relation Jacobian stack (raw concat — matches metrics_v4 semantics)
# ---------------------------------------------------------------------------

def _stack_relation_jacobians_at_cached(rels, state, context):
    """Cache-backed twin of ``metrics_v4._stack_relation_jacobians_at``.

    Output is byte-for-byte equivalent (same row ordering, same skip-on-empty
    rule). Reuses per-(rel, state) ``Jq`` via :data:`CACHE`.``get_jq`` so the
    same Jacobian computed by the v3 path is not recomputed.
    """
    rows: list[torch.Tensor] = []
    for rel in rels:
        j = CACHE.get_jq(rel, state, context)
        if j is None or j.shape[0] == 0:
            continue
        rows.append(j)
    if not rows:
        return None, 0
    Jr = torch.cat(rows, dim=0)
    return Jr, int(Jr.shape[0])


def _jphi(sketch, base_eids):
    """Cache the ``base_eids``-restricted observable Jacobian.

    Falls back to a direct call when ``base_eids is None`` (the v4 metric's
    default-base path; ``CACHE.get_jphi`` doesn't cache this case).
    """
    if base_eids is None:
        return pairwise_distance_observable(sketch)
    return CACHE.get_jphi(sketch, base_eids)


# ---------------------------------------------------------------------------
# Cached compute_metrics_v4 — mirrors metrics_v4.compute_metrics_v4
# ---------------------------------------------------------------------------

def compute_metrics_v4_cached(
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
    """Drop-in replacement for ``metrics_v4.compute_metrics_v4``.

    Identical return keys and numerical formula; the speedup comes from
    skipping repeated ``compute_native_response`` /
    ``pairwise_distance_observable`` / ``constraint_jacobian`` work that
    one rotation makes across its (variant, gt) combinations.
    """
    if hp is None:
        hp = V4Hyperparams()
    eps = hp.eps

    pred_native = CACHE.get_native(pred_sketch, context, eps=eps)
    gt_native = CACHE.get_native(gt_sketch, context, eps=eps)
    R_P = pred_native.R
    R_G = gt_native.R
    n_p_pred = R_P.shape[1]
    n_p_gt = R_G.shape[1]

    J_phi_P = _jphi(pred_sketch, base_eids)
    J_phi_G = _jphi(gt_sketch, base_eids)

    if J_phi_P.shape[0] != J_phi_G.shape[0] or J_phi_P.shape[0] == 0:
        W_P = torch.zeros(0, n_p_pred, dtype=R_P.dtype)
        W_G = torch.zeros(0, n_p_gt, dtype=R_G.dtype)
    else:
        W_P = J_phi_P @ R_P if n_p_pred > 0 else torch.zeros(
            J_phi_P.shape[0], 0, dtype=R_P.dtype)
        W_G = J_phi_G @ R_G if n_p_gt > 0 else torch.zeros(
            J_phi_G.shape[0], 0, dtype=R_G.dtype)
    U_W_P, dim_P_obs = _orthonormal_range(W_P, eps=eps)
    U_W_G, dim_G_obs = _orthonormal_range(W_G, eps=eps)

    d_norm = float(max(dim_P_obs, dim_G_obs, 1))
    if dim_P_obs > 0 and dim_G_obs > 0:
        Mpa = U_W_G.T @ U_W_P
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
    d_sub_angle_sq = (d_norm - cos2_sum) / d_norm
    d_sub_angle_sq = min(max(d_sub_angle_sq, 0.0), 1.0)
    d_sub_angle = math.sqrt(d_sub_angle_sq)
    rank_loss = abs(dim_P_obs - dim_G_obs) / d_norm

    U_P_prim, dim_P_prim = _orthonormal_range(R_P, eps=eps)
    J_R_pred_state, n_R_rows_pred = _stack_relation_jacobians_at_cached(
        rels, pred_native.state, context)
    if J_R_pred_state is not None and dim_P_prim > 0 and n_R_rows_pred > 0:
        leak_op = J_R_pred_state @ U_P_prim
        leak_prime_sq = float((leak_op ** 2).sum()) / float(
            n_R_rows_pred * dim_P_prim)
    else:
        leak_prime_sq = 0.0
    leak_prime = math.sqrt(max(leak_prime_sq, 0.0))

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

    D_p_pred = _column_norms(R_P, eps=eps)
    if n_p_pred > 0:
        R_tilde = R_P / D_p_pred.unsqueeze(0)
    else:
        R_tilde = R_P

    if (
        J_R_pred_state is not None
        and J_R_pred_state.shape[0] > 0
        and n_p_pred > 0
    ):
        A_R_full = J_R_pred_state @ R_tilde
        D_R = _row_norms(A_R_full, eps=eps)
        A_R_norm = A_R_full / D_R.unsqueeze(1)
    else:
        A_R_norm = None

    sum_w = 0.0
    sum_d_K_sq = 0.0
    sum_gain_K_sq = 0.0
    sum_leak_K_sq = 0.0
    sum_effort_K_sq = 0.0
    sum_gain_I = 0.0
    sum_gain_J = 0.0
    skipped_probes = 0
    M_Aq_rows: list[torch.Tensor] = []
    M_kq_total = 0

    for probe, weight in zip(edit_probes, edit_weights):
        Jq = CACHE.get_jq(probe, pred_native.state, context)
        if Jq is None:
            skipped_probes += 1
            continue
        k_q = Jq.shape[0]

        if n_p_pred == 0:
            sum_d_K_sq += float(weight) * (1.0 + hp.lambda_k * 0.0)
            sum_gain_K_sq += float(weight) * 1.0
            sum_gain_I += float(weight) * 1.0
            sum_gain_J += float(weight) * 1.0
            sum_w += float(weight)
            M_Aq_rows.append(torch.zeros(k_q, 0, dtype=Jq.dtype))
            M_kq_total += k_q
            continue

        A_q = Jq @ R_tilde

        u_star = _solve_regularised_ls(
            A_q, A_R_norm, n_p_pred, mu=hp.mu_k, lambda_=hp.lambda_k)
        eye_kq = torch.eye(k_q, dtype=A_q.dtype, device=A_q.device)
        gain_res = A_q @ u_star - eye_kq
        gain_K_sq = float(torch.linalg.norm(gain_res) ** 2) / max(k_q, 1)
        if A_R_norm is not None and A_R_norm.shape[0] > 0:
            leak_K_sq = float(
                torch.linalg.norm(A_R_norm @ u_star) ** 2) / max(
                    A_R_norm.shape[0] * k_q, 1)
        else:
            leak_K_sq = 0.0
        effort_K_sq = float(torch.linalg.norm(u_star) ** 2) / max(
            n_p_pred * k_q, 1)
        d_K_sq = gain_K_sq + hp.mu_k * leak_K_sq + hp.lambda_k * effort_K_sq

        try:
            u_I = torch.linalg.lstsq(A_q, eye_kq).solution
            achieved = A_q @ u_I
            if float(torch.linalg.norm(achieved - eye_kq)) > 1e-6 * math.sqrt(k_q):
                gain_I_val = 1.0
            else:
                e_q_sq = float(torch.linalg.norm(u_I) ** 2) / max(k_q, 1)
                gain_I_val = e_q_sq / (1.0 + e_q_sq)
        except Exception:
            gain_I_val = 1.0

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

    if n_p_pred == 0:
        k_Q = M_kq_total
        gain_M = 1.0 if k_Q > 0 else 0.0
        leak_M = 0.0
        effort_M = 0.0
        d_M = math.sqrt(gain_M ** 2)
    elif M_Aq_rows:
        A_Q = torch.cat(
            [t for t in M_Aq_rows if t.shape[1] == n_p_pred], dim=0)
        if A_Q.numel() == 0:
            k_Q = 0
            d_M = gain_M = leak_M = effort_M = 0.0
        else:
            k_Q = A_Q.shape[0]
            D_Q = _row_norms(A_Q, eps=eps)
            A_Q_norm = A_Q / D_Q.unsqueeze(1)
            U_M = _solve_regularised_ls(
                A_Q_norm, A_R_norm, n_p_pred, mu=hp.mu_k, lambda_=hp.lambda_k)
            eye_kQ = torch.eye(k_Q, dtype=A_Q.dtype, device=A_Q.device)
            gain_M_sq = float(
                torch.linalg.norm(A_Q_norm @ U_M - eye_kQ) ** 2) / max(k_Q, 1)
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
        k_Q = 0
        d_M = gain_M = leak_M = effort_M = 0.0

    return {
        "d_sub": d_sub,
        "d_sub_angle": d_sub_angle,
        "align_loss": align_loss,
        "rank_loss": rank_loss,
        "dim_P": int(dim_P_obs),
        "dim_G": int(dim_G_obs),
        "dim_P_prim": int(dim_P_prim),
        "d_H": d_H,
        "leak_prime": leak_prime,
        "d_H_angle": d_H_angle,
        "leak_angle": leak_angle,
        "d_K": d_K,
        "gain_K": gain_K,
        "leak_K": leak_K,
        "effort_K": effort_K,
        "d_M": d_M,
        "gain_M": gain_M,
        "leak_M": leak_M,
        "effort_M": effort_M,
        "k_Q_stack": int(k_Q),
        "gain_I": gain_I_mean,
        "gain_J": gain_J_mean,
        "n_R_rows": int(n_R_rows_pred),
        "n_p_pred": int(n_p_pred),
        "n_p_gt": int(n_p_gt),
        "n_relations": int(len(rels)),
        "skipped_probes": int(skipped_probes),
    }


# ---------------------------------------------------------------------------
# install() — patch a module's slow entry points to use the cache
# ---------------------------------------------------------------------------

def install(module, *, wrap_evaluate_rotation: bool = True) -> None:
    """Patch v3+v4 metric entry points on ``module`` and wrap its
    ``evaluate_rotation`` so the cache resets between rotations.

    Delegates to ``v3_exp._fast.install`` for ``compute_metrics`` /
    ``compute_pair_metrics`` / ``response_distance`` / ``edit_attainability``
    plus the ``evaluate_rotation`` wrap, then additionally patches
    ``compute_metrics_v4`` if the module imports it.
    """
    _install_v3(module, wrap_evaluate_rotation=wrap_evaluate_rotation)
    if hasattr(module, "compute_metrics_v4"):
        module.compute_metrics_v4 = compute_metrics_v4_cached


__all__ = [
    "CACHE",
    "clear",
    "compute_metrics_v4_cached",
    "install",
]
