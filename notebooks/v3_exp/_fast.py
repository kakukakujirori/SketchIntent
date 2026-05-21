"""Shared caching layer for the v3 experiment scripts.

The original ``notebooks/exp_constrained_response_metric.py`` recomputes
``compute_native_response`` / ``pairwise_distance_observable`` /
``stack_relation_jacobians`` / ``constraint_jacobian`` for every
``(pred, gt, validated, drop_rate)`` combination, even though within one
rotation only the probe SUBSET changes between many of those calls.

This module exposes:

* :class:`RotationCache` — keyed on ``id(sketch)`` and the small extras
  (eps, base_eids, rels identity, normalization). Memoises the expensive
  Jacobian / SVD machinery for the lifetime of one rotation.
* :func:`compute_metrics_cached` — drop-in replacement for
  ``exp_constrained_response_metric.compute_metrics`` that consults the
  cache and additionally reuses the probe Jacobian ``Jq`` across the
  three internal call-sites it was being recomputed at.
* :func:`compute_pair_metrics_cached` — drop-in replacement for
  ``exp_smatch_metric.compute_pair_metrics``.
* :func:`response_distance_cached` — drop-in replacement for
  ``exp_source_conditioned_dropout.response_distance``.
* :func:`edit_attainability_cached` — drop-in replacement for
  ``exp_source_conditioned_dropout.edit_attainability``.
* :func:`install` / :func:`clear` — small helpers that wrappers use to
  install the cache into the original modules and reset it between
  rotations.

The cache deliberately keys on ``id(sketch)`` (object identity). Sketches
that live inside a ``FileBundle`` are stable for the duration of a
rotation; ``dropout_variant`` returns a fresh sketch object that is
threaded through one ``add_pred_rows`` invocation, so the same ``id``
remains valid for all ``(validated, gt_name)`` reuses within that block.
The cache is cleared between rotations to bound memory.

Numerical equivalence with the originals is preserved: the cache stores
returned tensors as-is and the per-probe loops perform exactly the same
operations in the same order as before.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
V3_DIR = os.path.abspath(os.path.dirname(__file__))
if V3_DIR not in sys.path:
    sys.path.insert(0, V3_DIR)

from src.metrics.edit_response_distance import (
    _abs_pinv,
    compute_edit_response_distance,
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
    pairwise_distance_observable,
)


# ---------------------------------------------------------------------------
# Per-rotation cache
# ---------------------------------------------------------------------------

class RotationCache:
    """Memoises Jacobians/natives keyed by sketch identity.

    The cache instance is owned by the wrapper scripts via
    :data:`CACHE`. Wrappers clear it at the start of each rotation
    (``_fast.clear()``) so memory does not grow without bound.
    """

    def __init__(self) -> None:
        self.native: dict[tuple, Any] = {}
        self.jphi: dict[tuple, torch.Tensor | None] = {}
        self.jr: dict[tuple, tuple] = {}
        self.jq: dict[tuple, torch.Tensor | None] = {}

    # --- core lookups --------------------------------------------------

    def get_native(self, sketch, context, eps):
        key = (id(sketch), float(eps))
        entry = self.native.get(key)
        if entry is not None and entry[0] is sketch:
            return entry[1]
        else:
            nat = compute_native_response(sketch, context, eps=eps)
            self.native[key] = (sketch, nat)
        return nat

    def get_jphi(self, sketch, base_eids):
        if base_eids is None:
            return None
        be = tuple(base_eids)
        key = (id(sketch), be)
        entry = self.jphi.get(key, _SENTINEL)
        if entry is not _SENTINEL and entry[0] is sketch:
            return entry[1]
        else:
            j = pairwise_distance_observable(sketch, base_eids=be)
            self.jphi[key] = (sketch, j)
        return j

    def get_jq(self, probe, state, context):
        key = (id(probe), id(state))
        entry = self.jq.get(key, _SENTINEL)
        if entry is not _SENTINEL and entry[0] is probe and entry[1] is state:
            return entry[2]
        else:
            j = constraint_jacobian(probe, state, context)
            self.jq[key] = (probe, state, j)
        return j

    def get_jr(self, rels, state, context, *, eps, normalization="raw",
               scale_state=None):
        key = (
            id(rels), id(state), normalization,
            id(scale_state) if scale_state is not None else None,
            float(eps),
        )
        entry = self.jr.get(key)
        if (
            entry is not None
            and entry[0] is rels
            and entry[1] is state
            and entry[2] is scale_state
        ):
            return entry[3]
        else:
            val = _stack_relation_jacobians(
                self, rels, state, context, eps=eps,
                normalization=normalization, scale_state=scale_state,
            )
            self.jr[key] = (rels, state, scale_state, val)
        return val


_SENTINEL = object()
CACHE = RotationCache()


def clear() -> None:
    """Reset the per-rotation cache. Wrappers call this between rotations."""
    CACHE.native.clear()
    CACHE.jphi.clear()
    CACHE.jr.clear()
    CACHE.jq.clear()


# ---------------------------------------------------------------------------
# stack_relation_jacobians (reuses cached per-(rel,state) Jacobians).
# ---------------------------------------------------------------------------

def _stack_relation_jacobians(cache, rels, state, context, *, eps,
                              normalization="raw", scale_state=None):
    """Cached/optimised twin of
    ``exp_constrained_response_metric.stack_relation_jacobians``.

    Output is byte-for-byte equivalent (same row ordering, same
    accept/skip rule, same per_relation scaling) and we additionally
    share the per-(rel, state) Jacobian with the probe-loop cache.
    """
    rows = []
    unit_count = 0
    skipped = 0
    for rel in rels:
        j = cache.get_jq(rel, state, context)
        if j is None or j.shape[0] == 0:
            skipped += 1
            continue
        if normalization == "per_relation":
            scale_state_ = scale_state if scale_state is not None else state
            scale_j = cache.get_jq(rel, scale_state_, context)
            if scale_j is None or scale_j.shape[0] == 0:
                skipped += 1
                continue
            scale = float(torch.linalg.norm(scale_j))
            if scale <= eps:
                skipped += 1
                continue
            j = j / scale
            unit_count += 1
        elif normalization == "raw":
            unit_count += j.shape[0]
        else:
            raise ValueError(f"unknown relation normalization: {normalization}")
        rows.append(j)
    if not rows:
        return None, 0, skipped
    return torch.cat(rows, dim=0), unit_count, skipped


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (Jq-aware variants of the originals)
# ---------------------------------------------------------------------------

def _edit_response_from_jq(native, Jq, eps):
    """Same math as ``edit_response_for_probe`` but Jq is already known."""
    if Jq is None:
        return None
    n_x = native.R.shape[0]
    k_q = Jq.shape[0]
    if native.R.shape[1] == 0:
        return torch.zeros(n_x, k_q, dtype=Jq.dtype, device=Jq.device)
    A = Jq @ native.R
    A_pinv = _abs_pinv(A, eps=eps)
    return native.R @ A_pinv


def _constrained_edit_response_from_jq(native, Jq, J_R_external, eps):
    """Same math as
    ``exp_constrained_response_metric.constrained_edit_response_for_probe``
    but Jq is already known."""
    if Jq is None:
        return None
    k_q = Jq.shape[0]
    n_x = native.R.shape[0]
    if native.R.shape[1] == 0:
        return torch.zeros(n_x, k_q, dtype=Jq.dtype, device=Jq.device)

    A_top = Jq @ native.R
    if J_R_external is not None and J_R_external.shape[0] > 0:
        A_bot = J_R_external @ native.R
        A = torch.cat([A_top, A_bot], dim=0)
        b = torch.zeros(A.shape[0], k_q, dtype=Jq.dtype, device=Jq.device)
        b[:k_q, :] = torch.eye(k_q, dtype=Jq.dtype, device=Jq.device)
    else:
        A = A_top
        b = torch.eye(k_q, dtype=Jq.dtype, device=Jq.device)
    A_pinv = _abs_pinv(A, eps=eps)
    dp_star = A_pinv @ b
    return native.R @ dp_star


def _mean_square_rows(mat: torch.Tensor, row_units: int) -> float:
    if row_units <= 0 or mat.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(mat) ** 2) / float(row_units)


# ---------------------------------------------------------------------------
# Cached compute_metrics — mirrors exp_constrained_response_metric.compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics_cached(
    *,
    pred_sketch,
    gt_sketch,
    edit_probes,
    edit_weights,
    rels,
    context,
    base_eids,
    eps: float,
):
    """Drop-in replacement for
    ``exp_constrained_response_metric.compute_metrics`` that consults
    :data:`CACHE`. Behaviour is identical (same returned keys, same
    numerical formula); the speedup comes from skipping repeated work
    across the multiple ``compute_metrics`` calls one rotation makes."""
    cache = CACHE

    gt_native = cache.get_native(gt_sketch, context, eps=eps)
    pred_native = cache.get_native(pred_sketch, context, eps=eps)

    J_phi_gt = J_phi_pr = None
    if base_eids is not None:
        J_phi_gt = cache.get_jphi(gt_sketch, base_eids)
        J_phi_pr = cache.get_jphi(pred_sketch, base_eids)

    J_R_gt, _n_R_gt_units, _skipped_rel_gt = cache.get_jr(
        rels, gt_native.state, context, eps=eps)
    J_R_pr, n_R_units, skipped_rel_pr = cache.get_jr(
        rels, pred_native.state, context, eps=eps)
    J_R_pr_tilde, n_R_tilde_units, skipped_rel_pr_tilde = cache.get_jr(
        rels, pred_native.state, context, eps=eps,
        normalization="per_relation", scale_state=gt_native.state,
    )
    n_R_rows = 0 if J_R_pr is None else J_R_pr.shape[0]
    sqrt_R = math.sqrt(max(n_R_rows, 1))

    d_resp_total = 0.0
    d_cresp_total = 0.0
    d_leak_total = 0.0
    d_feas_total = 0.0
    d_feas_tilde_total = 0.0
    d_feas_gain_total = 0.0
    d_feas_leak_total = 0.0
    d_feas_tilde_leak_total = 0.0
    d_align_total = 0.0
    d_align_phi_total = 0.0
    weight_align = 0.0
    weight_align_phi = 0.0
    weight_feas = 0.0
    attain_total = 0.0
    cattain_total = 0.0
    skipped = 0
    n_phi_rows = 0

    for probe, weight in zip(edit_probes, edit_weights):
        Jq_gt = cache.get_jq(probe, gt_native.state, context)
        Jq_pr = cache.get_jq(probe, pred_native.state, context)
        S_gt = _edit_response_from_jq(gt_native, Jq_gt, eps)
        S_pr = _edit_response_from_jq(pred_native, Jq_pr, eps)
        S_pr_c = _constrained_edit_response_from_jq(
            pred_native, Jq_pr, J_R_pr, eps)
        if S_gt is None or S_pr is None or S_pr_c is None:
            skipped += 1
            continue

        if J_phi_gt is not None and J_phi_pr is not None:
            M_gt = J_phi_gt @ S_gt
            M_pr = J_phi_pr @ S_pr
            M_pr_c = J_phi_pr @ S_pr_c
        else:
            M_gt = S_gt
            M_pr = S_pr
            M_pr_c = S_pr_c
        if M_gt.shape != M_pr.shape:
            skipped += 1
            continue

        n_phi_rows = M_gt.shape[0]
        sqrt_phi = math.sqrt(max(n_phi_rows, 1))

        d_resp_total += float(weight) * float(
            torch.linalg.norm(M_pr - M_gt) / sqrt_phi)
        d_cresp_total += float(weight) * float(
            torch.linalg.norm(M_pr_c - M_gt) / sqrt_phi)

        leak_ms = 0.0
        leak_tilde_ms = 0.0
        if J_R_pr is not None and J_R_pr.shape[0] > 0:
            leak = J_R_pr @ S_pr
            leak_norm = float(torch.linalg.norm(leak))
            d_leak_total += float(weight) * leak_norm / sqrt_R
            leak_ms = _mean_square_rows(leak, n_R_units)

            s_pr_norm = float(torch.linalg.norm(S_pr))
            if s_pr_norm > eps:
                d_align_total += float(weight) * (leak_norm / s_pr_norm)
                weight_align += float(weight)

            phi_pr_norm = float(torch.linalg.norm(M_pr))
            if phi_pr_norm > eps:
                d_align_phi_total += float(weight) * (
                    (leak_norm / sqrt_R) / (phi_pr_norm / sqrt_phi))
                weight_align_phi += float(weight)

        if J_R_pr_tilde is not None and J_R_pr_tilde.shape[0] > 0:
            leak_tilde = J_R_pr_tilde @ S_pr
            leak_tilde_ms = _mean_square_rows(leak_tilde, n_R_tilde_units)

        if Jq_pr is not None:
            eye = torch.eye(Jq_pr.shape[0], dtype=Jq_pr.dtype, device=Jq_pr.device)
            gain_residual = Jq_pr @ S_pr - eye
            gain_ms = _mean_square_rows(gain_residual, Jq_pr.shape[0])
            gain_norm = math.sqrt(gain_ms)
            leak_norm_balanced = math.sqrt(leak_ms)
            leak_tilde_norm_balanced = math.sqrt(leak_tilde_ms)

            d_feas_total += float(weight) * math.sqrt(gain_ms + leak_ms)
            d_feas_tilde_total += float(weight) * math.sqrt(
                gain_ms + leak_tilde_ms)
            d_feas_gain_total += float(weight) * gain_norm
            d_feas_leak_total += float(weight) * leak_norm_balanced
            d_feas_tilde_leak_total += (
                float(weight) * leak_tilde_norm_balanced)
            weight_feas += float(weight)

            attain_total += float(weight) * float(torch.linalg.norm(gain_residual))
            cattain_total += float(weight) * float(
                torch.linalg.norm(Jq_pr @ S_pr_c - eye))

    d_align = d_align_total / weight_align if weight_align > 0 else 0.0
    d_align_phi = d_align_phi_total / weight_align_phi if weight_align_phi > 0 else 0.0
    d_feas = d_feas_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_tilde = d_feas_tilde_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_gain = d_feas_gain_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_leak = d_feas_leak_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_tilde_leak = (
        d_feas_tilde_leak_total / weight_feas if weight_feas > 0 else 0.0)

    return {
        "d_resp": d_resp_total,
        "d_cresp": d_cresp_total,
        "d_leak": d_leak_total,
        "d_feas": d_feas,
        "d_feas_tilde": d_feas_tilde,
        "d_feas_gain": d_feas_gain,
        "d_feas_leak": d_feas_leak,
        "d_feas_tilde_leak": d_feas_tilde_leak,
        "d_align": d_align,
        "d_align_phi": d_align_phi,
        "attain_err": attain_total,
        "cattain_err": cattain_total,
        "skipped_probes": skipped,
        "skipped_relations": skipped_rel_pr,
        "skipped_relations_tilde": skipped_rel_pr_tilde,
        "n_phi_rows": n_phi_rows,
        "n_rel_rows": n_R_rows,
        "n_rel_units": n_R_units,
        "n_rel_tilde_units": n_R_tilde_units,
        "n_relations": len(rels),
        "n_dim_pred": len(pred_native.dim_constraint_ids),
        "n_dim_gt": len(gt_native.dim_constraint_ids),
    }


# ---------------------------------------------------------------------------
# Cached compute_pair_metrics (exp_smatch_metric)
# ---------------------------------------------------------------------------

def compute_pair_metrics_cached(
    *,
    pred_sketch,
    gt_sketch,
    edit_probes,
    edit_weights,
    rels,
    context,
    eps: float,
):
    """Drop-in replacement for
    ``exp_smatch_metric.compute_pair_metrics`` using :data:`CACHE`."""
    cache = CACHE
    gt_native = cache.get_native(gt_sketch, context, eps=eps)
    pred_native = cache.get_native(pred_sketch, context, eps=eps)

    # smatch's stack_relation_jacobians is the "raw, no skip-on-no-row"
    # variant. Reuse our cached per-(rel,state) Jq machinery and stack.
    rel_rows = []
    for rel in rels:
        j = cache.get_jq(rel, pred_native.state, context)
        if j is None or j.shape[0] == 0:
            continue
        rel_rows.append(j)
    if rel_rows:
        J_R_pr = torch.cat(rel_rows, dim=0)
        n_R_rows = J_R_pr.shape[0]
    else:
        J_R_pr = None
        n_R_rows = 0

    d_smatch_rms_total = 0.0
    d_smatch_rel_total = 0.0
    d_smatch_sym_total = 0.0
    d_feas_total = 0.0
    d_feas_gain_total = 0.0
    d_feas_leak_total = 0.0
    weight = 0.0
    skipped = 0

    layout_ok = (
        gt_native.state.num_dof() == pred_native.state.num_dof()
        and gt_native.state.entity_ids == pred_native.state.entity_ids
    )

    for probe, w in zip(edit_probes, edit_weights):
        Jq_gt = cache.get_jq(probe, gt_native.state, context)
        Jq_pr = cache.get_jq(probe, pred_native.state, context)
        S_gt = _edit_response_from_jq(gt_native, Jq_gt, eps)
        S_pr = _edit_response_from_jq(pred_native, Jq_pr, eps)
        if S_gt is None or S_pr is None:
            skipped += 1
            continue
        if S_gt.shape != S_pr.shape:
            skipped += 1
            continue

        diff = S_pr - S_gt
        diff_norm = float(torch.linalg.norm(diff))
        gt_norm = float(torch.linalg.norm(S_gt))
        pr_norm = float(torch.linalg.norm(S_pr))

        n_x, k_q = S_gt.shape
        denom_rms = math.sqrt(max(n_x * k_q, 1))

        d_smatch_rms = diff_norm / denom_rms
        d_smatch_rel = diff_norm / max(gt_norm, eps)
        d_smatch_sym = 2.0 * diff_norm / (pr_norm + gt_norm + eps)

        d_smatch_rms_total += float(w) * d_smatch_rms
        d_smatch_rel_total += float(w) * d_smatch_rel
        d_smatch_sym_total += float(w) * d_smatch_sym

        if Jq_pr is not None:
            eye = torch.eye(Jq_pr.shape[0], dtype=Jq_pr.dtype, device=Jq_pr.device)
            gain_residual = Jq_pr @ S_pr - eye
            gain_ms = float(torch.linalg.norm(gain_residual) ** 2) / max(Jq_pr.shape[0], 1)
            if J_R_pr is not None and J_R_pr.shape[0] > 0:
                leak = J_R_pr @ S_pr
                leak_ms = float(torch.linalg.norm(leak) ** 2) / max(J_R_pr.shape[0], 1)
            else:
                leak_ms = 0.0
            d_feas_total += float(w) * math.sqrt(gain_ms + leak_ms)
            d_feas_gain_total += float(w) * math.sqrt(gain_ms)
            d_feas_leak_total += float(w) * math.sqrt(leak_ms)

        weight += float(w)

    if weight <= 0:
        return None

    return {
        "d_smatch_rms": d_smatch_rms_total / weight,
        "d_smatch_rel": d_smatch_rel_total / weight,
        "d_smatch_sym": d_smatch_sym_total / weight,
        "d_feas": d_feas_total / weight,
        "d_feas_gain": d_feas_gain_total / weight,
        "d_feas_leak": d_feas_leak_total / weight,
        "skipped_probes": skipped,
        "layout_ok": int(layout_ok),
        "n_rel_rows": n_R_rows,
        "n_x_pred": pred_native.state.num_dof(),
        "n_x_gt": gt_native.state.num_dof(),
    }


# ---------------------------------------------------------------------------
# Cached response_distance + edit_attainability (exp_source_conditioned_dropout)
# ---------------------------------------------------------------------------

def response_distance_cached(*, pred_sketch, gt_sketch, edit_probes,
                             edit_weights, context, base_eids):
    """Drop-in replacement for
    ``exp_source_conditioned_dropout.response_distance``.

    Only the pairwise-distance observable construction is cached;
    ``compute_edit_response_distance`` itself still runs untouched so
    its return shape / error semantics are bit-identical.
    """
    try:
        j_gt = j_pred = None
        if base_eids is not None:
            j_gt = CACHE.get_jphi(gt_sketch, base_eids)
            j_pred = CACHE.get_jphi(pred_sketch, base_eids)
        r = compute_edit_response_distance(
            gt_sketch=gt_sketch, pred_sketch=pred_sketch,
            edit_probes=edit_probes, context=context,
            probe_weights=edit_weights,
            observable_jacobian_gt=j_gt, observable_jacobian_pred=j_pred)
        return r.distance, r.n_dim_pred, r.n_dim_gt, r.skipped_probes, ""
    except Exception as e:
        return None, None, None, None, str(e)[:200]


def edit_attainability_cached(*, sketch, edit_probes, edit_weights, context,
                              eps=1e-8):
    """Drop-in replacement for
    ``exp_source_conditioned_dropout.edit_attainability``."""
    try:
        native = CACHE.get_native(sketch, context, eps=eps)
    except Exception:
        return None, len(edit_probes)
    total = total_w = 0.0
    skipped = 0
    for probe, w in zip(edit_probes, edit_weights):
        j_q = CACHE.get_jq(probe, native.state, context)
        s_q = _edit_response_from_jq(native, j_q, eps)
        if j_q is None or s_q is None:
            skipped += 1
            continue
        gain = j_q @ s_q
        I = torch.eye(gain.shape[0], dtype=gain.dtype)
        total += float(w) * float(torch.linalg.norm(gain - I))
        total_w += float(w)
    return (total / total_w if total_w > 0 else None), skipped


# ---------------------------------------------------------------------------
# install() — patch a module's slow entry points to use the cache
# ---------------------------------------------------------------------------

def install(module, *, wrap_evaluate_rotation: bool = True) -> None:
    """Patch ``compute_metrics`` / ``compute_pair_metrics`` /
    ``response_distance`` / ``edit_attainability`` on ``module`` and
    optionally wrap its ``evaluate_rotation`` so the cache resets
    between rotations.

    Only names that already exist on the module are replaced, so a
    given wrapper can call ``install`` for any module without knowing
    which symbols are defined there.
    """
    if hasattr(module, "compute_metrics"):
        module.compute_metrics = compute_metrics_cached
    if hasattr(module, "compute_pair_metrics"):
        module.compute_pair_metrics = compute_pair_metrics_cached
    if hasattr(module, "response_distance"):
        module.response_distance = response_distance_cached
    if hasattr(module, "edit_attainability"):
        module.edit_attainability = edit_attainability_cached

    if wrap_evaluate_rotation and hasattr(module, "evaluate_rotation"):
        original = module.evaluate_rotation

        def _wrapped(*args, **kwargs):
            clear()
            return original(*args, **kwargs)

        _wrapped.__wrapped__ = original  # type: ignore[attr-defined]
        module.evaluate_rotation = _wrapped
