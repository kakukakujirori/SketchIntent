"""
exp_constrained_response_metric.py — alternative single-knob metrics.

This script evaluates two candidates that aim to replace the augmented
response distance d_aug_beta with simpler single-knob designs.

Metrics computed per (Pred, GT) pair:

  d_resp(P, G)
      = sum_q w_q || J_phi (S_P^(q) - S_G^(q)) ||_F                 [baseline]

  d_cresp(P, G; R_G)                                            [proposal A]
      = sum_q w_q || J_phi (S_P^(q; R_G) - S_G^(q)) ||_F
    where S_P^(q; R_G) = R_P pinv([J_q R_P; J_{R_G} R_P]) [I; 0]
    enforces preservation of GT's relations as a constraint on Pred's
    response solver. For GT itself this is a no-op since J_{R_G} R_G = 0.

  d_leak(P, G; R_G)                                             [proposal B]
      = sum_q w_q || J_{R_G} S_P^(q) ||_F / sqrt(|R_G|)
    pure relation-leakage measure with no geometry block. GT scores 0 by
    construction; Vitruvion's over-parameterised response leaks.

Both are normalised by a human-human scale h_scale per source GT, just like
the existing norm_d_resp / norm_d_aug.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from itertools import combinations

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.dirname(__file__)
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from exp_source_conditioned_dropout import (
    DEFAULT_BATCHES,
    HistCADConfig,
    base_policy_pass,
    build_base_eids,
    build_file_bundle,
    dropout_variant,
    finite_or_none,
    get_fcstd_map,
    norm_by_scale,
    parse_human_specs,
    parse_percentages,
    parse_signs,
    parse_vitruvion_specs,
    safe_median,
    source_probes,
    stable_seed,
    validate_probes_on_source,
)
from src.metrics.edit_response_distance import (
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
    pairwise_distance_observable,
    _abs_pinv,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def constrained_edit_response_for_probe(
    native,
    probe,
    J_R_external: torch.Tensor | None,
    context,
    eps: float = 1e-9,
    rcond: float = 1e-6,
):
    """S = R · pinv([J_q R; J_{R_ext} R]) · [I; 0].

    R = native.R, the (n_x, n_p) native-dim response of Pred. Adding the
    bottom block J_{R_ext} R = 0 forces dx = R dp to lie in the null space
    of the external relation Jacobian. For dq = e_i (unit edit), the
    resulting dx is the minimum-norm-dp response that achieves the edit
    *and* preserves the external relations to first order.
    """
    Jq = constraint_jacobian(probe, native.state, context)
    if Jq is None:
        return None
    k_q = Jq.shape[0]
    n_x = native.R.shape[0]
    if native.R.shape[1] == 0:
        return torch.zeros(n_x, k_q, dtype=Jq.dtype, device=Jq.device)

    A_top = Jq @ native.R  # (k_q, n_p)
    if J_R_external is not None and J_R_external.shape[0] > 0:
        A_bot = J_R_external @ native.R  # (n_R, n_p)
        A = torch.cat([A_top, A_bot], dim=0)
        b = torch.zeros(A.shape[0], k_q, dtype=Jq.dtype, device=Jq.device)
        b[:k_q, :] = torch.eye(k_q, dtype=Jq.dtype, device=Jq.device)
        # Stacked A has sigma_max from J_R rows; rcond relative to that is fine.
        A_pinv = _abs_pinv(A, eps=eps, rcond=rcond)
    else:
        A = A_top
        b = torch.eye(k_q, dtype=Jq.dtype, device=Jq.device)
        # Unconstrained: normalize eps_A by ||J_q|| * ||R||_op so near-zero SVs
        # of A are dropped relative to the physical scale of the probe.
        if rcond > 0.0:
            Jq_norm = float(torch.linalg.norm(Jq))
            R_op = float(torch.linalg.norm(native.R, ord=2))
            eps_A = max(eps, rcond * Jq_norm * R_op)
        else:
            eps_A = eps
        A_pinv = _abs_pinv(A, eps=eps_A, rcond=0.0)
    dp_star = A_pinv @ b  # (n_p, k_q)
    return native.R @ dp_star  # (n_x, k_q)


def stack_relation_jacobians(
    rels,
    state,
    context,
    *,
    eps: float,
    normalization: str = "raw",
    scale_state=None,
):
    """Stack residual Jacobians J_r of each relation at a given state.

    For ``normalization='per_relation'``, each relation block is divided by
    the Frobenius norm of the same relation at ``scale_state``.  The returned
    ``unit_count`` is then the number of relation blocks rather than the
    number of scalar residual rows, matching the per-relation tilde_J_R
    convention used by the augmented metric.
    """
    rows = []
    unit_count = 0
    skipped = 0
    for rel in rels:
        j = constraint_jacobian(rel, state, context)
        if j is None or j.shape[0] == 0:
            skipped += 1
            continue
        if normalization == "per_relation":
            scale_j = constraint_jacobian(
                rel, scale_state if scale_state is not None else state, context)
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


def mean_square_rows(mat: torch.Tensor, row_units: int) -> float:
    if row_units <= 0 or mat.numel() == 0:
        return 0.0
    return float(torch.linalg.norm(mat) ** 2) / float(row_units)


def compute_metrics(
    *,
    pred_sketch,
    gt_sketch,
    edit_probes,
    edit_weights,
    rels,
    context,
    base_eids,
    eps: float,
    rcond: float = 1e-6,
):
    """Compute d_resp, d_cresp, d_leak for this (pred, gt) pair using
    `rels` (=GT's relations) as the source relation set R_G."""

    gt_native = compute_native_response(gt_sketch, context, eps=eps, rcond=rcond)
    pred_native = compute_native_response(pred_sketch, context, eps=eps, rcond=rcond)

    J_phi_gt = J_phi_pr = None
    if base_eids is not None:
        J_phi_gt = pairwise_distance_observable(gt_sketch, base_eids=base_eids)
        J_phi_pr = pairwise_distance_observable(pred_sketch, base_eids=base_eids)

    # Source relation Jacobians evaluated at each state.
    J_R_gt, _n_R_gt_units, _skipped_rel_gt = stack_relation_jacobians(
        rels, gt_native.state, context, eps=eps)
    J_R_pr, n_R_units, skipped_rel_pr = stack_relation_jacobians(
        rels, pred_native.state, context, eps=eps)
    J_R_pr_tilde, n_R_tilde_units, skipped_rel_pr_tilde = stack_relation_jacobians(
        rels,
        pred_native.state,
        context,
        eps=eps,
        normalization="per_relation",
        scale_state=gt_native.state,
    )
    n_R_rows = 0 if J_R_pr is None else J_R_pr.shape[0]
    sqrt_R = math.sqrt(max(n_R_rows, 1))

    d_resp_total = 0.0
    d_cresp_total = 0.0
    d_leak_total = 0.0
    d_feas_total = 0.0        # row-balanced gain + raw relation leak residual
    d_feas_tilde_total = 0.0  # row-balanced gain + per-relation-normalized leak
    d_feas_gain_total = 0.0
    d_feas_leak_total = 0.0
    d_feas_tilde_leak_total = 0.0
    d_align_total = 0.0          # leakage / response_magnitude, unitless
    d_align_phi_total = 0.0      # leakage / ||J_phi S_P||, unitless
    weight_align = 0.0
    weight_align_phi = 0.0
    weight_feas = 0.0
    attain_total = 0.0          # natural attainability of pred (||gain-I||)
    cattain_total = 0.0         # attainability of constrained pred
    skipped = 0
    n_phi_rows = 0

    for probe, weight in zip(edit_probes, edit_weights):
        S_gt = edit_response_for_probe(gt_native, probe, context, eps=eps, rcond=rcond)
        S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps, rcond=rcond)
        S_pr_c = constrained_edit_response_for_probe(
            pred_native, probe, J_R_pr, context, eps=eps, rcond=rcond)
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
            leak_ms = mean_square_rows(leak, n_R_units)

            s_pr_norm = float(torch.linalg.norm(S_pr))
            if s_pr_norm > eps:
                # Fraction of pred response in primitive space that
                # violates GT relations. Unitless, in [0, 1] (since J_R_pr
                # is not normalised, can exceed 1 if relations are scaled).
                d_align_total += float(weight) * (leak_norm / s_pr_norm)
                weight_align += float(weight)

            phi_pr_norm = float(torch.linalg.norm(M_pr))
            if phi_pr_norm > eps:
                # Relation leakage relative to observable response magnitude.
                d_align_phi_total += float(weight) * (
                    (leak_norm / sqrt_R) / (phi_pr_norm / sqrt_phi))
                weight_align_phi += float(weight)

        if J_R_pr_tilde is not None and J_R_pr_tilde.shape[0] > 0:
            leak_tilde = J_R_pr_tilde @ S_pr
            leak_tilde_ms = mean_square_rows(leak_tilde, n_R_tilde_units)

        # Diagnostics: gain error (||J_q S - I||) for unconstrained & constrained.
        Jq_pr = constraint_jacobian(probe, pred_native.state, context)
        if Jq_pr is not None:
            eye = torch.eye(Jq_pr.shape[0], dtype=Jq_pr.dtype, device=Jq_pr.device)
            gain_residual = Jq_pr @ S_pr - eye
            gain_ms = mean_square_rows(gain_residual, Jq_pr.shape[0])
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


METRIC_NAMES = (
    "d_resp",
    "d_cresp",
    "d_leak",
    "d_feas",
    "d_feas_tilde",
    "d_align",
    "d_align_phi",
)


def write_csv(path: str, rows: list[dict]):
    full = repo_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(full, "w", newline="") as f:
        if not keys:
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {full}")


def evaluate_rotation(
    *,
    bundle,
    refs,
    heldout,
    drop_rates,
    samples_per_rate,
    context,
    converter,
    histcad_cfg,
    base_policy,
    eps,
    rcond: float = 1e-6,
    eval_model=True,
):
    rotation_id = f"{bundle.batch}:{bundle.filename}:{'+'.join(refs)}__query={heldout}"
    participants = set(refs) | {heldout}
    bad = []
    for name in participants:
        ok, reason = base_policy_pass(bundle.audits[name], base_policy)
        if not ok:
            bad.append(f"{name}:{reason}")
    if bad:
        return [{"batch": bundle.batch, "filename": bundle.filename,
                 "rotation_id": rotation_id, "error": ",".join(bad)}], []

    heldout_sketch = bundle.sketches[heldout]
    base_eids = build_base_eids(bundle.canonical_sketch, base_policy)

    source_info = {}
    for gt_name in refs:
        gt_sketch = bundle.sketches[gt_name]
        probes, weights, rels, _rel_weights = source_probes(gt_sketch)
        mask = validate_probes_on_source(
            gt_sketch, probes, rels, context, converter, histcad_cfg)
        source_info[gt_name] = {
            "probes": probes,
            "weights": weights,
            "rels": rels,
            "valid_mask": mask,
            "n_before_validation": len(probes),
            "n_after_validation": sum(mask) if mask else 0,
        }

    # h_scale per source: median over other humans evaluated against this source.
    h_scales = {}  # (gt_name, validated, metric_name) -> float|None
    validation_modes = (True,)
    for gt_name in refs:
        other_humans = [name for name in refs if name != gt_name]
        info = source_info[gt_name]
        for validated in validation_modes:
            if validated:
                probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
                weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
            else:
                probes, weights = info["probes"], info["weights"]
            if not probes:
                for m in METRIC_NAMES:
                    h_scales[(gt_name, validated, m)] = None
                continue
            per_metric: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
            for other in other_humans:
                res = compute_metrics(
                    pred_sketch=bundle.sketches[other],
                    gt_sketch=bundle.sketches[gt_name],
                    edit_probes=probes,
                    edit_weights=weights,
                    rels=info["rels"],
                    context=context,
                    base_eids=base_eids,
                    eps=eps,
                    rcond=rcond,
                )
                for m in METRIC_NAMES:
                    v = finite_or_none(res.get(m))
                    if v is not None:
                        per_metric[m].append(v)
            for m in METRIC_NAMES:
                h_scales[(gt_name, validated, m)] = safe_median(per_metric[m])
    # Floor h_scale for d_leak so norm doesn't explode when humans share intent.
    # We use the same floor convention as h_scale eps in source_conditioned.
    pass

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    def add_pred_rows(pred_name, pred_sketch, drop_rate, sample_id,
                      n_orig=-1, n_kept=-1, n_dropped=-1):
        for validated in validation_modes:
            per_source_results = []
            for gt_name in refs:
                info = source_info[gt_name]
                if validated:
                    probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
                    weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
                else:
                    probes, weights = info["probes"], info["weights"]
                if not probes:
                    per_source_results.append(None)
                    continue
                res = compute_metrics(
                    pred_sketch=pred_sketch,
                    gt_sketch=bundle.sketches[gt_name],
                    edit_probes=probes,
                    edit_weights=weights,
                    rels=info["rels"],
                    context=context,
                    base_eids=base_eids,
                    eps=eps,
                    rcond=rcond,
                )
                row = {
                    "batch": bundle.batch,
                    "filename": bundle.filename,
                    "rotation_id": rotation_id,
                    "refs": "|".join(refs),
                    "heldout": heldout,
                    "sample_id": sample_id,
                    "drop_rate": drop_rate,
                    "pred": pred_name,
                    "gt": gt_name,
                    "source_validated": validated,
                    "n_edit_probes": len(probes),
                    "n_probes_before_validation": info["n_before_validation"],
                    "n_probes_after_validation": info["n_after_validation"],
                    **res,
                    **{f"h_scale_{m}": h_scales.get((gt_name, validated, m))
                       for m in METRIC_NAMES},
                    "error": "",
                }
                detail_rows.append(row)
                per_source_results.append(row)

            valid = [r for r in per_source_results if r is not None]
            srow = {
                "batch": bundle.batch,
                "filename": bundle.filename,
                "rotation_id": rotation_id,
                "refs": "|".join(refs),
                "heldout": heldout,
                "sample_id": sample_id,
                "drop_rate": drop_rate,
                "pred": pred_name,
                "source_validated": validated,
                "n_dim_original": n_orig,
                "n_dim_kept": n_kept,
                "n_dim_dropped": n_dropped,
                "valid_sources": len(valid),
                "error": "",
            }
            for m in METRIC_NAMES:
                d_vals = [finite_or_none(r.get(m)) for r in valid]
                d_vals = [v for v in d_vals if v is not None]
                if not d_vals:
                    srow[f"min_{m}"] = None
                    srow[f"norm_{m}"] = None
                    srow[f"h_scale_{m}"] = None
                    srow[f"best_gt_{m}"] = ""
                    continue
                min_v = min(d_vals)
                best = None
                for r in valid:
                    if finite_or_none(r.get(m)) == min_v:
                        best = r
                        break
                hs = best.get(f"h_scale_{m}") if best else None
                srow[f"min_{m}"] = min_v
                srow[f"norm_{m}"] = norm_by_scale(min_v, hs, eps)
                srow[f"h_scale_{m}"] = hs
                srow[f"best_gt_{m}"] = best.get("gt") if best else ""
            srow["median_attain_err"] = safe_median(
                [r.get("attain_err") for r in valid])
            srow["median_cattain_err"] = safe_median(
                [r.get("cattain_err") for r in valid])
            srow["median_d_feas_gain"] = safe_median(
                [r.get("d_feas_gain") for r in valid])
            srow["median_d_feas_leak"] = safe_median(
                [r.get("d_feas_leak") for r in valid])
            srow["median_d_feas_tilde_leak"] = safe_median(
                [r.get("d_feas_tilde_leak") for r in valid])
            summary_rows.append(srow)

    for drop_rate in drop_rates:
        sample_count = 1 if drop_rate in (0.0, 100.0) else samples_per_rate
        for sample_id in range(sample_count):
            seed = stable_seed(bundle.batch, bundle.filename, rotation_id,
                               drop_rate, sample_id)
            pred_var, n_orig, n_kept, n_dropped = dropout_variant(
                heldout_sketch, drop_rate=drop_rate, seed=seed)
            add_pred_rows(
                pred_name=f"{heldout}_drop{drop_rate:g}",
                pred_sketch=pred_var,
                drop_rate=drop_rate,
                sample_id=sample_id,
                n_orig=n_orig,
                n_kept=n_kept,
                n_dropped=n_dropped,
            )

    if eval_model:
        for model_label in bundle.loaded_model_labels():
            ok, _reason = base_policy_pass(bundle.audits[model_label], base_policy)
            if not ok:
                continue
            add_pred_rows(
                pred_name=model_label,
                pred_sketch=bundle.sketches[model_label],
                drop_rate=-1,
                sample_id=0,
            )

    return summary_rows, detail_rows


def print_summary(rows: list[dict]):
    valid = [r for r in rows if not r.get("error")]
    print("\n--- Source-conditioned metrics: d_resp / d_cresp / d_leak / d_feas ---")
    for validated in (True,):
        tag = "validated"
        rv = [r for r in valid if r.get("source_validated") == validated]
        print(f"\n  [{tag}] n={len(rv)}")
        groups: dict[tuple, list[dict]] = {}
        for r in rv:
            rate = float(r["drop_rate"])
            key = (rate, r.get("pred") or "Vitruvion") if rate == -1 else (rate, None)
            groups.setdefault(key, []).append(r)
        for (rate, pred_label) in sorted(groups, key=lambda k: (k[0], k[1] or "")):
            rs = groups[(rate, pred_label)]
            label = (pred_label or "Vitruvion") if rate == -1 else f"drop={rate:5.1f}%"
            print(
                f"    {label:14s} n={len(rs):4d}"
                f"  d_resp={safe_median([r.get('norm_d_resp') for r in rs])}"
                f"  d_cresp={safe_median([r.get('norm_d_cresp') for r in rs])}"
                f"  d_leak={safe_median([r.get('norm_d_leak') for r in rs])}"
                f"  d_feas={safe_median([r.get('norm_d_feas') for r in rs])}"
                f"  d_feas_tilde={safe_median([r.get('norm_d_feas_tilde') for r in rs])}"
                f"  attain={safe_median([r.get('median_attain_err') for r in rs])}"
                f"  cattain={safe_median([r.get('median_cattain_err') for r in rs])}"
            )


def paired_inversion(rows: list[dict], metric: str,
                     value_key: str = "min",
                     pred_label: str = "Vitruvion") -> tuple[int, int, int]:
    """Pair `pred_label` model vs heldout drop 0 within the same rotation.
    Returns (model_better, human_better, tie). value_key is 'min' or 'norm'."""
    field = f"{value_key}_{metric}"
    by_rot: dict[str, dict[str, float | None]] = {}
    for r in rows:
        if r.get("error") or not r.get("source_validated"):
            continue
        rotid = r["rotation_id"]
        d = by_rot.setdefault(rotid, {})
        if float(r["drop_rate"]) == -1.0 and r.get("pred") == pred_label:
            d["vit"] = finite_or_none(r.get(field))
        elif float(r["drop_rate"]) == 0.0:
            d["h0"] = finite_or_none(r.get(field))
    model, human, tie = 0, 0, 0
    for rotid, vals in by_rot.items():
        v, h = vals.get("vit"), vals.get("h0")
        if v is None or h is None:
            continue
        if abs(v - h) <= 1e-9:
            tie += 1
        elif v < h:
            model += 1
        else:
            human += 1
    return model, human, tie


def loaded_model_labels_in_rows(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if float(r.get("drop_rate", 0)) == -1.0:
            lbl = r.get("pred")
            if lbl and lbl not in seen:
                seen.append(lbl)
    return seen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--vitruvion-root",
        default=None,
        help="Legacy single-model root. Prefer --vitruvion-spec.",
    )
    parser.add_argument(
        "--vitruvion-spec",
        action="append",
        default=None,
        help="label:path, repeatable. Defaults to base + solver_rl when neither flag given.",
    )
    parser.add_argument("--batches", nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--human", action="append", default=None)
    parser.add_argument("--ref-size", type=int, default=2)
    parser.add_argument("--drop-rates", default="0,100")
    parser.add_argument("--samples-per-rate", type=int, default=1)
    parser.add_argument("--base-policy", default="canonical_solid",
                        choices=("none", "canonical_solid", "canonical_all"))
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument("--canonical-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-batch", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument(
        "--rcond", type=float, default=1e-6,
        help="Relative singular value cutoff for pinv(J_x): eps_eff = max(eps, rcond*sigma_max). "
             "Set to 0 to use absolute eps only.")
    parser.add_argument(
        "--coordinate-scale",
        type=float,
        default=1.0,
        help="Uniform coordinate multiplier applied before metric computation.",
    )
    parser.add_argument(
        "--bbox-normalize-dst-length",
        type=float,
        default=1.0,
        help=(
            "If set, center by the canonical bbox and rescale its long axis to "
            "this length before applying --coordinate-scale. Use 1.0 for the "
            "repo's [-0.5,0.5] convention."
        ),
    )
    parser.add_argument(
        "--no-bbox-normalization",
        action="store_true",
        help="Use raw coordinates except for any --coordinate-scale multiplier.",
    )
    parser.add_argument("--delta-basis", default="bbox_diag")
    parser.add_argument("--delta-scale", type=float, default=0.01)
    parser.add_argument("--delta-absolute", type=float, default=1.0)
    parser.add_argument("--delta-signs", default="-1,+1")
    parser.add_argument("--newton-max-iter", type=int, default=80)
    parser.add_argument("--newton-tol", type=float, default=1e-6)
    parser.add_argument("--er-target-tol", type=float, default=1e-3)
    parser.add_argument("--relation-tol", type=float, default=1e-3)
    parser.add_argument("--output-csv",
                        default="artifacts/constrained_response_scores.csv")
    parser.add_argument("--details-csv",
                        default="artifacts/constrained_response_details.csv")
    return parser.parse_args()


def main():
    import _fast

    _fast.install(sys.modules[__name__])

    args = parse_args()
    humans_by_batch = parse_human_specs(args.human)
    drop_rates = parse_percentages(args.drop_rates)
    histcad_cfg = HistCADConfig(
        delta_basis=args.delta_basis,
        delta_scale=args.delta_scale,
        delta_absolute=args.delta_absolute,
        delta_signs=parse_signs(args.delta_signs),
        newton_max_iter=args.newton_max_iter,
        newton_tol=args.newton_tol,
        er_target_tol=args.er_target_tol,
        relation_tol_default=args.relation_tol,
        relation_source="gt",
    )

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    n_files = 0
    bbox_dst = None if args.no_bbox_normalization else args.bbox_normalize_dst_length
    model_specs = parse_vitruvion_specs(args.vitruvion_spec, args.vitruvion_root)
    if model_specs:
        print("[Vitruvion specs] " + ", ".join(f"{l}={p}" for l, p in model_specs))

    for batch in args.batches:
        humans = [(n, d) for n, d in humans_by_batch.get(batch, [])
                  if os.path.isdir(d)]
        if len(humans) < args.ref_size + 1:
            print(f"[skip] batch {batch}: need {args.ref_size + 1} humans")
            continue
        file_maps = {n: get_fcstd_map(d) for n, d in humans}
        common = sorted(set.intersection(*(set(m) for m in file_maps.values())))
        print(f"Batch {batch}: common={len(common)}")

        n_batch_files = 0
        for filename in common:
            if args.limit is not None and n_files >= args.limit:
                break
            if args.limit_per_batch is not None and n_batch_files >= args.limit_per_batch:
                break
            n_files += 1
            n_batch_files += 1
            try:
                bundle = build_file_bundle(
                    batch=batch,
                    filename=filename,
                    humans=humans,
                    file_maps=file_maps,
                    converter=converter,
                    geometry_tol=args.geometry_tol,
                    canonical_dir=args.canonical_dir,
                    model_specs=model_specs,
                    coordinate_scale=args.coordinate_scale,
                    bbox_normalize_dst_length=bbox_dst,
                )
            except Exception as exc:
                print(f"  [skip] {batch}/{filename}: {exc}")
                continue

            names = tuple(name for name, _ in humans)
            n_rot = 0
            for refs in combinations(names, args.ref_size):
                for heldout in names:
                    if heldout in refs:
                        continue
                    srows, drows = evaluate_rotation(
                        bundle=bundle,
                        refs=tuple(refs),
                        heldout=heldout,
                        drop_rates=drop_rates,
                        samples_per_rate=args.samples_per_rate,
                        context=context,
                        converter=converter,
                        histcad_cfg=histcad_cfg,
                        base_policy=args.base_policy,
                        eps=args.eps,
                        rcond=args.rcond,
                    )
                    for row in srows:
                        row["coordinate_scale"] = args.coordinate_scale
                        row["bbox_normalize_dst_length"] = bbox_dst
                    for row in drows:
                        row["coordinate_scale"] = args.coordinate_scale
                        row["bbox_normalize_dst_length"] = bbox_dst
                    summary_rows.extend(srows)
                    detail_rows.extend(drows)
                    n_rot += 1
            print(f"  [done] {batch}/{filename}: {n_rot} rotations")

        if args.limit is not None and n_files >= args.limit:
            break

    print_summary(summary_rows)

    model_labels = loaded_model_labels_in_rows(summary_rows) or ["Vitruvion"]
    for lbl in model_labels:
        print(f"\n--- Paired {lbl} vs heldout drop 0% (validated) ---")
        for m in METRIC_NAMES:
            model_n, human_n, tie_n = paired_inversion(summary_rows, m, "norm", pred_label=lbl)
            model_r, human_r, tie_r = paired_inversion(summary_rows, m, "min", pred_label=lbl)
            print(
                f"  {m:14s}"
                f"  norm[m={model_n} h={human_n} t={tie_n}]"
                f"  raw [m={model_r} h={human_r} t={tie_r}]"
            )

    write_csv(args.output_csv, summary_rows)
    write_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
