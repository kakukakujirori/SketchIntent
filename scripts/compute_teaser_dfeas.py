"""
compute_teaser_dfeas.py — compute d_feas for teaser FCStd files.

GT:   artifacts/teaser1.FCStd
Pred: artifacts/teaser2.FCStd, teaser3.FCStd, teaser4.FCStd

d_feas(Pred_i; GT) is computed using:
  - Q_G = dimensional constraints from GT (edit probes)
  - R_G = relational constraints from GT (preservation set)
"""
from __future__ import annotations

import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

import torch

from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.metrics.edit_response_distance import (
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
    _is_relational,
    _abs_pinv,
    collect_probes,
)
from canonical_sketch import load_sketch_from_fcstd
from exp_constrained_response_metric import (
    stack_relation_jacobians,
    mean_square_rows,
    constrained_edit_response_for_probe,
)


def compute_dfeas(
    gt_sketch,
    pred_sketch,
    edit_probes,
    rels,
    context,
    eps: float = 1e-9,
    rcond: float = 1e-6,
) -> dict:
    """
    Compute d_feas(Pred; GT) given:
      edit_probes : dimensional constraints from GT (Q_G)
      rels        : relational constraints from GT (R_G)
    """
    gt_native = compute_native_response(gt_sketch, context, eps=eps, rcond=rcond)
    pred_native = compute_native_response(pred_sketch, context, eps=eps, rcond=rcond)

    J_R_pr, n_R_units, skipped_rel = stack_relation_jacobians(
        rels, pred_native.state, context, eps=eps
    )
    J_R_pr_tilde, n_R_tilde_units, _ = stack_relation_jacobians(
        rels, pred_native.state, context, eps=eps,
        normalization="per_relation",
        scale_state=gt_native.state,
    )
    n_R_rows = 0 if J_R_pr is None else J_R_pr.shape[0]

    d_feas_total = 0.0
    d_feas_tilde_total = 0.0
    d_feas_gain_total = 0.0
    d_feas_leak_total = 0.0
    weight_feas = 0.0
    skipped = 0

    probe_details = []

    for probe in edit_probes:
        S_gt = edit_response_for_probe(gt_native, probe, context, eps=eps, rcond=rcond)
        S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps, rcond=rcond)

        if S_gt is None or S_pr is None:
            skipped += 1
            probe_details.append(None)
            continue

        Jq_pr = constraint_jacobian(probe, pred_native.state, context)
        if Jq_pr is None:
            skipped += 1
            probe_details.append(None)
            continue

        k_q = Jq_pr.shape[0]
        eye = torch.eye(k_q, dtype=Jq_pr.dtype, device=Jq_pr.device)
        gain_residual = Jq_pr @ S_pr - eye
        gain_ms = mean_square_rows(gain_residual, k_q)

        leak_ms = 0.0
        leak_tilde_ms = 0.0
        if J_R_pr is not None and J_R_pr.shape[0] > 0:
            leak = J_R_pr @ S_pr
            leak_ms = mean_square_rows(leak, n_R_units)
        if J_R_pr_tilde is not None and J_R_pr_tilde.shape[0] > 0:
            leak_tilde = J_R_pr_tilde @ S_pr
            leak_tilde_ms = mean_square_rows(leak_tilde, n_R_tilde_units)

        probe_dfeas = math.sqrt(gain_ms + leak_ms)
        probe_dfeas_tilde = math.sqrt(gain_ms + leak_tilde_ms)
        probe_gain = math.sqrt(gain_ms)
        probe_leak = math.sqrt(leak_ms)

        d_feas_total += probe_dfeas
        d_feas_tilde_total += probe_dfeas_tilde
        d_feas_gain_total += probe_gain
        d_feas_leak_total += probe_leak
        weight_feas += 1.0

        ptype = probe.constraint_type.name
        probe_details.append({
            "type": ptype,
            "d_feas": probe_dfeas,
            "gain": probe_gain,
            "leak": probe_leak,
        })

    n_probes = len(edit_probes)
    d_feas = d_feas_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_tilde = d_feas_tilde_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_gain = d_feas_gain_total / weight_feas if weight_feas > 0 else 0.0
    d_feas_leak = d_feas_leak_total / weight_feas if weight_feas > 0 else 0.0

    return {
        "d_feas": d_feas,
        "d_feas_tilde": d_feas_tilde,
        "d_feas_gain": d_feas_gain,
        "d_feas_leak": d_feas_leak,
        "n_probes": n_probes,
        "n_valid_probes": int(weight_feas),
        "skipped_probes": skipped,
        "n_relations": len(rels),
        "n_rel_rows": n_R_rows,
        "n_dim_gt": len(gt_native.dim_constraint_ids),
        "n_dim_pred": len(pred_native.dim_constraint_ids),
        "probe_details": probe_details,
    }


def main():
    artifacts = os.path.join(REPO_ROOT, "artifacts")
    gt_path = os.path.join(artifacts, "teaser1.FCStd")
    pred_paths = {
        "teaser2": os.path.join(artifacts, "teaser2.FCStd"),
        "teaser3": os.path.join(artifacts, "teaser3.FCStd"),
        "teaser4": os.path.join(artifacts, "teaser4.FCStd"),
    }

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)

    print(f"Loading GT: {gt_path}")
    gt_sketch = load_sketch_from_fcstd(gt_path, converter)

    # Extract Q_G (dimensional probes) and R_G (relational constraints)
    probes, _ = collect_probes([gt_sketch], kind="dim")
    rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]

    print(f"GT: {len(gt_sketch.entities)} entities, "
          f"{len(gt_sketch.constraints)} constraints")
    print(f"  Dimensional probes (Q_G): {len(probes)}")
    print(f"  Relational constraints (R_G): {len(rels)}")

    dim_types = {}
    for p in probes:
        t = p.constraint_type.name
        dim_types[t] = dim_types.get(t, 0) + 1
    rel_types = {}
    for r in rels:
        t = r.constraint_type.name
        rel_types[t] = rel_types.get(t, 0) + 1
    print(f"  Probe types: {dim_types}")
    print(f"  Relation types: {rel_types}")

    print()
    results = {}
    for name, path in pred_paths.items():
        print(f"Loading {name}: {path}")
        pred_sketch = load_sketch_from_fcstd(path, converter)
        print(f"  {len(pred_sketch.entities)} entities, "
              f"{len(pred_sketch.constraints)} constraints "
              f"(dim={sum(1 for c in pred_sketch.constraints.values() if not _is_relational(c))}, "
              f"rel={sum(1 for c in pred_sketch.constraints.values() if _is_relational(c))})")

        r = compute_dfeas(gt_sketch, pred_sketch, probes, rels, context)
        results[name] = r

        print(f"  d_feas       = {r['d_feas']:.6f}")
        print(f"  d_feas_tilde = {r['d_feas_tilde']:.6f}")
        print(f"  gain         = {r['d_feas_gain']:.6f}")
        print(f"  leak         = {r['d_feas_leak']:.6f}")
        print(f"  valid probes = {r['n_valid_probes']}/{r['n_probes']} "
              f"(skipped={r['skipped_probes']})")
        print(f"  n_dim GT/Pred = {r['n_dim_gt']}/{r['n_dim_pred']}")
        print(f"  n_rel rows = {r['n_rel_rows']}")
        print()

    print("=" * 60)
    print("Summary table (GT = teaser1.FCStd)")
    print(f"{'Pred':<12} {'d_feas':>10} {'gain':>10} {'leak':>10} {'n_dim_pred':>12}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<12} {r['d_feas']:>10.4f} {r['d_feas_gain']:>10.4f} "
              f"{r['d_feas_leak']:>10.4f} {r['n_dim_pred']:>12}")


if __name__ == "__main__":
    main()
