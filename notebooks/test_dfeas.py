"""
compute_teaser_dfeas.py — compute d_feas for teaser FCStd files.

GT:   artifacts/teaser1.FCStd
Pred: artifacts/teaser2.FCStd, teaser3.FCStd, teaser4.FCStd

d_feas(Pred_i; GT) is computed using:
  - Q_G = dimensional constraints from GT (edit probes)
  - R_G = relational constraints from GT (preservation set)

Note on eps:
  The teaser sketches have gauge freedom (no absolute position/orientation fix),
  so J_x has near-zero singular values (4e-06, 1e-06 range) corresponding to
  rigid body modes.  Using eps=1e-9 (as in the main experiments, where sketches
  are fully pinned) causes pinv(J_x) to blow up.  eps=1e-3 cleanly separates
  the numerical rank gap (0.85 vs 4e-06) and gives stable results across
  eps in [1e-3, 1e-5].  The main-experiment eps=1e-9 is not applicable here.
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
    eps: float = 1e-3,
) -> dict:
    """
    Compute d_feas(Pred; GT) given:
      edit_probes : dimensional constraints from GT (Q_G)
      rels        : relational constraints from GT (R_G)
    """
    gt_native = compute_native_response(gt_sketch, context, eps=eps)
    pred_native = compute_native_response(pred_sketch, context, eps=eps)

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
        S_gt = edit_response_for_probe(gt_native, probe, context, eps=eps)
        S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps)

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
    eps = 1e-3  # see module docstring for rationale

    artifacts = os.path.join(REPO_ROOT, "artifacts")
    # gt_path = os.path.join(artifacts, "teaser1.FCStd")
    # pred_paths = {
    #     "teaser2": os.path.join(artifacts, "teaser2.FCStd"),
    #     "teaser3": os.path.join(artifacts, "teaser3.FCStd"),
    #     "teaser4": os.path.join(artifacts, "teaser4.FCStd"),
    # }
    gt_path = "/home/ryotaro/Downloads/fiverr/Ryotaro/0/21803_8a36dcda_0011_sketch01.FCStd"
    pred_paths = {
        "solver_rl": "/home/ryotaro/Downloads/fiverr/Vitruvion_2026-05-03_17-08-23_step_56400/0/21803_8a36dcda_0011_sketch01.FCStd",
        "dfeas_rl": "/home/ryotaro/Downloads/fiverr/Vitruvion_dfeas_2026-05-19_11-08-58_step_10300/0/21803_8a36dcda_0011_sketch01.FCStd",
    }

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)

    print(f"Loading GT: {gt_path}")
    gt_sketch = load_sketch_from_fcstd(gt_path, converter)

    probes, _ = collect_probes([gt_sketch], kind="dim")
    rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]

    dim_types: dict[str, int] = {}
    for p in probes:
        dim_types[p.constraint_type.name] = dim_types.get(p.constraint_type.name, 0) + 1
    rel_types: dict[str, int] = {}
    for r in rels:
        rel_types[r.constraint_type.name] = rel_types.get(r.constraint_type.name, 0) + 1

    print(f"GT (teaser1): {len(gt_sketch.entities)} entities, "
          f"{len(gt_sketch.constraints)} constraints "
          f"(dim={len(probes)}, rel={len(rels)})")
    print(f"  Q_G probe types : {dim_types}")
    print(f"  R_G relation types: {rel_types}")
    print(f"  eps = {eps}  (relative to sigma_max≈16, drops modes < ~6e-5)")
    print()

    results = {}
    for name, path in pred_paths.items():
        pred_sketch = load_sketch_from_fcstd(path, converter)
        n_dim = sum(1 for c in pred_sketch.constraints.values() if not _is_relational(c))
        n_rel = sum(1 for c in pred_sketch.constraints.values() if _is_relational(c))
        r = compute_dfeas(gt_sketch, pred_sketch, probes, rels, context, eps=eps)
        r["n_dim"] = n_dim
        r["n_rel"] = n_rel
        results[name] = r
        print(f"{name}: {n_dim} dim + {n_rel} rel constraints")
        print(f"  d_feas       = {r['d_feas']:.6f}")
        print(f"  d_feas_tilde = {r['d_feas_tilde']:.6f}")
        print(f"  gain         = {r['d_feas_gain']:.6f}")
        print(f"  leak         = {r['d_feas_leak']:.6f}")
        print(f"  probes valid = {r['n_valid_probes']}/{r['n_probes']}")
        print()

    sep = "=" * 70
    print(sep)
    print("Summary  (GT = teaser1.FCStd, lower d_feas = more design-intent-equivalent)")
    print(f"{'Pred':<10} {'d_feas':>9} {'d_feas~':>9} {'gain':>9} {'leak':>9} "
          f"{'n_dim':>6} {'n_rel':>6}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<10} {r['d_feas']:>9.4f} {r['d_feas_tilde']:>9.4f} "
              f"{r['d_feas_gain']:>9.4f} {r['d_feas_leak']:>9.4f} "
              f"{r['n_dim']:>6} {r['n_rel']:>6}")
    print(sep)
    print()
    print("Interpretation:")
    print("  gain = ||J_q S_P - I||  (edit not fully achievable via native dims)")
    print("  leak = ||J_{R_G} S_P||  (response violates GT relational constraints)")
    print("  d_feas = avg_q sqrt(gain^2 + leak^2)  per row-balanced residual")


if __name__ == "__main__":
    main()