"""test_dH.py — compute d_feas, d_H, and d_SVD for the single-sketch case
used in ``notebooks/test_dfeas.py``.

GT:   Ryotaro's annotation of 21803
Pred:
  - solver_rl   (Vitruvion + solver-feedback RL)
  - dfeas_rl    (Vitruvion + d_feas-reward RL)
  - heldout_Sagar  (intent-equivalent human reference → d_* should be ≈ 0)

The motivating question for d_H is whether it correctly ranks
``solver_rl < dfeas_rl`` (matching qualitative judgement) on this sketch,
even though ``d_feas`` prefers ``dfeas_rl``.

d_SVD is an additional candidate: instead of decomposing ``S_C^(Q_G)`` into
gain / leak manually, we extract its principal components via SVD and
compare them between Pred and GT.  Formulation:

    M_C       := J_Q R_C                        (stack GT probes)
    S_C^(Q_G) := R_C pinv(M_C)                  (n_x × |Q_G|)
    T_C       := J_phi @ S_C^(Q_G)              (m_phi × |Q_G|)
    T_C = U_C diag(sigma_C) V_C^T               (thin SVD)

    d_SVD_resp  = ||T_P - T_G||_F / sqrt(m_phi · k_Q)     (raw d_resp baseline)
    d_SVD_sigma = ||sigma_P - sigma_G||_2                  (spectrum)
    d_SVD_sub   = subspace dist of top-k left singular vecs
    d_SVD_top1  = sqrt(2 - 2|<u1_P, u1_G>|)                (leading PC only)

The smoke test is: on the heldout pair (Sagar vs Ryotaro), does d_SVD
return ≈ 0?  If yes, the parametrisation-artefact concern that killed
v3 d_resp is absorbed.  If no, the SVD truncation does not save d_resp.
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
V4_EXP_DIR = os.path.join(NOTEBOOKS_DIR, "v4_exp")
if V4_EXP_DIR not in sys.path:
    sys.path.insert(0, V4_EXP_DIR)

import torch

from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.metrics.edit_response_distance import (
    collect_probes,
    compute_native_response,
    constraint_jacobian,
    pairwise_distance_observable,
    _is_relational,
)
from canonical_sketch import load_sketch_from_fcstd
from exp_constrained_response_metric import compute_metrics
from metrics_v4 import V4Hyperparams, compute_metrics_v4


def compute_d_svd(
    *,
    pred_sketch,
    gt_sketch,
    edit_probes,
    base_eids,
    context,
    eps: float,
    tau: float = 0.01,
) -> dict | None:
    """SVD-based comparison of T_C = J_phi @ R_C @ pinv(J_Q @ R_C).

    Only probes for which BOTH Pred and GT can compute a Jacobian are
    retained (otherwise the column counts of T_P / T_G differ and the
    comparison is ill-defined).

    ``tau`` is the relative threshold for truncated subspace comparison:
    components with sigma < tau * max(sigma_P[0], sigma_G[0]) are excluded
    when computing ``d_SVD_sub_trunc``.  Without truncation the principal
    angles include numerically-rank-deficient directions whose left
    singular vectors are essentially arbitrary, producing spurious
    annotator-variance signal.
    """
    pred_native = compute_native_response(pred_sketch, context, eps=eps)
    gt_native = compute_native_response(gt_sketch, context, eps=eps)

    J_phi_P = pairwise_distance_observable(pred_sketch, base_eids=base_eids)
    J_phi_G = pairwise_distance_observable(gt_sketch, base_eids=base_eids)
    if J_phi_P.shape[0] != J_phi_G.shape[0] or J_phi_P.shape[0] == 0:
        return None

    J_Q_P_rows, J_Q_G_rows = [], []
    for probe in edit_probes:
        Jq_P = constraint_jacobian(probe, pred_native.state, context)
        Jq_G = constraint_jacobian(probe, gt_native.state, context)
        if Jq_P is None or Jq_G is None:
            continue
        # Use only first row to keep scalar-probe semantics; multi-row
        # dimensional residuals are unusual for our probe set.
        J_Q_P_rows.append(Jq_P[:1])
        J_Q_G_rows.append(Jq_G[:1])
    if not J_Q_P_rows:
        return None

    J_Q_P = torch.cat(J_Q_P_rows, dim=0)
    J_Q_G = torch.cat(J_Q_G_rows, dim=0)

    R_P = pred_native.R
    R_G = gt_native.R
    if R_P.shape[1] == 0 or R_G.shape[1] == 0:
        return None

    M_P = J_Q_P @ R_P
    M_G = J_Q_G @ R_G

    S_P = R_P @ torch.linalg.pinv(M_P)
    S_G = R_G @ torch.linalg.pinv(M_G)

    T_P = J_phi_P @ S_P
    T_G = J_phi_G @ S_G

    m_phi, k_Q = T_P.shape
    n_entries = max(m_phi * k_Q, 1)
    d_resp = float(torch.linalg.norm(T_P - T_G)) / math.sqrt(n_entries)

    U_P, sig_P, _Vh_P = torch.linalg.svd(T_P, full_matrices=False)
    U_G, sig_G, _Vh_G = torch.linalg.svd(T_G, full_matrices=False)

    d_sigma = float(torch.linalg.norm(sig_P - sig_G))
    sig_G_norm = float(torch.linalg.norm(sig_G)) + 1e-12
    d_sigma_rel = d_sigma / sig_G_norm

    # Subspace distance via principal angles (all k_Q components).
    M_align = U_G.T @ U_P
    svals = torch.linalg.svdvals(M_align).clamp(0.0, 1.0)
    cos2 = float((svals ** 2).sum())
    d_sub = math.sqrt(max(2.0 * k_Q - 2.0 * cos2, 0.0) / max(k_Q, 1))

    # Truncated subspace distance: keep components with sigma above
    # tau * max(sigma_P[0], sigma_G[0]) on each side.  Components below this
    # cutoff have unstable left singular vectors (essentially arbitrary
    # null-space directions) and would otherwise dominate the principal-angle
    # sum with spurious annotator-variance signal.
    #
    # Rank mismatch handling mirrors d_sub in metrics_v4.py:
    #   d_sub_trunc^2 = (k_eff_P + k_eff_G - 2 * sum cos^2 theta_i) / max(k_eff)
    # so that "one side has signal, the other collapses to 0" returns ~1
    # (max), not 0 (degenerate).
    sig_max = max(float(sig_P[0]), float(sig_G[0]), 1e-30)
    thresh = tau * sig_max
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
        # Both observable responses collapsed — undefined, conventionally 0.
        d_sub_trunc = 0.0
    else:
        # One side collapsed: max subspace distance.
        d_sub_trunc = math.sqrt((k_eff_P + k_eff_G) / k_norm)

    # Top-1 leading-direction comparison (sign-invariant).
    cos1 = abs(float(U_P[:, 0] @ U_G[:, 0]))
    d_top1 = math.sqrt(max(2.0 - 2.0 * cos1, 0.0))

    return {
        "d_SVD_resp": d_resp,
        "d_SVD_sigma": d_sigma,
        "d_SVD_sigma_rel": d_sigma_rel,
        "d_SVD_sub": d_sub,
        "d_SVD_sub_trunc": d_sub_trunc,
        "d_SVD_top1": d_top1,
        "sigma_P": sig_P.detach().cpu().tolist(),
        "sigma_G": sig_G.detach().cpu().tolist(),
        "k_Q_used": k_Q,
        "k_eff_P": k_eff_P,
        "k_eff_G": k_eff_G,
        "m_phi": m_phi,
        "tau": tau,
    }


def evaluate_pair(
    *,
    gt_sketch,
    pred_sketch,
    edit_probes,
    edit_weights,
    rels,
    base_eids,
    context,
    hp: V4Hyperparams,
) -> dict:
    res_v3 = compute_metrics(
        pred_sketch=pred_sketch,
        gt_sketch=gt_sketch,
        edit_probes=edit_probes,
        edit_weights=edit_weights,
        rels=rels,
        context=context,
        base_eids=base_eids,
        eps=hp.eps,
    )
    res_v4 = compute_metrics_v4(
        pred_sketch=pred_sketch,
        gt_sketch=gt_sketch,
        edit_probes=edit_probes,
        edit_weights=edit_weights,
        rels=rels,
        context=context,
        base_eids=base_eids,
        hp=hp,
    )
    res_svd = compute_d_svd(
        pred_sketch=pred_sketch,
        gt_sketch=gt_sketch,
        edit_probes=edit_probes,
        base_eids=base_eids,
        context=context,
        eps=hp.eps,
    ) or {}
    return {**res_v3, **res_v4, **res_svd}


def main():
    hp = V4Hyperparams(eps=1e-9)

    gt_path = "/home/sturk/Downloads/fiverr/Ryotaro/0/21803_8a36dcda_0011_sketch01.FCStd"
    pred_paths = {
        # intent-equivalent human reference → expect d_* ≈ 0  (smoke test)
        "heldout_Sagar":
            "/home/sturk/Downloads/fiverr/Sagar/0/21803_8a36dcda_0011_sketch01.FCStd",
        "solver_rl":
            "/home/sturk/Downloads/fiverr/Vitruvion_2026-05-03_17-08-23_step_56400/0/21803_8a36dcda_0011_sketch01.FCStd",
        "dfeas_rl":
            "/home/sturk/Downloads/fiverr/Vitruvion_dfeas_2026-05-19_11-08-58_step_10300/0/21803_8a36dcda_0011_sketch01.FCStd",
    }

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)

    print(f"Loading GT: {gt_path}")
    gt_sketch = load_sketch_from_fcstd(gt_path, converter)

    probes, weights = collect_probes([gt_sketch], kind="dim")
    rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]

    pred_sketches = {name: load_sketch_from_fcstd(path, converter)
                     for name, path in pred_paths.items()}

    gt_eids_set = set(gt_sketch.entities.keys())
    common = gt_eids_set.intersection(
        *(set(s.entities.keys()) for s in pred_sketches.values()))
    base_eids = [eid for eid in gt_sketch.entities.keys() if eid in common]

    dim_types: dict[str, int] = {}
    for p in probes:
        dim_types[p.constraint_type.name] = dim_types.get(p.constraint_type.name, 0) + 1
    rel_types: dict[str, int] = {}
    for r in rels:
        rel_types[r.constraint_type.name] = rel_types.get(r.constraint_type.name, 0) + 1

    print(f"GT: {len(gt_sketch.entities)} entities, "
          f"{len(gt_sketch.constraints)} constraints "
          f"(dim={len(probes)}, rel={len(rels)})")
    print(f"  Q_G probe types : {dim_types}")
    print(f"  R_G relation types: {rel_types}")
    print(f"  base_eids       : {len(base_eids)} entities (intersection across "
          f"GT and {len(pred_paths)} preds)")
    if len(base_eids) < len(gt_sketch.entities):
        dropped = sorted(gt_eids_set - common)
        print(f"  GT-only eids    : {dropped} (excluded from J_phi)")
    print(f"  eps = {hp.eps}")
    print()

    results = {}
    for name, pred_sketch in pred_sketches.items():
        n_dim = sum(1 for c in pred_sketch.constraints.values()
                    if not _is_relational(c))
        n_rel = sum(1 for c in pred_sketch.constraints.values()
                    if _is_relational(c))
        r = evaluate_pair(
            gt_sketch=gt_sketch,
            pred_sketch=pred_sketch,
            edit_probes=probes,
            edit_weights=weights,
            rels=rels,
            base_eids=base_eids,
            context=context,
            hp=hp,
        )
        r["n_dim"] = n_dim
        r["n_rel"] = n_rel
        r["n_entities"] = len(pred_sketch.entities)
        results[name] = r

        print(f"{name}: {len(pred_sketch.entities)} entities, "
              f"{n_dim} dim + {n_rel} rel constraints")
        print(f"  d_feas      = {r['d_feas']:.6f}   "
              f"(gain={r['d_feas_gain']:.4f}, leak={r['d_feas_leak']:.4f})")
        print(f"  d_H         = {r['d_H']:.6f}   "
              f"(d_sub={r['d_sub']:.4f}, leak'={r['leak_prime']:.4f})")
        print(f"  d_SVD       resp={r.get('d_SVD_resp', float('nan')):.4f}  "
              f"sigma={r.get('d_SVD_sigma', float('nan')):.4f}  "
              f"sigma_rel={r.get('d_SVD_sigma_rel', float('nan')):.4f}  "
              f"sub={r.get('d_SVD_sub', float('nan')):.4f}  "
              f"sub_trunc={r.get('d_SVD_sub_trunc', float('nan')):.4f}  "
              f"top1={r.get('d_SVD_top1', float('nan')):.4f}  "
              f"k_eff={r.get('k_eff_P', 0)}/{r.get('k_eff_G', 0)} of {r.get('k_Q_used', 0)}")
        sig_P = r.get('sigma_P', [])
        sig_G = r.get('sigma_G', [])
        print(f"  sigma_P     = {[f'{s:.3f}' for s in sig_P]}")
        print(f"  sigma_G     = {[f'{s:.3f}' for s in sig_G]}")
        print(f"  dim_P={r['dim_P']}  dim_G={r['dim_G']}  "
              f"align_loss={r['align_loss']:.4f}  rank_loss={r['rank_loss']:.4f}")
        print(f"  n_p_pred={r['n_p_pred']}  n_p_gt={r['n_p_gt']}  "
              f"n_R_rows={r['n_R_rows']}  k_Q={r.get('k_Q_used', 0)}")
        print()

    sep = "=" * 120
    print(sep)
    print("Summary  (GT = Ryotaro/21803, lower = closer to GT design intent)")
    print(f"{'Pred':<16} {'d_feas':>8} {'d_H':>8} {'d_sub':>8} {'leak^':>8} "
          f"{'SVD_resp':>9} {'SVD_sig':>9} {'SVD_sigR':>9} {'SVD_sub':>9} "
          f"{'SVD_subT':>9} {'SVD_top1':>9}")
    print("-" * 120)
    for name, r in results.items():
        print(f"{name:<16} {r['d_feas']:>8.4f} {r['d_H']:>8.4f} "
              f"{r['d_sub']:>8.4f} {r['leak_prime']:>8.4f} "
              f"{r.get('d_SVD_resp', float('nan')):>9.4f} "
              f"{r.get('d_SVD_sigma', float('nan')):>9.4f} "
              f"{r.get('d_SVD_sigma_rel', float('nan')):>9.4f} "
              f"{r.get('d_SVD_sub', float('nan')):>9.4f} "
              f"{r.get('d_SVD_sub_trunc', float('nan')):>9.4f} "
              f"{r.get('d_SVD_top1', float('nan')):>9.4f}")
    print(sep)
    print()

    print("Rankings (best → worst):")
    for metric in ("d_feas", "d_H", "d_SVD_resp", "d_SVD_sigma",
                   "d_SVD_sigma_rel", "d_SVD_sub", "d_SVD_sub_trunc",
                   "d_SVD_top1"):
        try:
            order = sorted(results.keys(),
                           key=lambda k: results[k].get(metric, float('nan')))
            print(f"  {metric:<16}: {' < '.join(order)}")
        except Exception:
            pass
    print()
    print("Smoke-test criterion: heldout_Sagar should yield d_* ≈ 0 on a metric")
    print("that absorbs annotator variance.  Compare its row above against the")
    print("two RL preds — if d_SVD_* on heldout_Sagar is comparable in magnitude")
    print("to solver_rl/dfeas_rl, the SVD-based metric inherits d_resp's")
    print("parametrisation-artefact failure mode.")


if __name__ == "__main__":
    main()
