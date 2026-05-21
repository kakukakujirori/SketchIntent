"""
exp_smatch_metric.py — Naive S_C(q) match metric vs d_feas.

User's question: instead of measuring d_feas = ||J_q S_P - I|| + ||J_{R_G} S_P||
                 why not directly compare S_P^(q) and S_G^(q)?

This script computes three naive direct-match variants on the same
source-conditioned multi-reference setting as d_feas:

  d_smatch_rms^(q)   = ||S_P^(q) - S_G^(q)||_F / sqrt(n_x * k_q)
  d_smatch_rel^(q)   = ||S_P^(q) - S_G^(q)||_F / max(||S_G^(q)||_F, eps)
  d_smatch_sym^(q)   = 2 ||S_P^(q) - S_G^(q)||_F / (||S_P^(q)||_F + ||S_G^(q)||_F + eps)

Each is averaged over probes per (Pred, source GT) pair, then aggregated
across multiple reference GTs by min_i (low-is-good).

We also re-compute d_feas (and its gain/leak components) on the same
rotations so we can pair them directly without alignment issues.

Saves:
  - artifacts/smatch_vs_dfeas_summary.csv      (per rotation, per pred)
  - artifacts/smatch_vs_dfeas_details.csv      (per (pred, gt) pair)
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
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def stack_relation_jacobians(rels, state, context):
    rows = []
    for rel in rels:
        j = constraint_jacobian(rel, state, context)
        if j is None or j.shape[0] == 0:
            continue
        rows.append(j)
    if not rows:
        return None, 0
    cat = torch.cat(rows, dim=0)
    return cat, cat.shape[0]


def compute_pair_metrics(
    *,
    pred_sketch,
    gt_sketch,
    edit_probes,
    edit_weights,
    rels,
    context,
    eps: float,
):
    gt_native = compute_native_response(gt_sketch, context, eps=eps)
    pred_native = compute_native_response(pred_sketch, context, eps=eps)

    # GT-side relation jacobian (used by d_feas leak).
    J_R_pr, n_R_rows = stack_relation_jacobians(rels, pred_native.state, context)

    # Per-probe accumulators.
    d_smatch_rms_total = 0.0
    d_smatch_rel_total = 0.0
    d_smatch_sym_total = 0.0
    d_feas_total = 0.0
    d_feas_gain_total = 0.0
    d_feas_leak_total = 0.0
    weight = 0.0
    skipped = 0

    # Layout match check.
    layout_ok = (
        gt_native.state.num_dof() == pred_native.state.num_dof()
        and gt_native.state.entity_ids == pred_native.state.entity_ids
    )

    for probe, w in zip(edit_probes, edit_weights):
        S_gt = edit_response_for_probe(gt_native, probe, context, eps=eps)
        S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps)
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

        # d_feas: gain + leak (row-balanced)
        Jq_pr = constraint_jacobian(probe, pred_native.state, context)
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


METRIC_NAMES = (
    "d_smatch_rms",
    "d_smatch_rel",
    "d_smatch_sym",
    "d_feas",
)


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

    source_info = {}
    for gt_name in refs:
        gt_sketch = bundle.sketches[gt_name]
        probes, weights, rels, _rel_weights = source_probes(gt_sketch)
        mask = validate_probes_on_source(
            gt_sketch, probes, rels, context, converter, histcad_cfg)
        source_info[gt_name] = {
            "probes": probes, "weights": weights, "rels": rels,
            "valid_mask": mask,
            "n_before": len(probes),
            "n_after": sum(mask) if mask else 0,
        }

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    def add_pred_rows(pred_name, pred_sketch, drop_rate, sample_id):
        for validated in (True,):  # only validated, like main d_feas table
            per_source = []
            for gt_name in refs:
                info = source_info[gt_name]
                if validated:
                    probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
                    weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
                else:
                    probes, weights = info["probes"], info["weights"]
                if not probes:
                    per_source.append(None)
                    continue
                res = compute_pair_metrics(
                    pred_sketch=pred_sketch,
                    gt_sketch=bundle.sketches[gt_name],
                    edit_probes=probes,
                    edit_weights=weights,
                    rels=info["rels"],
                    context=context,
                    eps=eps,
                )
                if res is None:
                    per_source.append(None)
                    continue
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
                    **res,
                    "error": "",
                }
                detail_rows.append(row)
                per_source.append(row)

            valid = [r for r in per_source if r is not None]
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
                "valid_sources": len(valid),
                "error": "",
            }
            for m in METRIC_NAMES:
                vals = [finite_or_none(r.get(m)) for r in valid]
                vals = [v for v in vals if v is not None]
                srow[f"min_{m}"] = min(vals) if vals else None
            summary_rows.append(srow)

    for drop_rate in drop_rates:
        sample_count = 1 if drop_rate in (0.0, 100.0) else samples_per_rate
        for sample_id in range(sample_count):
            seed = stable_seed(bundle.batch, bundle.filename, rotation_id,
                               drop_rate, sample_id)
            pred_var, _no, _nk, _nd = dropout_variant(
                heldout_sketch, drop_rate=drop_rate, seed=seed)
            add_pred_rows(
                pred_name=f"{heldout}_drop{drop_rate:g}",
                pred_sketch=pred_var,
                drop_rate=drop_rate,
                sample_id=sample_id,
            )

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


def write_csv(path: str, rows: list[dict]):
    full = repo_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(full, "w", newline="") as f:
        if not keys:
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {full}")


def wilcoxon_z(diffs):
    """Wilcoxon signed-rank test on paired differences; normal approx z, two-sided p."""
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0, 0
    abs_d = sorted(((abs(d), 1 if d > 0 else -1) for d in nonzero), key=lambda x: x[0])
    # Assign ranks (with average ranks for ties on |d|).
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_d[j + 1][0] == abs_d[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    W_plus = sum(r for r, (_, s) in zip(ranks, abs_d) if s > 0)
    W_minus = sum(r for r, (_, s) in zip(ranks, abs_d) if s < 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 0.0, 1.0, sum(1 for _, s in abs_d if s > 0), sum(1 for _, s in abs_d if s < 0)
    z = (W_plus - mu) / sigma
    # two-sided p via erf approximation
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p, sum(1 for _, s in abs_d if s > 0), sum(1 for _, s in abs_d if s < 0)


def paired_analysis(summary_rows, pred_label: str = "Vitruvion"):
    """`pred_label` model vs heldout drop 0% paired comparison, per metric.

    Convention: diff = pred(model) - pred(heldout-drop0).
    For low-is-good metrics, positive diff => human-better.
    Returns dict[metric] -> (human, vit, tie, z, p).
    """
    by_rot: dict[str, dict] = {}
    for r in summary_rows:
        if r.get("error") or not r.get("source_validated"):
            continue
        rotid = r["rotation_id"]
        d = by_rot.setdefault(rotid, {})
        if float(r["drop_rate"]) == -1.0 and r.get("pred") == pred_label:
            d["vit"] = r
        elif float(r["drop_rate"]) == 0.0:
            d["h0"] = r

    out = {}
    for m in METRIC_NAMES:
        diffs = []
        h_better = v_better = tie = 0
        for rotid, d in by_rot.items():
            if "vit" not in d or "h0" not in d:
                continue
            v = finite_or_none(d["vit"].get(f"min_{m}"))
            h = finite_or_none(d["h0"].get(f"min_{m}"))
            if v is None or h is None:
                continue
            diff = v - h
            if abs(diff) <= 1e-9:
                tie += 1
            elif diff > 0:
                h_better += 1
            else:
                v_better += 1
            diffs.append(diff)
        z, p, _, _ = wilcoxon_z(diffs)
        out[m] = (h_better, v_better, tie, z, p)
    return out


def trajectory_by_dropout(summary_rows):
    """Per-metric median across drop rates."""
    rates = sorted({float(r["drop_rate"]) for r in summary_rows
                   if not r.get("error") and r.get("source_validated")
                   and float(r["drop_rate"]) >= 0})
    out = {}
    for rate in rates:
        rs = [r for r in summary_rows
              if not r.get("error") and r.get("source_validated")
              and float(r["drop_rate"]) == rate]
        out[rate] = {m: safe_median([finite_or_none(r.get(f"min_{m}")) for r in rs])
                     for m in METRIC_NAMES}
        out[rate]["n"] = len(rs)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vitruvion-root",
        default=None,
        help="Legacy single-model root. Prefer --vitruvion-spec.")
    ap.add_argument(
        "--vitruvion-spec",
        action="append",
        default=None,
        help="label:path, repeatable. Defaults to base + solver_rl.")
    ap.add_argument("--batches", nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--human", action="append", default=None)
    ap.add_argument("--ref-size", type=int, default=2)
    ap.add_argument("--drop-rates", default="0,25,50,75,100")
    ap.add_argument("--samples-per-rate", type=int, default=1)
    ap.add_argument("--base-policy", default="canonical_solid",
                    choices=("none", "canonical_solid", "canonical_all"))
    ap.add_argument("--geometry-tol", type=float, default=5e-1)
    ap.add_argument("--canonical-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--delta-basis", default="bbox_diag")
    ap.add_argument("--delta-scale", type=float, default=0.01)
    ap.add_argument("--delta-absolute", type=float, default=1.0)
    ap.add_argument("--delta-signs", default="-1,+1")
    ap.add_argument("--newton-max-iter", type=int, default=80)
    ap.add_argument("--newton-tol", type=float, default=1e-6)
    ap.add_argument("--er-target-tol", type=float, default=1e-3)
    ap.add_argument("--relation-tol", type=float, default=1e-3)
    ap.add_argument("--output-csv", default="artifacts/smatch_vs_dfeas_summary.csv")
    ap.add_argument("--details-csv", default="artifacts/smatch_vs_dfeas_details.csv")
    return ap.parse_args()


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

        for filename in common:
            if args.limit is not None and n_files >= args.limit:
                break
            n_files += 1
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
                    )
                    summary_rows.extend(srows)
                    detail_rows.extend(drows)
                    n_rot += 1
            print(f"  [done] {batch}/{filename}: {n_rot} rotations")

        if args.limit is not None and n_files >= args.limit:
            break

    write_csv(args.output_csv, summary_rows)
    write_csv(args.details_csv, detail_rows)

    print("\n=== Trajectory by dropout (validated, multi-ref min) ===")
    traj = trajectory_by_dropout(summary_rows)
    header = "  drop   n   " + "  ".join(f"{m:14s}" for m in METRIC_NAMES)
    print(header)
    for rate in sorted(traj.keys()):
        vals = traj[rate]
        row = f"  {rate:5.1f} {vals['n']:4d}  " + "  ".join(
            f"{(vals[m] if vals[m] is not None else float('nan')):14.4f}"
            for m in METRIC_NAMES)
        print(row)

    model_labels = sorted({r.get("pred") for r in summary_rows
                           if float(r.get("drop_rate", 0)) == -1.0 and r.get("pred")})
    if not model_labels:
        model_labels = ["Vitruvion"]
    for lbl in model_labels:
        print(f"\n=== {lbl} vs heldout drop 0% paired (validated) ===")
        print(f"  {'metric':14s}  human  vit   tie       z         p")
        res = paired_analysis(summary_rows, pred_label=lbl)
        for m in METRIC_NAMES:
            h, v, t, z, p = res[m]
            print(f"  {m:14s}   {h:4d}  {v:4d}  {t:4d}  {z:7.3f}  {p:.2e}")


if __name__ == "__main__":
    main()
