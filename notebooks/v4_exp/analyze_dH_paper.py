"""analyze_dH_paper.py — recompute v3 paper sections (§6.1, §6.2, §7, §8, §9)
with d_H as the main metric, plus the R3 subset cohort filter (n_dim_G <= 3).

Reads:
- artifacts/v4_exp/v4_controls_scores_full.csv (v4 metrics including d_H, leak_prime, d_sub)
- artifacts/source_conditioned_dropout_scores_full.csv (HistCAD OES per (rotation, pred))

Emits to artifacts/v4_exp/:
- dH_paired_paper.csv       : §6.1 paired Wilcoxon for d_feas / d_H / d_sub / leak_prime
                              (base, solver_rl, vs heldout); also gain'/leak' breakdown.
- dH_dropout_sweep.csv      : §6.2 median d_H, d_sub (=gain'), leak_prime per drop rate.
- dH_model_ranking.csv      : §7 per-model n, median d_H / d_sub / leak_prime.
- dH_cdf_long.csv           : §8 long-form (variant, rotation_id, d_H, d_sub, leak_prime)
                              for downstream CDF plotting.
- dH_oes_divergent_cases.csv: §9 top high-d_H/low-OES and low-d_H/high-OES per model.
- dH_subset_cohort.csv      : R3 paired Wilcoxon restricted to rotations with
                              n_dim_G <= 3 (fully-constrained subspace cohort).
- dH_paper_summary.txt      : one-page console-style report.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import median

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _finite(v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _wilcoxon(diffs, tie=1e-9):
    pairs = [d for d in diffs if d is not None and abs(d) > tie]
    n = len(pairs)
    if n == 0:
        return 0.0, 1.0, 0
    abs_vals = sorted(((abs(d), i) for i, d in enumerate(pairs)), key=lambda t: t[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_vals[j + 1][0] == abs_vals[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            _, orig_idx = abs_vals[k]
            ranks[orig_idx] = avg_rank
        i = j + 1
    W_plus = sum(r for d, r in zip(pairs, ranks) if d > 0)
    mean_W = n * (n + 1) / 4.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0
    if var_W == 0:
        return 0.0, 1.0, n
    z = (W_plus - mean_W) / math.sqrt(var_W)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p, n


def load_v4_rows(path):
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    rows = []
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            rows.append(r)
    return rows


def load_oes_rows(path):
    """Load source_conditioned_dropout_scores rows (model rows only, validated)."""
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        return []
    rows = []
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source_validated") not in ("True", "true", "1"):
                continue
            if r.get("drop_rate") in ("-1", "-1.0"):
                rows.append(r)
    return rows


def paired_wilcoxon_table(v4_rows, models, metrics):
    """Compute paired Wilcoxon (model vs heldout) per (model, metric).

    Returns list[dict].  Diff = model - heldout (lower-is-better -> z>0 = human better).
    """
    by_rot = defaultdict(dict)
    for r in v4_rows:
        by_rot[r["rotation_id"]][r["variant"]] = r

    out = []
    for model in models:
        for m in metrics:
            diffs = []
            n_held = 0
            for rid, d in by_rot.items():
                if "heldout" not in d or model not in d:
                    continue
                vh = _finite(d["heldout"].get(f"min_{m}"))
                vm = _finite(d[model].get(f"min_{m}"))
                if vh is None or vm is None:
                    continue
                diffs.append(vm - vh)
                n_held += 1
            if not diffs:
                continue
            h_better = sum(1 for x in diffs if x > 1e-9)
            m_better = sum(1 for x in diffs if x < -1e-9)
            tie = len(diffs) - h_better - m_better
            z, p, _ = _wilcoxon(diffs)
            out.append({
                "model": model,
                "metric": m,
                "n_paired": len(diffs),
                "human_better": h_better,
                "model_better": m_better,
                "tie": tie,
                "z": z,
                "p_two_sided": p,
                "median_diff": _quantile(diffs, 50),
            })
    return out


def dropout_sweep_table(v4_rows, metrics):
    """Median per variant for the dropout sweep variants + base/solver_rl/heldout/source_self."""
    by_variant = defaultdict(list)
    for r in v4_rows:
        by_variant[r["variant"]].append(r)
    canonical_order = ["source_self", "heldout",
                       "dim_drop25", "dim_drop50", "dim_drop75", "dim_drop100",
                       "rel_drop50", "rel_drop100",
                       "base", "solver_rl"]
    rows = []
    for v in canonical_order:
        if v not in by_variant:
            continue
        vrows = by_variant[v]
        out = {"variant": v, "n": len(vrows)}
        for m in metrics:
            vals = [_finite(r.get(f"min_{m}")) for r in vrows]
            vals = [x for x in vals if x is not None]
            out[f"med_{m}"] = _quantile(vals, 50) if vals else None
            out[f"q90_{m}"] = _quantile(vals, 90) if vals else None
        rows.append(out)
    return rows


def model_ranking_table(v4_rows):
    by_variant = defaultdict(list)
    for r in v4_rows:
        by_variant[r["variant"]].append(r)
    rows = []
    for v in ("heldout", "base", "solver_rl"):
        vrows = by_variant.get(v, [])
        n_files = len({r["filename"] for r in vrows})
        n_rot = len(vrows)
        out = {"variant": v, "n_files": n_files, "n_rotations": n_rot}
        for m in ("d_feas", "d_H", "d_sub", "leak_prime",
                  "d_feas_gain", "d_feas_leak"):
            vals = [_finite(r.get(f"min_{m}")) for r in vrows]
            vals = [x for x in vals if x is not None]
            out[f"med_{m}"] = _quantile(vals, 50) if vals else None
            out[f"q90_{m}"] = _quantile(vals, 90) if vals else None
        rows.append(out)
    return rows


def cdf_long_rows(v4_rows, variants, metrics):
    out = []
    for r in v4_rows:
        v = r.get("variant", "")
        if v not in variants:
            continue
        row = {
            "variant": v,
            "rotation_id": r["rotation_id"],
            "filename": r["filename"],
            "batch": r["batch"],
        }
        for m in metrics:
            row[m] = _finite(r.get(f"min_{m}"))
        out.append(row)
    return out


def divergent_cases(v4_rows, oes_rows, model, k=12,
                    high_oes_thresh=0.5, high_dH_thresh=0.5,
                    low_oes_thresh=0.05, low_dH_thresh=0.05):
    """Join v4 (variant=model) with OES (pred=model) on rotation_id.

    Returns (high_oes_high_dH, low_oes_low_dH).
    """
    v4_by_rid = {r["rotation_id"]: r for r in v4_rows
                 if r["variant"] == model}
    oes_by_rid = {r["rotation_id"]: r for r in oes_rows
                  if r.get("pred") == model}

    joined = []
    for rid, v in v4_by_rid.items():
        if rid not in oes_by_rid:
            continue
        o = oes_by_rid[rid]
        d_H = _finite(v.get("min_d_H"))
        d_sub = _finite(v.get("min_d_sub"))
        leak_prime = _finite(v.get("min_leak_prime"))
        d_feas = _finite(v.get("min_d_feas"))
        oes = _finite(o.get("max_histcad_oes"))
        if d_H is None or oes is None:
            continue
        joined.append({
            "rotation_id": rid,
            "filename": v["filename"],
            "batch": v["batch"],
            "heldout": o.get("heldout", ""),
            "pred": model,
            "histcad_oes": oes,
            "d_feas": d_feas,
            "d_H": d_H,
            "d_sub": d_sub,
            "leak_prime": leak_prime,
        })

    high = [r for r in joined
            if r["histcad_oes"] >= high_oes_thresh and r["d_H"] >= high_dH_thresh]
    high.sort(key=lambda x: (-x["d_H"], -x["histcad_oes"]))
    low = [r for r in joined
           if r["histcad_oes"] <= low_oes_thresh and r["d_H"] <= low_dH_thresh]
    low.sort(key=lambda x: (x["d_H"], x["histcad_oes"]))
    return high[:k], low[:k]


def subset_cohort_paired(v4_rows, models, metrics, max_n_dim_G=3):
    """R3: restrict heldout rotations to those with n_dim_G <= max_n_dim_G.

    Use heldout row's min_dim_G as the cohort key.  Pair model row to heldout
    only when the cohort criterion is met on the heldout side.
    """
    by_rot = defaultdict(dict)
    for r in v4_rows:
        by_rot[r["rotation_id"]][r["variant"]] = r

    out = []
    for model in models:
        for m in metrics:
            diffs = []
            for rid, d in by_rot.items():
                if "heldout" not in d or model not in d:
                    continue
                n_dim_G = _finite(d["heldout"].get("min_dim_G"))
                if n_dim_G is None or n_dim_G > max_n_dim_G:
                    continue
                vh = _finite(d["heldout"].get(f"min_{m}"))
                vm = _finite(d[model].get(f"min_{m}"))
                if vh is None or vm is None:
                    continue
                diffs.append(vm - vh)
            if not diffs:
                continue
            h_better = sum(1 for x in diffs if x > 1e-9)
            m_better = sum(1 for x in diffs if x < -1e-9)
            tie = len(diffs) - h_better - m_better
            z, p, _ = _wilcoxon(diffs)
            out.append({
                "model": model,
                "metric": m,
                "max_n_dim_G": max_n_dim_G,
                "n_paired": len(diffs),
                "human_better": h_better,
                "model_better": m_better,
                "tie": tie,
                "z": z,
                "p_two_sided": p,
                "median_diff": _quantile(diffs, 50),
            })
    return out


def write_csv(path, rows):
    if not rows:
        print(f"  (skip {os.path.relpath(path, REPO_ROOT)}: empty)")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved: {os.path.relpath(path, REPO_ROOT)}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--v4-csv", default="artifacts/v4_exp/v4_controls_scores_full.csv")
    p.add_argument("--oes-csv",
                   default="artifacts/source_conditioned_dropout_scores_full.csv")
    p.add_argument("--out-dir", default="artifacts/v4_exp")
    p.add_argument("--models", nargs="+", default=["base", "solver_rl"])
    p.add_argument("--cohort-max-dim-g", type=int, default=3,
                   help="R3 subset cohort: heldout rotations with n_dim_G <= this")
    p.add_argument("--divergent-topk", type=int, default=12)
    args = p.parse_args()

    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    v4_rows = load_v4_rows(args.v4_csv)
    oes_rows = load_oes_rows(args.oes_csv)
    print(f"Loaded {len(v4_rows)} v4 rows, {len(oes_rows)} validated OES model rows")

    metrics_main = ["d_feas", "d_H", "d_sub", "leak_prime"]
    metrics_full = metrics_main + ["d_feas_gain", "d_feas_leak"]

    # ----- §6.1 paired -----
    paired = paired_wilcoxon_table(v4_rows, args.models, metrics_full)
    write_csv(os.path.join(out_dir, "dH_paired_paper.csv"), paired)

    # ----- §6.2 dropout sweep with d_H / d_sub / leak_prime -----
    dropout = dropout_sweep_table(v4_rows,
                                  metrics=["d_feas", "d_H", "d_sub", "leak_prime",
                                           "d_feas_gain", "d_feas_leak"])
    write_csv(os.path.join(out_dir, "dH_dropout_sweep.csv"), dropout)

    # ----- §7 model ranking with d_H -----
    ranking = model_ranking_table(v4_rows)
    write_csv(os.path.join(out_dir, "dH_model_ranking.csv"), ranking)

    # ----- §8 CDF long-form (per-rotation values) -----
    cdf_variants = {"source_self", "heldout",
                    "dim_drop25", "dim_drop50", "dim_drop75", "dim_drop100",
                    "rel_drop50", "rel_drop100",
                    "base", "solver_rl"}
    cdf_rows = cdf_long_rows(v4_rows, cdf_variants,
                             metrics=["d_feas", "d_H", "d_sub", "leak_prime"])
    write_csv(os.path.join(out_dir, "dH_cdf_long.csv"), cdf_rows)

    # ----- §9 divergent cases per model -----
    div_all = []
    for model in args.models:
        high, low = divergent_cases(v4_rows, oes_rows, model,
                                    k=args.divergent_topk)
        for r in high:
            r2 = dict(r); r2["category"] = "high_OES_high_dH"
            div_all.append(r2)
        for r in low:
            r2 = dict(r); r2["category"] = "low_OES_low_dH"
            div_all.append(r2)
    write_csv(os.path.join(out_dir, "dH_oes_divergent_cases.csv"), div_all)

    # ----- R3 subset cohort -----
    subset = subset_cohort_paired(v4_rows, args.models, metrics_full,
                                  max_n_dim_G=args.cohort_max_dim_g)
    write_csv(os.path.join(out_dir, "dH_subset_cohort.csv"), subset)

    # ----- console summary -----
    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("=" * 78)
    emit(f"d_H paper-style analysis (input: {os.path.relpath(args.v4_csv, REPO_ROOT)})")
    emit("=" * 78)

    emit("\n[§6.1] Paired Wilcoxon (model vs heldout, drop=0); lower-is-better; z>0=human better")
    for model in args.models:
        emit(f"  -- model = {model} --")
        for r in paired:
            if r["model"] != model:
                continue
            emit(f"    {r['metric']:14s}  n={r['n_paired']:>4d}"
                 f"  h={r['human_better']:>4d} m={r['model_better']:>4d}"
                 f"  t={r['tie']:>4d}  z={r['z']:+7.2f}  p={r['p_two_sided']:.2e}")

    emit("\n[§6.2] Dropout sweep median (gain'=d_sub, leak'=leak_prime)")
    emit(f"  {'variant':14s} {'n':>4s} {'d_feas':>9s} {'d_H':>9s} "
         f"{'d_sub':>9s} {'leak_p':>9s} {'gain':>9s} {'leak':>9s}")
    for r in dropout:
        def fmt(x):
            return f"{x:>9.4f}" if x is not None else f"{'NA':>9s}"
        emit(f"  {r['variant']:14s} {r['n']:>4d} "
             f"{fmt(r['med_d_feas'])} {fmt(r['med_d_H'])} "
             f"{fmt(r['med_d_sub'])} {fmt(r['med_leak_prime'])} "
             f"{fmt(r['med_d_feas_gain'])} {fmt(r['med_d_feas_leak'])}")

    emit("\n[§7] Model ranking with d_H")
    emit(f"  {'variant':14s} {'n_files':>7s} {'n_rot':>5s} "
         f"{'med_d_feas':>10s} {'med_d_H':>10s} {'med_d_sub':>10s} {'med_leak_p':>10s} "
         f"{'Q90_d_H':>10s}")
    for r in ranking:
        def fmt(x):
            return f"{x:>10.4f}" if x is not None else f"{'NA':>10s}"
        emit(f"  {r['variant']:14s} {r['n_files']:>7d} {r['n_rotations']:>5d} "
             f"{fmt(r['med_d_feas'])} {fmt(r['med_d_H'])} "
             f"{fmt(r['med_d_sub'])} {fmt(r['med_leak_prime'])} "
             f"{fmt(r['q90_d_H'])}")

    emit("\n[§9] Divergent rotations (top high-OES high-d_H per model)")
    for model in args.models:
        emit(f"  -- model = {model} --")
        head = [r for r in div_all
                if r["pred"] == model and r["category"] == "high_OES_high_dH"][:5]
        for r in head:
            emit(f"    OES={r['histcad_oes']:.3f}  d_H={r['d_H']:.3f}  "
                 f"d_sub={r['d_sub']:.3f}  leak'={r['leak_prime']:.3f}  "
                 f"d_feas={(r['d_feas'] or 0):.3f}  "
                 f"{r['filename']} gt={r['heldout']}")

    emit("\n[R3] Subset cohort paired (heldout n_dim_G <= "
         f"{args.cohort_max_dim_g})")
    for model in args.models:
        emit(f"  -- model = {model} --")
        for r in subset:
            if r["model"] != model:
                continue
            emit(f"    {r['metric']:14s}  n={r['n_paired']:>4d}"
                 f"  h={r['human_better']:>4d} m={r['model_better']:>4d}"
                 f"  t={r['tie']:>4d}  z={r['z']:+7.2f}  p={r['p_two_sided']:.2e}")

    summary_path = os.path.join(out_dir, "dH_paper_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {os.path.relpath(summary_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
