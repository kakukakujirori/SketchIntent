"""analyze_v4.py — aggregate per-variant statistics and paired tests for v4 metrics.

Reads the summary CSV produced by ``exp_v4_controls.py`` and emits:

- ``v4_quantile_table.csv``  : median / Q75 / Q90 / mean / frac<1e-4 per variant per
                                metric.  This is the v4 analogue of v3's
                                ``v3_controls_scores_full.csv`` summary.
- ``v4_paired_stats.csv``    : paired Wilcoxon model-vs-heldout per metric and model.
- ``v4_dropout_sweep.csv``   : median per drop rate (dim and rel) per metric.
- ``v4_summary.txt``         : a one-page console-style report.

The script intentionally avoids scipy as a hard dependency for the Wilcoxon
test; we fall back to a normal-approximation z-score if scipy is missing.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

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


def _wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank test on paired diffs.

    Returns ``(z, p, n_nonzero)``.  Uses the normal approximation; ties
    receive average ranks.  This mirrors v3 ``analyze_dalign_evidence.py``
    convention (tie threshold |d| <= 1e-9) so paired z/p reports compare
    directly to existing artifacts.
    """
    pairs = [d for d in diffs if d is not None and abs(d) > 1e-9]
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
        # ranks i..j get same average rank
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed
        for k in range(i, j + 1):
            _, orig_idx = abs_vals[k]
            ranks[orig_idx] = avg_rank
        i = j + 1
    W_plus = sum(r for d, r in zip(pairs, ranks) if d > 0)
    W_minus = sum(r for d, r in zip(pairs, ranks) if d < 0)
    mean_W = n * (n + 1) / 4.0
    var_W = n * (n + 1) * (2 * n + 1) / 24.0
    if var_W == 0:
        return 0.0, 1.0, n
    z = (W_plus - mean_W) / math.sqrt(var_W)
    # two-sided p via complementary error function
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p, n


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input-csv",
        default="artifacts/v4_exp/v4_controls_scores_full.csv",
        help="Summary CSV produced by exp_v4_controls.py.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/v4_exp",
        help="Where to write the aggregated CSVs.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["base", "solver_rl"],
        help="Model variants to include in paired test (vs heldout drop=0).")
    args = parser.parse_args()

    input_path = args.input_csv
    if not os.path.isabs(input_path):
        input_path = os.path.join(REPO_ROOT, input_path)
    rows = load_rows(input_path)
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        print(f"No valid rows in {input_path}")
        return

    # Derive d_K_pure = sqrt(gain_K^2 + leak_K^2) and likewise for d_M.  These
    # drop the effort regulariser so source_self can reach near-zero even
    # when the LS solver leaves a small residual ||u||.  We compute both the
    # per-(rotation, gt) detail value and the source-aggregate min/median
    # so downstream scoring treats them the same as the native metrics.
    for r in rows:
        for tag in ("K", "M"):
            for agg in ("min", "median"):
                g = _finite(r.get(f"{agg}_gain_{tag}"))
                lk = _finite(r.get(f"{agg}_leak_{tag}"))
                if g is None or lk is None:
                    r[f"{agg}_d_{tag}_pure"] = None
                else:
                    r[f"{agg}_d_{tag}_pure"] = math.sqrt(g * g + lk * lk)

    metric_cols = [
        "d_feas", "d_feas_gain", "d_feas_leak",                # v3 baseline
        "d_sub", "align_loss", "rank_loss",                    # Case D
        "d_H", "leak_prime",                                   # Case H
        "d_K", "d_K_pure", "gain_K", "leak_K", "effort_K",     # Case K
        "d_M", "d_M_pure", "gain_M", "leak_M", "effort_M",     # Case M
        "gain_I", "gain_J",                                    # Case I / J
    ]

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r.get("variant", "")].append(r)

    # ---------- 1. Per-variant quantile table ----------
    quant_rows = []
    canonical = ["source_self", "heldout", "dim_drop25", "dim_drop50",
                 "dim_drop75", "dim_drop100", "rel_drop50", "rel_drop100"]
    head = [v for v in canonical if v in by_variant]
    tail = sorted(v for v in by_variant if v not in head)
    for variant in head + tail:
        vrows = by_variant[variant]
        out = {"variant": variant, "n": len(vrows)}
        for m in metric_cols:
            vals = [_finite(r.get(f"min_{m}")) for r in vrows]
            vals = [v for v in vals if v is not None]
            if not vals:
                out[f"med_{m}"] = None
                out[f"q75_{m}"] = None
                out[f"q90_{m}"] = None
                out[f"mean_{m}"] = None
                out[f"frac_lt1e-4_{m}"] = None
                continue
            out[f"med_{m}"] = _quantile(vals, 50)
            out[f"q75_{m}"] = _quantile(vals, 75)
            out[f"q90_{m}"] = _quantile(vals, 90)
            out[f"mean_{m}"] = sum(vals) / len(vals)
            out[f"frac_lt1e-4_{m}"] = sum(1 for v in vals if v < 1e-4) / len(vals)
        quant_rows.append(out)

    os.makedirs(os.path.join(REPO_ROOT, args.out_dir), exist_ok=True)
    quant_path = os.path.join(REPO_ROOT, args.out_dir, "v4_quantile_table.csv")
    if quant_rows:
        with open(quant_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(quant_rows[0].keys()))
            writer.writeheader()
            writer.writerows(quant_rows)
        print(f"Saved: {quant_path}")

    # ---------- 2. Paired Wilcoxon: model vs heldout (drop=0) ----------
    # Pair on rotation_id.  Each rotation has one heldout row and (potentially)
    # one row per model.  Lower-is-better diff = model - heldout; positive
    # means the model is worse.
    paired_rows = []
    by_rot: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        v = r.get("variant", "")
        rid = r.get("rotation_id", "")
        by_rot[rid][v] = r

    for model in args.models:
        for m in metric_cols:
            diffs = []
            for rid, d in by_rot.items():
                if "heldout" not in d or model not in d:
                    continue
                vh = _finite(d["heldout"].get(f"min_{m}"))
                vm = _finite(d[model].get(f"min_{m}"))
                if vh is None or vm is None:
                    continue
                diffs.append(vm - vh)
            if not diffs:
                continue
            n_paired = len(diffs)
            n_model_better = sum(1 for x in diffs if x < -1e-9)
            n_human_better = sum(1 for x in diffs if x > 1e-9)
            n_tie = n_paired - n_model_better - n_human_better
            z, p, _ = _wilcoxon_signed_rank(diffs)
            paired_rows.append({
                "model": model,
                "metric": m,
                "n_paired": n_paired,
                "human_better": n_human_better,
                "model_better": n_model_better,
                "tie": n_tie,
                "z": z,
                "p_two_sided": p,
                "median_diff_model_minus_human": _quantile(diffs, 50),
            })

    paired_path = os.path.join(REPO_ROOT, args.out_dir, "v4_paired_stats.csv")
    if paired_rows:
        with open(paired_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)
        print(f"Saved: {paired_path}")

    # ---------- 3. Dropout sweep table ----------
    sweep_rows = []
    sweep_variants = [v for v in by_variant if v.startswith("dim_drop") or v.startswith("rel_drop") or v == "heldout"]
    sweep_variants.sort(key=lambda v: (
        0 if v == "heldout" else (1 if v.startswith("dim_drop") else 2),
        int(v[len("dim_drop"):]) if v.startswith("dim_drop") else (
            int(v[len("rel_drop"):]) if v.startswith("rel_drop") else 0)
    ))
    for variant in sweep_variants:
        vrows = by_variant[variant]
        out = {"variant": variant, "n": len(vrows)}
        for m in metric_cols:
            vals = [_finite(r.get(f"min_{m}")) for r in vrows]
            vals = [v for v in vals if v is not None]
            out[f"med_{m}"] = _quantile(vals, 50) if vals else None
        sweep_rows.append(out)

    sweep_path = os.path.join(REPO_ROOT, args.out_dir, "v4_dropout_sweep.csv")
    if sweep_rows:
        with open(sweep_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"Saved: {sweep_path}")

    # ---------- 4. Console summary ----------
    summary_lines: list[str] = []

    def emit(s=""):
        summary_lines.append(s)
        print(s)

    emit("=" * 78)
    emit(f"V4 metric analysis — input {os.path.relpath(input_path, REPO_ROOT)}")
    emit(f"Total valid rows: {len(rows)}")
    emit("=" * 78)

    emit("\n[Quantile table — median per variant]")
    header_metrics = ["d_feas", "d_sub", "d_H", "d_K", "d_M", "gain_I", "gain_J"]
    emit(f"  {'variant':14s}  {'n':>4s}  " + "  ".join(f"{m:>9s}" for m in header_metrics))
    for q in quant_rows:
        line = f"  {q['variant']:14s}  {q['n']:>4d}"
        for m in header_metrics:
            v = q.get(f"med_{m}")
            line += f"  {v:>9.4f}" if v is not None else f"  {'NA':>9s}"
        emit(line)

    emit("\n[Paired Wilcoxon — model vs heldout (lower-is-better; signed z>0 = human better)]")
    for model in args.models:
        emit(f"  -- model = {model} --")
        for r in paired_rows:
            if r["model"] != model:
                continue
            emit(
                f"    {r['metric']:14s}  n={r['n_paired']:>4d}"
                f"  h={r['human_better']:>4d} m={r['model_better']:>4d} t={r['tie']:>4d}"
                f"  z={r['z']:+8.3f}  p={r['p_two_sided']:.2e}"
            )

    emit("\n[Dropout sweep — median (lower-is-better)]")
    emit(f"  {'variant':14s}  {'n':>4s}  " + "  ".join(f"{m:>9s}" for m in header_metrics))
    for s in sweep_rows:
        line = f"  {s['variant']:14s}  {s['n']:>4d}"
        for m in header_metrics:
            v = s.get(f"med_{m}")
            line += f"  {v:>9.4f}" if v is not None else f"  {'NA':>9s}"
        emit(line)

    summary_path = os.path.join(REPO_ROOT, args.out_dir, "v4_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
