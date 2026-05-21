"""Task 5 sub-ablations (smatch correlation, gate sensitivity) for d_H.

These two ablations do not require re-running the v4 pipeline; they reuse
existing CSVs (smatch_vs_dfeas_details_full.csv from v3 and
source_conditioned_dropout_scores_full.prev.csv as the strict-gate cohort).

Outputs to artifacts/v4_exp/:
- ablation_smatch_dH_correlation.csv : per-(rotation, gt) join of smatch and
                                       d_H, plus Spearman/Pearson correlations
                                       grouped by variant.
- ablation_gate_strict.csv            : paired Wilcoxon (model vs heldout) for
                                       d_H/d_sub/leak_prime/d_feas restricted
                                       to the strict-gate cohort (DoF==2 ∧
                                       Converged), using rotation_id
                                       membership in the prev OES CSV.
- ablation_summary.txt                : console summary.
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


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx = rank(xs); ry = rank(ys)
    return _pearson(rx, ry)


# ---------------------------------------------------------------------------
# Ablation 1: smatch vs d_H correlation
# ---------------------------------------------------------------------------

def load_smatch(path):
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    rows = []
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source_validated") not in ("True", "true", "1"):
                continue
            rows.append(r)
    return rows


def load_v4_details(path):
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    rows = []
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            rows.append(r)
    return rows


def smatch_variant_from_pred(pred_label: str) -> str:
    """Map smatch CSV's 'pred' label to v4 'variant' label.

    smatch pred labels look like 'Ryotaro_drop50' or 'base' / 'solver_rl'.
    v4 variants are 'dim_drop50' / 'base' / 'solver_rl' / 'heldout' / ...
    smatch always uses dim-drop and labels human heldout as a separate human row;
    we restrict to comparable model + dim_drop variants and 'drop0' = heldout.
    """
    if pred_label in ("base", "solver_rl"):
        return pred_label
    if "_drop" in pred_label:
        rate = pred_label.split("_drop", 1)[1]
        try:
            r = float(rate)
        except ValueError:
            return ""
        if abs(r) < 1e-9:
            return "heldout"
        return f"dim_drop{int(r) if r == int(r) else r:g}"
    return ""


def smatch_correlation(smatch_rows, v4_detail_rows):
    """Join on (rotation_id, gt, mapped_variant) and compute correlations."""
    v4_index: dict[tuple[str, str, str], dict] = {}
    for r in v4_detail_rows:
        key = (r["rotation_id"], r["variant"], r["gt"])
        v4_index[key] = r

    joined: list[dict] = []
    for s in smatch_rows:
        rid = s["rotation_id"]
        gt = s["gt"]
        variant = smatch_variant_from_pred(s.get("pred", ""))
        if not variant:
            continue
        key = (rid, variant, gt)
        v = v4_index.get(key)
        if v is None:
            continue
        joined.append({
            "rotation_id": rid,
            "variant": variant,
            "gt": gt,
            "d_smatch_rel": _finite(s.get("d_smatch_rel")),
            "d_smatch_sym": _finite(s.get("d_smatch_sym")),
            "d_smatch_rms": _finite(s.get("d_smatch_rms")),
            "d_feas": _finite(v.get("d_feas")),
            "d_H": _finite(v.get("d_H")),
            "d_sub": _finite(v.get("d_sub")),
            "leak_prime": _finite(v.get("leak_prime")),
        })

    # Aggregate correlations by variant
    by_var: dict[str, list[dict]] = defaultdict(list)
    for j in joined:
        by_var[j["variant"]].append(j)

    summary = []
    for variant in sorted(by_var.keys()):
        group = by_var[variant]
        for smatch_key in ("d_smatch_rel", "d_smatch_sym", "d_smatch_rms"):
            for v4_key in ("d_H", "d_sub", "leak_prime", "d_feas"):
                pairs = [(g[smatch_key], g[v4_key]) for g in group
                         if g[smatch_key] is not None and g[v4_key] is not None]
                if len(pairs) < 5:
                    continue
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                summary.append({
                    "variant": variant,
                    "smatch_metric": smatch_key,
                    "v4_metric": v4_key,
                    "n": len(pairs),
                    "spearman": _spearman(xs, ys),
                    "pearson": _pearson(xs, ys),
                    "med_smatch": _quantile(xs, 50),
                    "med_v4": _quantile(ys, 50),
                })
    return joined, summary


# ---------------------------------------------------------------------------
# Ablation 2: strict-gate cohort paired Wilcoxon for d_H
# ---------------------------------------------------------------------------

def load_strict_cohort(prev_oes_path):
    """Return dict[model_label] = set[rotation_id] admitted under strict gate."""
    full = prev_oes_path if os.path.isabs(prev_oes_path) else os.path.join(REPO_ROOT, prev_oes_path)
    by_pred: dict[str, set[str]] = defaultdict(set)
    if not os.path.exists(full):
        return by_pred
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("drop_rate") not in ("-1", "-1.0"):
                continue
            if r.get("source_validated") not in ("True", "true", "1"):
                continue
            pred = r.get("pred", "")
            if pred:
                by_pred[pred].add(r["rotation_id"])
    return by_pred


def load_v4_scores(path):
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    rows = []
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            rows.append(r)
    return rows


def strict_gate_paired(v4_rows, strict_cohort, metrics):
    """For each model, restrict to rotations in strict_cohort[model], pair with
    heldout rotations, run Wilcoxon."""
    by_rot = defaultdict(dict)
    for r in v4_rows:
        by_rot[r["rotation_id"]][r["variant"]] = r

    out = []
    for model, rids in strict_cohort.items():
        if model not in ("base", "solver_rl"):
            continue
        for m in metrics:
            diffs = []
            for rid in rids:
                if rid not in by_rot:
                    continue
                d = by_rot[rid]
                if "heldout" not in d or model not in d:
                    continue
                vh = _finite(d["heldout"].get(f"min_{m}"))
                vm = _finite(d[model].get(f"min_{m}"))
                if vh is None or vm is None:
                    continue
                diffs.append(vm - vh)
            if not diffs:
                continue
            h = sum(1 for x in diffs if x > 1e-9)
            mb = sum(1 for x in diffs if x < -1e-9)
            tie = len(diffs) - h - mb
            z, p, _ = _wilcoxon(diffs)
            out.append({
                "model": model,
                "metric": m,
                "n_strict_cohort": len(rids),
                "n_paired": len(diffs),
                "human_better": h,
                "model_better": mb,
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
    p.add_argument("--smatch-csv", default="artifacts/smatch_vs_dfeas_details_full.csv")
    p.add_argument("--v4-details", default="artifacts/v4_exp/v4_controls_details_full.csv")
    p.add_argument("--v4-scores", default="artifacts/v4_exp/v4_controls_scores_full.csv")
    p.add_argument("--prev-oes-csv",
                   default="artifacts/source_conditioned_dropout_scores_full.prev.csv")
    p.add_argument("--out-dir", default="artifacts/v4_exp")
    args = p.parse_args()

    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    smatch_rows = load_smatch(args.smatch_csv)
    v4_details = load_v4_details(args.v4_details)
    v4_scores = load_v4_scores(args.v4_scores)
    strict_cohort = load_strict_cohort(args.prev_oes_csv)

    print(f"smatch validated rows: {len(smatch_rows)}")
    print(f"v4 detail rows: {len(v4_details)}")
    print(f"v4 score rows: {len(v4_scores)}")
    print(f"strict-gate cohort: " +
          ", ".join(f"{k}={len(v)}" for k, v in strict_cohort.items()))

    # ---------- smatch correlation ----------
    joined, corr_summary = smatch_correlation(smatch_rows, v4_details)
    print(f"smatch ↔ v4 joined rows: {len(joined)}")
    write_csv(os.path.join(out_dir, "ablation_smatch_dH_correlation.csv"),
              corr_summary)

    # ---------- strict gate ----------
    metrics = ["d_feas", "d_H", "d_sub", "leak_prime",
               "d_feas_gain", "d_feas_leak"]
    strict_rows = strict_gate_paired(v4_scores, strict_cohort, metrics)
    write_csv(os.path.join(out_dir, "ablation_gate_strict.csv"), strict_rows)

    # ---------- console summary ----------
    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("=" * 78)
    emit("Task 5 sub-ablations: smatch correlation + strict-gate sensitivity")
    emit("=" * 78)

    emit("\n[smatch ↔ d_H correlation by variant]")
    emit(f"  {'variant':14s} {'smatch':14s} {'v4':12s} {'n':>4s} "
         f"{'rho':>8s} {'r':>8s} {'med_sm':>10s} {'med_v4':>10s}")
    for r in corr_summary:
        emit(f"  {r['variant']:14s} {r['smatch_metric']:14s} {r['v4_metric']:12s} "
             f"{r['n']:>4d} {(r['spearman'] or 0):>+8.3f} "
             f"{(r['pearson'] or 0):>+8.3f} "
             f"{(r['med_smatch'] or 0):>10.4f} {(r['med_v4'] or 0):>10.4f}")

    emit("\n[Strict-gate cohort paired Wilcoxon (DoF==2 ∧ Converged)]")
    emit("Comparison vs heldout (lower-is-better; z>0 = human better)")
    for model in ("base", "solver_rl"):
        emit(f"  -- model = {model} (n_strict_cohort=" +
             str(len(strict_cohort.get(model, set()))) + ") --")
        for r in strict_rows:
            if r["model"] != model:
                continue
            emit(f"    {r['metric']:14s}  n={r['n_paired']:>4d}"
                 f"  h={r['human_better']:>4d} m={r['model_better']:>4d}"
                 f"  t={r['tie']:>4d}  z={r['z']:+7.2f}  p={r['p_two_sided']:.2e}")

    summary_path = os.path.join(out_dir, "ablation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {os.path.relpath(summary_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
