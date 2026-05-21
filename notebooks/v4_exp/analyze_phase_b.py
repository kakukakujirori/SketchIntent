"""analyze_phase_b.py — aggregate results of Task 2 (sensitivity sweep) and
Task 5 (PP-Coincident ablation) into d_H-centric robustness tables.

Inputs (auto-detected when present):
- artifacts/v4_exp/v4_controls_scores_full.csv  (center: eps=1e-9 / delta=0.01 / bbox=1.0)
- artifacts/v4_exp/sensitivity/scores_{eps_1e-07,eps_1e-11,delta_0.005,delta_0.05,bbox_0.5,bbox_2.0}.csv
- artifacts/v4_exp/v4_controls_scores_no_ppcoincident.csv

Outputs to artifacts/v4_exp/:
- sensitivity_summary.csv : per (axis, value, variant) median d_H / d_sub / leak_prime
- sensitivity_paired.csv  : per (axis, value, model) paired Wilcoxon for d_H, etc.
- pp_coincident_paired.csv: paired Wilcoxon for d_H with vs without PP-coincident
- phase_b_summary.txt     : human-readable rollup.
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


def load_rows(path):
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        return None
    rows = []
    with open(full, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            rows.append(r)
    return rows


def per_variant_median(rows, metric_key):
    out = {}
    by_v = defaultdict(list)
    for r in rows:
        by_v[r.get("variant", "")].append(r)
    for v, vs in by_v.items():
        vals = [_finite(r.get(f"min_{metric_key}")) for r in vs]
        vals = [x for x in vals if x is not None]
        out[v] = {
            "n": len(vs),
            "median": _quantile(vals, 50) if vals else None,
            "q90": _quantile(vals, 90) if vals else None,
        }
    return out


def paired_wilcoxon(rows, model, metric):
    by_rot = defaultdict(dict)
    for r in rows:
        by_rot[r["rotation_id"]][r["variant"]] = r
    diffs = []
    for rid, d in by_rot.items():
        if "heldout" not in d or model not in d:
            continue
        vh = _finite(d["heldout"].get(f"min_{metric}"))
        vm = _finite(d[model].get(f"min_{metric}"))
        if vh is None or vm is None:
            continue
        diffs.append(vm - vh)
    if not diffs:
        return None
    h = sum(1 for x in diffs if x > 1e-9)
    m = sum(1 for x in diffs if x < -1e-9)
    tie = len(diffs) - h - m
    z, p, _ = _wilcoxon(diffs)
    return {
        "n": len(diffs),
        "h_better": h,
        "m_better": m,
        "tie": tie,
        "z": z,
        "p": p,
        "median_diff": _quantile(diffs, 50),
    }


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
    p.add_argument("--center-csv",
                   default="artifacts/v4_exp/v4_controls_scores_full.csv")
    p.add_argument("--sensitivity-dir",
                   default="artifacts/v4_exp/sensitivity")
    p.add_argument("--pp-coincident-csv",
                   default="artifacts/v4_exp/v4_controls_scores_no_ppcoincident.csv")
    p.add_argument("--out-dir", default="artifacts/v4_exp")
    args = p.parse_args()

    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    center = load_rows(args.center_csv)
    if center is None:
        print(f"ERROR: missing center CSV {args.center_csv}")
        return
    print(f"loaded center ({len(center)} rows)")

    metrics = ["d_feas", "d_H", "d_sub", "leak_prime"]
    canonical_variants = ["source_self", "heldout",
                          "dim_drop25", "dim_drop50", "dim_drop75", "dim_drop100",
                          "rel_drop50", "rel_drop100",
                          "base", "solver_rl"]

    # ---------- Sensitivity sweep ----------
    sens_dir = os.path.join(REPO_ROOT, args.sensitivity_dir)
    settings = [
        ("center", "eps", "1e-9", center),
        ("center", "delta_scale", "0.01", center),
        ("center", "bbox", "1.0", center),
    ]
    if os.path.isdir(sens_dir):
        sweep_files = {
            ("eps", "1e-7"):  os.path.join(sens_dir, "scores_eps_1e-07.csv"),
            ("eps", "1e-11"): os.path.join(sens_dir, "scores_eps_1e-11.csv"),
            ("delta_scale", "0.005"): os.path.join(sens_dir, "scores_delta_0.005.csv"),
            ("delta_scale", "0.05"):  os.path.join(sens_dir, "scores_delta_0.05.csv"),
            ("bbox", "0.5"): os.path.join(sens_dir, "scores_bbox_0.5.csv"),
            ("bbox", "2.0"): os.path.join(sens_dir, "scores_bbox_2.0.csv"),
        }
        for (axis, val), path in sweep_files.items():
            rs = load_rows(path)
            if rs is None:
                print(f"  [missing] {axis}={val} ({os.path.relpath(path, REPO_ROOT)})")
                continue
            settings.append(("off", axis, val, rs))
            print(f"  loaded {axis}={val} ({len(rs)} rows)")
    else:
        print(f"  (no sensitivity dir at {sens_dir})")

    # Sensitivity summary table
    sens_rows = []
    seen_axes_centers = set()
    for kind, axis, val, rs in settings:
        # avoid duplicating center row 3 times for 3 axes
        center_key = (kind, axis)
        if kind == "center":
            if axis in seen_axes_centers:
                # only show once
                pass
            else:
                seen_axes_centers.add(axis)
        for m in metrics:
            stats_by_v = per_variant_median(rs, m)
            for v in canonical_variants:
                if v not in stats_by_v:
                    continue
                s = stats_by_v[v]
                sens_rows.append({
                    "kind": kind,
                    "axis": axis,
                    "value": val,
                    "variant": v,
                    "metric": m,
                    "n": s["n"],
                    "median": s["median"],
                    "q90": s["q90"],
                })
    write_csv(os.path.join(out_dir, "sensitivity_summary.csv"), sens_rows)

    # Sensitivity paired Wilcoxon (model vs heldout) for d_H + d_sub + leak_prime
    sens_paired_rows = []
    for kind, axis, val, rs in settings:
        for model in ("base", "solver_rl"):
            for m in metrics:
                res = paired_wilcoxon(rs, model, m)
                if res is None:
                    continue
                sens_paired_rows.append({
                    "kind": kind,
                    "axis": axis,
                    "value": val,
                    "model": model,
                    "metric": m,
                    "n": res["n"],
                    "h_better": res["h_better"],
                    "m_better": res["m_better"],
                    "tie": res["tie"],
                    "z": res["z"],
                    "p": res["p"],
                    "median_diff": res["median_diff"],
                })
    write_csv(os.path.join(out_dir, "sensitivity_paired.csv"), sens_paired_rows)

    # ---------- PP-Coincident ablation ----------
    pp_rows = load_rows(args.pp_coincident_csv)
    pp_paired_rows = []
    if pp_rows is not None:
        print(f"loaded PP-coincident excluded ({len(pp_rows)} rows)")
        for model in ("base", "solver_rl"):
            for m in metrics:
                res_center = paired_wilcoxon(center, model, m)
                res_pp = paired_wilcoxon(pp_rows, model, m)
                if res_center is None or res_pp is None:
                    continue
                pp_paired_rows.append({
                    "model": model,
                    "metric": m,
                    "setting": "include_pp",
                    **{k: res_center[k] for k in ("n", "h_better", "m_better",
                                                  "tie", "z", "p", "median_diff")},
                })
                pp_paired_rows.append({
                    "model": model,
                    "metric": m,
                    "setting": "exclude_pp",
                    **{k: res_pp[k] for k in ("n", "h_better", "m_better",
                                              "tie", "z", "p", "median_diff")},
                })
        write_csv(os.path.join(out_dir, "pp_coincident_paired.csv"), pp_paired_rows)

        # Diagnostic: per-rotation leak_prime change between PP-included and PP-excluded
        center_idx = {(r["rotation_id"], r["variant"]): r for r in center}
        pp_idx = {(r["rotation_id"], r["variant"]): r for r in pp_rows}
        diffs = []
        for k, c_row in center_idx.items():
            p_row = pp_idx.get(k)
            if p_row is None:
                continue
            for m in ("d_H", "leak_prime", "d_sub"):
                vc = _finite(c_row.get(f"min_{m}"))
                vp = _finite(p_row.get(f"min_{m}"))
                if vc is None or vp is None:
                    continue
                diffs.append({"rotation_id": k[0], "variant": k[1], "metric": m,
                              "include_pp": vc, "exclude_pp": vp,
                              "delta": vp - vc})
        # summarise
        pp_diag = []
        for m in ("d_H", "leak_prime", "d_sub"):
            for v in ("heldout", "base", "solver_rl", "rel_drop50",
                      "rel_drop100", "dim_drop50"):
                xs = [d["delta"] for d in diffs if d["metric"] == m and d["variant"] == v]
                if not xs:
                    continue
                xs_abs = [abs(x) for x in xs]
                changed = sum(1 for x in xs if abs(x) > 1e-6)
                pp_diag.append({
                    "metric": m, "variant": v, "n": len(xs),
                    "changed": changed,
                    "med_abs_delta": _quantile(xs_abs, 50),
                    "q90_abs_delta": _quantile(xs_abs, 90),
                    "med_signed_delta": _quantile(xs, 50),
                })
        write_csv(os.path.join(out_dir, "pp_coincident_diagnostic.csv"), pp_diag)
    else:
        print(f"  (no PP-coincident CSV at {args.pp_coincident_csv})")

    # ---------- console summary ----------
    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("=" * 78)
    emit("Phase B summary (Task 2 sensitivity + Task 5 PP-Coincident)")
    emit("=" * 78)

    # Sensitivity: per axis × value, show median d_H for base / solver_rl / heldout
    emit("\n[Sensitivity median d_H by setting]")
    emit(f"  {'axis':12s} {'value':>8s}  {'heldout':>10s} {'base':>10s} {'solver_rl':>10s}  "
         f"{'rel_drop50':>10s} {'dim_drop50':>10s}")
    for kind, axis, val, rs in settings:
        if kind == "center" and axis != "eps":
            continue  # only show center once
        d_H_by_v = per_variant_median(rs, "d_H")
        def f(v):
            x = d_H_by_v.get(v, {}).get("median")
            return f"{x:>10.4f}" if x is not None else f"{'NA':>10s}"
        lbl = "1e-9 (ctr)" if kind == "center" else val
        emit(f"  {axis:12s} {lbl:>8s}  {f('heldout')} {f('base')} {f('solver_rl')}  "
             f"{f('rel_drop50')} {f('dim_drop50')}")

    emit("\n[Sensitivity paired z (d_H, model vs heldout)]")
    emit(f"  {'axis':12s} {'value':>10s}  {'base_z':>8s} {'base_p':>9s}  "
         f"{'rl_z':>8s} {'rl_p':>9s}")
    for kind, axis, val, rs in settings:
        if kind == "center" and axis != "eps":
            continue
        rb = paired_wilcoxon(rs, "base", "d_H") or {}
        rr = paired_wilcoxon(rs, "solver_rl", "d_H") or {}
        lbl = "1e-9 (ctr)" if kind == "center" else val
        emit(f"  {axis:12s} {lbl:>10s}  "
             f"{(rb.get('z') or 0):>+8.2f} {(rb.get('p') or 1):>9.2e}  "
             f"{(rr.get('z') or 0):>+8.2f} {(rr.get('p') or 1):>9.2e}")

    if pp_paired_rows:
        emit("\n[PP-Coincident ablation paired (d_H)]")
        for model in ("base", "solver_rl"):
            emit(f"  -- model = {model} --")
            for r in pp_paired_rows:
                if r["model"] != model or r["metric"] != "d_H":
                    continue
                emit(f"    {r['setting']:12s}  n={r['n']:>4d}"
                     f"  h={r['h_better']:>4d} m={r['m_better']:>4d}"
                     f"  t={r['tie']:>4d}  z={r['z']:+7.2f}  p={r['p']:.2e}")
        emit("\n[PP-Coincident leak_prime paired]")
        for model in ("base", "solver_rl"):
            emit(f"  -- model = {model} --")
            for r in pp_paired_rows:
                if r["model"] != model or r["metric"] != "leak_prime":
                    continue
                emit(f"    {r['setting']:12s}  n={r['n']:>4d}"
                     f"  h={r['h_better']:>4d} m={r['m_better']:>4d}"
                     f"  t={r['tie']:>4d}  z={r['z']:+7.2f}  p={r['p']:.2e}")

    summary_path = os.path.join(out_dir, "phase_b_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {os.path.relpath(summary_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
