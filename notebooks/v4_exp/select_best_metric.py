"""select_best_metric.py — score candidate metrics against v4 criteria.

Combines the per-variant quantile table, paired stats, and dropout sweep
into a single weighted scorecard.  Mirrors the v4 §4 criteria:

- N1  ``source_self frac<1e-4 >= 0.99``
- N2  ``heldout median << dim_drop50 median``
- N3  paired Wilcoxon (model vs heldout): human_better > model_better with
      ``p < 1e-5`` for at least one of {base, solver_rl}
- N4  dropout monotonic: drop_0 <= drop_25 <= drop_50 <= drop_75 <= drop_100
- N5  rel_drop50 / rel_drop100 medians clearly above heldout
- R1  paired |z| for solver_rl and base (larger is better)
- R3  base vs heldout median ratio (larger separation is better)
- R4  Pearson |rho|(gain_*, leak_*) — smaller means more independent channels

Outputs ``artifacts/v4_exp/v4_scorecard.csv`` and a console verdict.
"""
from __future__ import annotations

import argparse
import csv
import math
import os

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


def _load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _row(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quantile-csv",
                        default="artifacts/v4_exp/v4_quantile_table.csv")
    parser.add_argument("--paired-csv",
                        default="artifacts/v4_exp/v4_paired_stats.csv")
    parser.add_argument("--sweep-csv",
                        default="artifacts/v4_exp/v4_dropout_sweep.csv")
    parser.add_argument("--out-csv",
                        default="artifacts/v4_exp/v4_scorecard.csv")
    args = parser.parse_args()

    def _path(p):
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

    quant = _load(_path(args.quantile_csv))
    paired = _load(_path(args.paired_csv))
    sweep = _load(_path(args.sweep_csv))

    candidates = [
        "d_feas",
        "d_sub", "d_H",
        "d_K", "d_K_pure",
        "d_M", "d_M_pure",
        "gain_I", "gain_J",
    ]

    rows = []
    for m in candidates:
        srow = _row(quant, variant="source_self")
        hrow = _row(quant, variant="heldout")
        dd50 = _row(quant, variant="dim_drop50")
        dd100 = _row(quant, variant="dim_drop100")
        rd50 = _row(quant, variant="rel_drop50")
        rd100 = _row(quant, variant="rel_drop100")

        out = {"metric": m}
        out["self_frac<1e-4"] = _finite(srow.get(f"frac_lt1e-4_{m}")) if srow else None
        out["heldout_median"] = _finite(hrow.get(f"med_{m}")) if hrow else None
        out["heldout_frac<1e-4"] = _finite(hrow.get(f"frac_lt1e-4_{m}")) if hrow else None
        out["heldout_Q90"] = _finite(hrow.get(f"q90_{m}")) if hrow else None
        out["dim50_median"] = _finite(dd50.get(f"med_{m}")) if dd50 else None
        out["dim100_median"] = _finite(dd100.get(f"med_{m}")) if dd100 else None
        out["rel50_median"] = _finite(rd50.get(f"med_{m}")) if rd50 else None
        out["rel100_median"] = _finite(rd100.get(f"med_{m}")) if rd100 else None

        # Sweep monotonicity check (heldout -> dim_drop25 -> 50 -> 75 -> 100)
        sweep_vals = []
        for v in ("heldout", "dim_drop25", "dim_drop50", "dim_drop75", "dim_drop100"):
            r = _row(sweep, variant=v)
            sweep_vals.append(_finite(r.get(f"med_{m}")) if r else None)
        is_monotonic = all(
            sweep_vals[i] is None or sweep_vals[i + 1] is None
            or sweep_vals[i] <= sweep_vals[i + 1] + 1e-9
            for i in range(len(sweep_vals) - 1)
        )
        out["dim_sweep_monotonic"] = is_monotonic

        # Paired stats (model vs heldout). Lower-is-better → human_better=good signal.
        for model in ("base", "solver_rl"):
            pr = _row(paired, model=model, metric=m)
            if pr is not None:
                out[f"{model}_n"] = pr.get("n_paired")
                out[f"{model}_h"] = pr.get("human_better")
                out[f"{model}_m"] = pr.get("model_better")
                out[f"{model}_z"] = _finite(pr.get("z"))
                out[f"{model}_p"] = _finite(pr.get("p_two_sided"))

        # Score (heuristic, all in [0,1])
        score_parts = []
        # N1 — self-near-zero
        if out["self_frac<1e-4"] is not None:
            score_parts.append(("N1", 1.0 if out["self_frac<1e-4"] >= 0.99 else 0.0))
        # N2 — heldout much smaller than dim_drop50
        if out["heldout_median"] is not None and out["dim50_median"] not in (None, 0.0):
            ratio = out["heldout_median"] / out["dim50_median"]
            score_parts.append(("N2", 1.0 if ratio < 0.5 else max(0.0, (1.0 - ratio))))
        # N3 — model gets penalised (signed z>0 in our convention)
        if "base_z" in out and out.get("base_z") is not None:
            z = out["base_z"]
            score_parts.append(("N3a", min(1.0, max(0.0, z / 20.0))))
        if "solver_rl_z" in out and out.get("solver_rl_z") is not None:
            z = out["solver_rl_z"]
            score_parts.append(("N3b", min(1.0, max(0.0, z / 20.0))))
        # N4 — monotonic sweep
        score_parts.append(("N4", 1.0 if out["dim_sweep_monotonic"] else 0.0))
        # N5 — rel_drop separates from heldout
        if out["heldout_median"] is not None and out["rel50_median"] is not None:
            score_parts.append((
                "N5",
                1.0 if out["rel50_median"] - out["heldout_median"] > 0.05 else 0.0,
            ))
        # R3 — base vs heldout separation
        if out.get("base_n") and out.get("base_h") and out.get("base_m"):
            try:
                fh = int(out["base_h"]) / int(out["base_n"])
                fm = int(out["base_m"]) / int(out["base_n"])
                score_parts.append(("R3_base", max(0.0, fh - fm)))
            except (ValueError, ZeroDivisionError):
                pass

        out["score_total"] = sum(s for _, s in score_parts)
        out["score_n_terms"] = len(score_parts)
        out["score_breakdown"] = ";".join(f"{k}={s:.2f}" for k, s in score_parts)
        rows.append(out)

    rows.sort(key=lambda r: r.get("score_total") or 0, reverse=True)

    out_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(REPO_ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")

    print("\n[Scorecard (higher = better, max ≈ 7)]")
    print(f"  {'metric':12s}  {'score':>6s}  {'self':>6s}  {'hold':>8s}  {'d50':>8s}  {'r50':>8s}  {'mono':>5s}  {'base_z':>8s}  {'rl_z':>8s}")
    for r in rows:
        score = f"{r['score_total']:6.2f}"
        sf = f"{r['self_frac<1e-4']:.3f}" if r.get("self_frac<1e-4") is not None else "NA"
        hm = f"{r['heldout_median']:.4f}" if r.get("heldout_median") is not None else "NA"
        d5 = f"{r['dim50_median']:.4f}" if r.get("dim50_median") is not None else "NA"
        r5 = f"{r['rel50_median']:.4f}" if r.get("rel50_median") is not None else "NA"
        mono = "yes" if r["dim_sweep_monotonic"] else "no"
        bz = f"{r['base_z']:+.2f}" if r.get("base_z") is not None else "NA"
        sz = f"{r['solver_rl_z']:+.2f}" if r.get("solver_rl_z") is not None else "NA"
        print(f"  {r['metric']:12s}  {score}  {sf:>6s}  {hm:>8s}  {d5:>8s}  {r5:>8s}  {mono:>5s}  {bz:>8s}  {sz:>8s}")

    print("\nBreakdown:")
    for r in rows[:3]:
        print(f"  {r['metric']}: {r['score_breakdown']}")


if __name__ == "__main__":
    main()
