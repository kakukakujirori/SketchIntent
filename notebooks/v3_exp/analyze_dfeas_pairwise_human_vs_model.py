"""human-human vs model-human d_feas distribution overlay.

TODO §3 wants distributional evidence that d_feas:
  - is small when comparing two human annotators of the same sketch (human
    variation gets absorbed), and
  - is larger for model outputs (real edit-response deviations).

Input: v3 controls summary CSV (`v3_controls_scores.csv`). Variants
`heldout` and `source_self` give the human-human reference distribution;
loaded model labels (e.g. `base`, `solver_rl`, `Vitruvion`) give the
model-human distribution.

Outputs:
- `artifacts/dfeas_human_vs_model_cdf.png`
- `artifacts/dfeas_human_vs_model_stats.csv` — paired stats per model label.

Paired comparison is done per rotation_id: model d_feas vs heldout d_feas
from the same rotation. Reports {wins, losses, ties} and Wilcoxon z / p.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def load(path: str) -> list[dict]:
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        print(f"  [warn] missing: {full}")
        return []
    with open(full, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  loaded {len(rows):5d} rows from {full}")
    return rows


def validated(row: dict) -> bool:
    sv = row.get("source_validated")
    return sv in (True, "True", "true", "1")


def wilcoxon_z(diffs: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank z and two-sided p (normal approx)."""
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0
    pairs = sorted(((abs(d), 1 if d > 0 else -1) for d in nonzero), key=lambda x: x[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    w_plus = sum(r for r, (_, s) in zip(ranks, pairs) if s > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 0.0, 1.0
    z = (w_plus - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def collect(rows: list[dict], variant: str, eps: float | None,
            field: str) -> dict[str, float]:
    """rotation_id -> field value."""
    out: dict[str, float] = {}
    for r in rows:
        if not validated(r):
            continue
        if r.get("variant") != variant:
            continue
        if eps is not None and r.get("eps"):
            try:
                if abs(float(r["eps"]) - eps) > 1e-18:
                    continue
            except ValueError:
                continue
        v = _f(r.get(field))
        if v is None:
            continue
        out[r.get("rotation_id", "")] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--controls-csv",
                   default="artifacts/v3_controls_scores.csv")
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--field", default="min_d_feas")
    p.add_argument("--reference-variant", default="heldout",
                   help="Variant treated as human-human baseline.")
    p.add_argument("--model-variants", nargs="*", default=None,
                   help="Variant labels to treat as model-human. Defaults: "
                        "all non-control variants observed.")
    p.add_argument("--out-cdf",
                   default="artifacts/dfeas_human_vs_model_cdf.png")
    p.add_argument("--out-stats",
                   default="artifacts/dfeas_human_vs_model_stats.csv")
    args = p.parse_args()

    rows = load(args.controls_csv)
    if not rows:
        sys.exit(1)

    ref_vals = collect(rows, args.reference_variant, args.eps, args.field)
    print(f"  reference variant '{args.reference_variant}': n={len(ref_vals)}")

    seen_variants: list[str] = []
    for r in rows:
        v = r.get("variant")
        if v and v not in seen_variants:
            seen_variants.append(v)
    excluded = {"source_self", "heldout", "dim_drop50", "dim_drop100",
                "rel_drop50", "rel_drop100"}
    if args.model_variants:
        model_labels = list(args.model_variants)
    else:
        model_labels = [v for v in seen_variants if v not in excluded]
    print(f"  model variants: {model_labels}")

    # Stats CSV
    stats_path = args.out_stats if os.path.isabs(args.out_stats) \
        else os.path.join(REPO_ROOT, args.out_stats)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_variant", "n_paired", "model_better", "human_better",
                    "tie", "z", "p_two_sided"])
        for lbl in model_labels:
            model_vals = collect(rows, lbl, args.eps, args.field)
            diffs: list[float] = []
            m_better = h_better = tie = 0
            for rotid, mv in model_vals.items():
                hv = ref_vals.get(rotid)
                if hv is None:
                    continue
                d = mv - hv  # low-is-good: positive => human-better
                if abs(d) <= 1e-9:
                    tie += 1
                elif d > 0:
                    h_better += 1
                else:
                    m_better += 1
                diffs.append(d)
            z, pval = wilcoxon_z(diffs)
            w.writerow([lbl, len(diffs), m_better, h_better, tie,
                        f"{z:.3f}", f"{pval:.3e}"])
            print(f"  {lbl:14s} n={len(diffs):4d} m={m_better:4d}"
                  f" h={h_better:4d} t={tie:4d}  z={z:+.3f}  p={pval:.2e}")
    print(f"Saved: {stats_path}")

    # CDF plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib missing; skipping CDF.")
        return
    cdf_path = args.out_cdf if os.path.isabs(args.out_cdf) \
        else os.path.join(REPO_ROOT, args.out_cdf)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series: list[tuple[str, list[float]]] = []
    series.append((f"{args.reference_variant} (human-human)",
                   sorted(ref_vals.values())))
    for lbl in model_labels:
        vals = sorted(collect(rows, lbl, args.eps, args.field).values())
        if vals:
            series.append((f"{lbl} (model-human)", vals))
    for label, xs in series:
        if not xs:
            continue
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=f"{label}  n={len(xs)}", linewidth=1.4)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(args.field)
    ax.set_ylabel("CDF")
    ax.set_title("human-human vs model-human d_feas")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {cdf_path}")


if __name__ == "__main__":
    main()
