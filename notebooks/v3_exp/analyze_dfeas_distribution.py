"""CDF / violin / quantile distribution analysis of d_feas across variants.

Inputs:
- `--input-csv` (default `artifacts/v3_controls_scores.csv`): a v3 controls
  summary CSV with `variant`, `eps`, `min_d_feas` columns.
- `--input-csv-extra` (optional, repeatable): additional summary CSVs to merge
  (e.g. `constrained_response_scores_full.csv`). Each row needs `variant` and
  `min_d_feas`; we infer variant from `pred` if missing.

Outputs (saved next to inputs unless overridden):
- `artifacts/dfeas_cdf.png`           — overlay CDFs of `min_d_feas` per variant.
- `artifacts/dfeas_violin.png`        — half-violin per variant.
- `artifacts/dfeas_quantile_table.csv`— median / Q75 / Q90 / % near-zero / n.

For multi-eps v3 controls we keep eps=1e-9 by default (configurable via `--eps`).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _is_finite(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def infer_variant(row: dict) -> str | None:
    v = row.get("variant")
    if v:
        return v
    pred = row.get("pred")
    drop_rate = row.get("drop_rate")
    try:
        dr = float(drop_rate) if drop_rate is not None else None
    except ValueError:
        dr = None
    if pred and dr == -1.0:
        return pred  # model variant label (e.g. base, solver_rl, Vitruvion)
    if dr == 0.0:
        return "heldout"
    if dr is not None and dr > 0:
        return f"dim_drop{int(dr)}"
    return None


def collect_values(rows: list[dict], variant: str,
                   eps: float | None,
                   field: str) -> list[float]:
    vals: list[float] = []
    for r in rows:
        v = infer_variant(r)
        if v != variant:
            continue
        if eps is not None and r.get("eps") not in (None, ""):
            try:
                if abs(float(r["eps"]) - eps) > 1e-18:
                    continue
            except ValueError:
                continue
        # validation gate: require source_validated truthy if present
        sv = r.get("source_validated")
        if sv is not None and sv not in ("", "True", "true", "1", True):
            continue
        x = r.get(field)
        if _is_finite(x):
            vals.append(float(x))
    return vals


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def near_zero_fraction(xs: list[float], threshold: float) -> float | None:
    if not xs:
        return None
    return sum(1 for x in xs if x < threshold) / len(xs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input-csv",
                   default="artifacts/v3_controls_scores.csv")
    p.add_argument("--input-csv-extra", action="append", default=None,
                   help="Additional summary CSVs to merge.")
    p.add_argument("--eps", type=float, default=1e-9,
                   help="Filter eps value for v3 control rows. None = no filter.")
    p.add_argument("--field", default="min_d_feas",
                   help="Column to extract (default: min_d_feas).")
    p.add_argument("--near-zero", type=float, default=1e-4)
    p.add_argument("--variants", nargs="*", default=None,
                   help="Override variant order. Default: auto-detect.")
    p.add_argument("--out-cdf", default="artifacts/dfeas_cdf.png")
    p.add_argument("--out-violin", default="artifacts/dfeas_violin.png")
    p.add_argument("--out-table", default="artifacts/dfeas_quantile_table.csv")
    args = p.parse_args()

    paths = [args.input_csv] + list(args.input_csv_extra or [])
    rows: list[dict] = []
    for q in paths:
        full = q if os.path.isabs(q) else os.path.join(REPO_ROOT, q)
        if os.path.exists(full):
            r = load_rows(full)
            print(f"  loaded {len(r):5d} rows from {full}")
            rows.extend(r)
        else:
            print(f"  [warn] missing {full}")
    if not rows:
        print("No rows; aborting.")
        sys.exit(1)

    if args.variants is None:
        variants_seen: list[str] = []
        for r in rows:
            v = infer_variant(r)
            if v and v not in variants_seen:
                variants_seen.append(v)
        ordering = ["source_self", "heldout"]
        rest = [v for v in variants_seen if v not in ordering]
        variants = [v for v in ordering if v in variants_seen] + sorted(rest)
    else:
        variants = list(args.variants)

    data: dict[str, list[float]] = {}
    for v in variants:
        data[v] = collect_values(rows, v, eps=args.eps, field=args.field)
        print(f"  variant {v:20s}  n={len(data[v]):5d}")

    # --- Quantile table CSV ---
    table_path = args.out_table if os.path.isabs(args.out_table) \
        else os.path.join(REPO_ROOT, args.out_table)
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "median", "Q75", "Q90", "mean",
                    f"frac<{args.near_zero}"])
        for v in variants:
            xs = data[v]
            w.writerow([
                v, len(xs),
                quantile(xs, 0.5),
                quantile(xs, 0.75),
                quantile(xs, 0.90),
                (sum(xs) / len(xs)) if xs else None,
                near_zero_fraction(xs, args.near_zero),
            ])
    print(f"Saved: {table_path}")

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping plots.")
        return

    cdf_path = args.out_cdf if os.path.isabs(args.out_cdf) \
        else os.path.join(REPO_ROOT, args.out_cdf)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        xs = sorted(data[v])
        if not xs:
            continue
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=f"{v} (n={len(xs)})", linewidth=1.4)
    ax.set_xlabel(args.field)
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF of {args.field}")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {cdf_path}")

    violin_path = args.out_violin if os.path.isabs(args.out_violin) \
        else os.path.join(REPO_ROOT, args.out_violin)
    keep = [(v, data[v]) for v in variants if data[v]]
    if keep:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.violinplot([d for _, d in keep], showmedians=True)
        ax.set_xticks(range(1, len(keep) + 1))
        ax.set_xticklabels([v for v, _ in keep], rotation=30, ha="right")
        ax.set_ylabel(args.field)
        ax.set_title(f"Distribution of {args.field}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(violin_path, dpi=140)
        plt.close(fig)
        print(f"Saved: {violin_path}")


if __name__ == "__main__":
    main()
