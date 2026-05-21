"""analyze_r2_continuity.py — per-rotation continuity diagnostics (v4 §4.3 R2).

Given a summary CSV from ``exp_v4_controls.py`` that contains a fine-grained
``dim_drop`` sweep, this script measures:

- **Monotonicity rate**: fraction of (rotation, transition) pairs where the
  metric strictly increases with the drop rate.  v3 ``d_feas`` collapsed to
  ``{0, 1/k_q, ..., 1}`` per probe and showed step-shaped trajectories; a
  smoother metric will have higher monotonicity at small transitions.
- **Step-size distribution**: median absolute first difference between
  consecutive drop rates.  A truly continuous metric varies in proportion
  to the dropout step; a binary metric mostly emits 0 or 1.
- **Distinct-value count** per rotation across the sweep.  Binary gain in
  v3 had at most ``k_q+1`` distinct values per rotation; v4 candidates K/M
  should produce strictly more.

Output:
  ``artifacts/v4_exp/v4_r2_continuity.csv`` and a short console summary.
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


def _drop_rate_from_variant(name: str):
    """Return (kind, rate) for variants like 'dim_drop25' or ('heldout', 0)."""
    if name == "heldout":
        return ("dim", 0.0)
    if name.startswith("dim_drop"):
        try:
            return ("dim", float(name[len("dim_drop"):]))
        except ValueError:
            return None
    if name.startswith("rel_drop"):
        try:
            return ("rel", float(name[len("rel_drop"):]))
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input-csv",
        default="artifacts/v4_exp/v4_controls_scores_sweep.csv",
        help="Summary CSV produced by exp_v4_controls.py with a fine dim_drop sweep.",
    )
    parser.add_argument(
        "--out-csv",
        default="artifacts/v4_exp/v4_r2_continuity.csv",
    )
    parser.add_argument("--kind", default="dim", choices=("dim", "rel"))
    args = parser.parse_args()

    input_path = args.input_csv
    if not os.path.isabs(input_path):
        input_path = os.path.join(REPO_ROOT, input_path)
    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if not r.get("error")]

    metric_cols = [
        "d_feas", "d_feas_gain", "d_feas_leak",
        "d_sub", "d_H", "leak_prime",
        "d_K", "gain_K", "leak_K",
        "d_M", "gain_M", "leak_M",
        "gain_I", "gain_J",
    ]

    # Per (rotation, drop_rate) → metric values
    trajectories: dict[str, dict[float, dict]] = defaultdict(dict)
    for r in rows:
        kr = _drop_rate_from_variant(r.get("variant", ""))
        if kr is None or kr[0] != args.kind:
            continue
        rid = r.get("rotation_id", "")
        trajectories[rid][kr[1]] = r

    out_rows = []
    for metric in metric_cols:
        all_diffs = []
        n_up = 0
        n_down = 0
        n_flat = 0
        small_step_diffs = []  # |Δ| at smallest transitions (rate step ≤ 25)
        distinct_counts = []
        for rid, by_rate in trajectories.items():
            sorted_rates = sorted(by_rate.keys())
            vals = []
            for rate in sorted_rates:
                v = _finite(by_rate[rate].get(f"min_{metric}"))
                vals.append((rate, v))
            # distinct value count for this rotation
            present = [v for _, v in vals if v is not None]
            if present:
                distinct_counts.append(len(set(round(v, 6) for v in present)))
            # Transitions
            for (r0, v0), (r1, v1) in zip(vals, vals[1:]):
                if v0 is None or v1 is None or r1 <= r0:
                    continue
                diff = v1 - v0
                all_diffs.append(diff)
                if abs(diff) <= 1e-9:
                    n_flat += 1
                elif diff > 0:
                    n_up += 1
                else:
                    n_down += 1
                if (r1 - r0) <= 25.0:
                    small_step_diffs.append(abs(diff))

        n_tot = n_up + n_down + n_flat
        if n_tot == 0:
            continue
        med_step = sorted(small_step_diffs)[len(small_step_diffs) // 2] if small_step_diffs else None
        med_distinct = sorted(distinct_counts)[len(distinct_counts) // 2] if distinct_counts else None
        out_rows.append({
            "metric": metric,
            "n_transitions": n_tot,
            "frac_up": n_up / n_tot,
            "frac_down": n_down / n_tot,
            "frac_flat": n_flat / n_tot,
            "median_small_step_abs_diff": med_step,
            "median_distinct_values_per_rotation": med_distinct,
            "n_rotations": len(distinct_counts),
        })

    if not out_rows:
        print(f"No data to analyse in {input_path}.")
        return

    out_path = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(REPO_ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved: {out_path}")

    print(f"\n[R2 continuity — kind={args.kind}]")
    print(f"  {'metric':14s}  {'n_tr':>5s}  {'frac↑':>6s}  {'frac↓':>6s}  {'frac=':>6s}  {'med|Δ|':>8s}  {'distinct':>8s}")
    for r in out_rows:
        step = f"{r['median_small_step_abs_diff']:.4f}" if r['median_small_step_abs_diff'] is not None else "NA"
        dist = f"{r['median_distinct_values_per_rotation']:>4}" if r['median_distinct_values_per_rotation'] is not None else "NA"
        print(
            f"  {r['metric']:14s}  {r['n_transitions']:>5d}"
            f"  {r['frac_up']:>6.2%}  {r['frac_down']:>6.2%}  {r['frac_flat']:>6.2%}"
            f"  {step:>8s}  {dist:>8s}"
        )


if __name__ == "__main__":
    main()
