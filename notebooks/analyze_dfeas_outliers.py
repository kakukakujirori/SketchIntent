"""Diagnose heavy-tailed d_feas values in the dimensional-dropout sweep.

This is a companion to analyze_dfeas_trajectory.py.  The trajectory evidence is
median/quantile based, but raw means can be dominated by a tiny number of
degenerate dropout variants.  This script identifies those variants, prints the
per-source detail rows that caused them, and recomputes the trajectory after
removing the offending rotations.
"""
from __future__ import annotations

import csv
import math
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCORES = "artifacts/constrained_response_scores_dropout.csv"
DETAILS = "artifacts/constrained_response_details_dropout.csv"


def load_csv(path: str) -> list[dict[str, str]]:
    with open(os.path.join(REPO_ROOT, path), newline="") as f:
        return list(csv.DictReader(f))


def finite_or_none(value) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def quantile(values: list[float], q: float) -> float | None:
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def score_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        r for r in rows
        if r.get("source_validated") == "True"
        and not r.get("error")
        and finite_or_none(r.get("drop_rate")) not in (None, -1.0)
    ]


def top_outliers(rows: list[dict[str, str]], column: str, threshold: float):
    out = []
    for r in score_rows(rows):
        value = finite_or_none(r.get(column))
        if value is not None and value > threshold:
            out.append((value, r))
    return sorted(out, key=lambda item: item[0], reverse=True)


def trajectory(rows: list[dict[str, str]], *, exclude_rotations: set[str] | None = None):
    exclude_rotations = exclude_rotations or set()
    rates = [0.0, 25.0, 50.0, 75.0, 100.0]
    print(f"{'rate':>6s} {'n':>4s} {'mean':>12s} {'p50':>10s} "
          f"{'p75':>10s} {'p90':>10s} {'p95':>10s} {'max':>12s} {'n>10':>5s}")
    print("-" * 88)
    for rate in rates:
        values = []
        for r in score_rows(rows):
            if r.get("rotation_id") in exclude_rotations:
                continue
            if finite_or_none(r.get("drop_rate")) != rate:
                continue
            value = finite_or_none(r.get("min_d_feas"))
            if value is not None:
                values.append(value)
        mean = sum(values) / len(values) if values else None
        print(f"{rate:6.1f} {len(values):4d} {fmt(mean):>12s} "
              f"{fmt(quantile(values, 0.50)):>10s} "
              f"{fmt(quantile(values, 0.75)):>10s} "
              f"{fmt(quantile(values, 0.90)):>10s} "
              f"{fmt(quantile(values, 0.95)):>10s} "
              f"{fmt(max(values) if values else None):>12s} "
              f"{sum(v > 10.0 for v in values):5d}")


def print_detail_rows(details: list[dict[str, str]], rotation_id: str, rate: float):
    print(f"\nDetail rows for {rotation_id} @ drop={rate:g}%")
    matches = [
        r for r in details
        if r.get("source_validated") == "True"
        and r.get("rotation_id") == rotation_id
        and finite_or_none(r.get("drop_rate")) == rate
    ]
    for r in matches:
        print(
            f"  gt={r['gt']:8s} pred={r['pred']:16s} "
            f"d_feas={fmt(finite_or_none(r.get('d_feas'))):>12s} "
            f"gain={fmt(finite_or_none(r.get('d_feas_gain'))):>10s} "
            f"leak={fmt(finite_or_none(r.get('d_feas_leak'))):>12s} "
            f"leak~={fmt(finite_or_none(r.get('d_feas_tilde_leak'))):>10s} "
            f"d_align={fmt(finite_or_none(r.get('d_align'))):>10s} "
            f"d_resp={fmt(finite_or_none(r.get('d_resp'))):>12s} "
            f"n_rel={r.get('n_rel_rows')}/{r.get('n_rel_units')} "
            f"n_dim={r.get('n_dim_pred')}/{r.get('n_dim_gt')}"
        )


def main():
    rows = load_csv(SCORES)
    details = load_csv(DETAILS)

    print("Top summary outliers by min_d_feas")
    print(f"{'value':>12s} {'drop':>6s} {'rotation':70s} {'pred':16s} "
          f"{'leak_med':>12s} {'gain_med':>10s} {'d_resp':>12s}")
    print("-" * 150)
    for value, r in top_outliers(rows, "min_d_feas", threshold=10.0)[:12]:
        print(f"{fmt(value):>12s} {r['drop_rate']:>6s} "
              f"{r['rotation_id'][:70]:70s} {r['pred'][:16]:16s} "
              f"{fmt(finite_or_none(r.get('median_d_feas_leak'))):>12s} "
              f"{fmt(finite_or_none(r.get('median_d_feas_gain'))):>10s} "
              f"{fmt(finite_or_none(r.get('min_d_resp'))):>12s}")

    print("\nTop summary outliers by median_d_feas_leak")
    print(f"{'value':>12s} {'drop':>6s} {'rotation':70s} {'pred':16s} "
          f"{'d_feas':>12s} {'d_resp':>12s}")
    print("-" * 128)
    leak_outliers = top_outliers(rows, "median_d_feas_leak", threshold=100.0)
    for value, r in leak_outliers[:12]:
        print(f"{fmt(value):>12s} {r['drop_rate']:>6s} "
              f"{r['rotation_id'][:70]:70s} {r['pred'][:16]:16s} "
              f"{fmt(finite_or_none(r.get('min_d_feas'))):>12s} "
              f"{fmt(finite_or_none(r.get('min_d_resp'))):>12s}")

    outlier_rotations = {r["rotation_id"] for _, r in top_outliers(
        rows, "min_d_feas", threshold=10.0)}
    leak_rotations = {r["rotation_id"] for _, r in leak_outliers}
    print()
    print(f"Unique rotations with min_d_feas > 10: {len(outlier_rotations)}")
    print(f"Unique rotations with median_d_feas_leak > 100: {len(leak_rotations)}")
    print(f"Union rotations: {len(outlier_rotations | leak_rotations)}")

    for value, r in leak_outliers[:4]:
        print_detail_rows(details, r["rotation_id"], finite_or_none(r["drop_rate"]))

    print("\nTrajectory of min_d_feas, all rows")
    trajectory(rows)
    print("\nTrajectory of min_d_feas, excluding union of outlier rotations")
    trajectory(rows, exclude_rotations=outlier_rotations | leak_rotations)


if __name__ == "__main__":
    main()
