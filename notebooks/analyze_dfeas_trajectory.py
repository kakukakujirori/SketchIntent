"""d_feas trajectory analysis on heldout dim-dropout sweep.

Mirrors analyze_dalign_trajectory_deep.py but for d_feas + decomposition
(gain block and leak block). Splits rotations into:
  - discriminative subset:    d_align(drop=0) > 1e-6
  - humans-aligned subset:    d_align(drop=0) ≤ 1e-6

Reports per-rate mean / quantile of d_feas, d_feas_gain, d_feas_leak,
HistCAD OES, attain_err. Verifies whether d_feas monotonically rises with
drop_rate (target property) and whether the gain block is the driver.
"""
from __future__ import annotations

import csv
import math
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_csv(p):
    with open(os.path.join(REPO_ROOT, p)) as f:
        return list(csv.DictReader(f))


def finite_or_none(s):
    if s in (None, "", "None"):
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def collect(rows, validated=True):
    out = {}
    for r in rows:
        if r.get("error") or (r.get("source_validated") == "True") != validated:
            continue
        rate = float(r["drop_rate"])
        if rate == -1:
            continue
        rid = r["rotation_id"]
        rd = out.setdefault(rid, {})
        rd[rate] = {
            "d_align": finite_or_none(r.get("min_d_align")),
            "d_feas": finite_or_none(r.get("min_d_feas")),
            "d_feas_tilde": finite_or_none(r.get("min_d_feas_tilde")),
            "d_feas_gain": finite_or_none(r.get("median_d_feas_gain")),
            "d_feas_leak": finite_or_none(r.get("median_d_feas_leak")),
            "d_feas_tilde_leak": finite_or_none(r.get("median_d_feas_tilde_leak")),
            "attain": finite_or_none(r.get("median_attain_err")),
            "d_resp": finite_or_none(r.get("min_d_resp")),
        }
    return out


def collect_hist(rows, validated=True):
    out = {}
    for r in rows:
        if (r.get("source_validated") == "True") != validated:
            continue
        rate = float(r["drop_rate"])
        if rate == -1:
            continue
        rid = r["rotation_id"]
        rd = out.setdefault(rid, {})
        rd[rate] = {"oes": finite_or_none(r.get("max_histcad_oes"))}
    return out


def filter_discriminative(rid, by_rot, thresh=1e-6):
    rd = by_rot.get(rid, {})
    if 0.0 not in rd:
        return False
    v = rd[0.0]["d_align"]
    return v is not None and v > thresh


def filter_aligned(rid, by_rot, thresh=1e-6):
    rd = by_rot.get(rid, {})
    if 0.0 not in rd:
        return False
    v = rd[0.0]["d_align"]
    return v is not None and v <= thresh


def report(rates, by_rot, by_rot_hist, label, filter_fn=None):
    print()
    print(f"=== {label} ===")
    n_filt = sum(1 for rid in by_rot if filter_fn is None or filter_fn(rid, by_rot))
    print(f"  rotations = {n_filt}")
    print()
    print(f"{'rate':>6s} | {'mean_feas':>9s} {'p50':>7s} {'p75':>7s} {'p90':>7s} "
          f"{'max':>7s} | {'mean_gain':>9s} {'mean_leak':>9s} | "
          f"{'mean_OES':>9s} {'mean_attain':>11s} {'mean_d_resp':>11s}")
    print("-" * 117)
    for rate in rates:
        feas = []
        gains = []
        leaks = []
        oeses = []
        attains = []
        resps = []
        for rid, rd in by_rot.items():
            if filter_fn and not filter_fn(rid, by_rot):
                continue
            if rate not in rd:
                continue
            v = rd[rate]
            if v["d_feas"] is not None:
                feas.append(v["d_feas"])
            if v["d_feas_gain"] is not None:
                gains.append(v["d_feas_gain"])
            if v["d_feas_leak"] is not None:
                leaks.append(v["d_feas_leak"])
            if v["attain"] is not None:
                attains.append(v["attain"])
            if v["d_resp"] is not None:
                resps.append(v["d_resp"])
            h = by_rot_hist.get(rid, {}).get(rate)
            if h and h["oes"] is not None:
                oeses.append(h["oes"])

        def mean(xs): return sum(xs) / len(xs) if xs else None
        def f(v): return f"{v:9.4f}" if v is not None else "    -    "
        print(f"{rate:>6.1f} | {f(mean(feas))} "
              f"{f(quantile(feas, 0.5))} "
              f"{f(quantile(feas, 0.75))} "
              f"{f(quantile(feas, 0.90))} "
              f"{f(max(feas) if feas else None)} | "
              f"{f(mean(gains))} "
              f"{f(mean(leaks))} | "
              f"{f(mean(oeses))} "
              f"  {f(mean(attains)).strip():>11s}  "
              f"{f(mean(resps)).strip():>11s}")


def monotonicity(by_rot, rates, key, filter_fn=None, eps=1e-9):
    print()
    print(f"Within-rotation monotonicity of {key}:")
    for r1, r2 in zip(rates, rates[1:]):
        up = down = tie = 0
        for rid, rd in by_rot.items():
            if filter_fn and not filter_fn(rid, by_rot):
                continue
            v1 = rd.get(r1, {}).get(key)
            v2 = rd.get(r2, {}).get(key)
            if v1 is None or v2 is None:
                continue
            if v2 > v1 + eps:
                up += 1
            elif v2 < v1 - eps:
                down += 1
            else:
                tie += 1
        total = up + down + tie
        print(f"  {r1:>5.1f}% -> {r2:>5.1f}%: up={up:>3d} "
              f"down={down:>3d} tie={tie:>3d} (n={total}, "
              f"up%={100*up/max(total,1):.1f})")


def main():
    rows = load_csv("artifacts/constrained_response_scores_dropout.csv")
    hist = load_csv("artifacts/source_conditioned_dropout_scores.csv")
    by_rot = collect(rows, validated=True)
    by_rot_hist = collect_hist(hist, validated=True)
    rates = [0.0, 25.0, 50.0, 75.0, 100.0]

    n_total = len(by_rot)
    n_disc = sum(1 for rid in by_rot if filter_discriminative(rid, by_rot))
    n_align = sum(1 for rid in by_rot if filter_aligned(rid, by_rot))
    print(f"Total validated rotations: {n_total}")
    print(f"  discriminative (d_align(0)>1e-6): {n_disc}")
    print(f"  humans-aligned  (d_align(0)≤1e-6): {n_align}")

    report(rates, by_rot, by_rot_hist, "ALL rotations")
    monotonicity(by_rot, rates, "d_feas")
    monotonicity(by_rot, rates, "d_align")

    report(rates, by_rot, by_rot_hist, "discriminative (d_align(0)>1e-6)",
           filter_fn=filter_discriminative)
    monotonicity(by_rot, rates, "d_feas", filter_fn=filter_discriminative)
    monotonicity(by_rot, rates, "d_align", filter_fn=filter_discriminative)

    report(rates, by_rot, by_rot_hist, "humans-aligned (d_align(0)≤1e-6)",
           filter_fn=filter_aligned)
    monotonicity(by_rot, rates, "d_feas", filter_fn=filter_aligned)
    monotonicity(by_rot, rates, "d_align", filter_fn=filter_aligned)


if __name__ == "__main__":
    main()
