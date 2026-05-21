"""Find rotations where HistCAD OES and `d_feas` disagree strongly.

We join two CSVs by (rotation_id, pred) and surface two failure types:

1. high-OES / high-d_feas: model edits succeed under HistCAD's binary
   preservation check but `d_feas` flags relational leakage. Most defensible
   "OES misses what d_feas catches" qualitative examples.
2. low-OES / low-d_feas: HistCAD fails (edit conflict / insertion failure)
   but `d_feas` is small. Often fully constrained sketches that OES unfairly
   penalises.

Output: top-K rows per category, plus a markdown bullet summary.
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


def load_validated(path: str) -> list[dict]:
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    if not os.path.exists(full):
        print(f"  [warn] missing: {full}")
        return []
    with open(full, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("source_validated") in ("True", "true", "1", True)]
    print(f"  loaded {len(rows):5d} validated rows from {full}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dfeas-csv",
                   default="artifacts/constrained_response_scores_full.csv")
    p.add_argument("--oes-csv",
                   default="artifacts/source_conditioned_dropout_scores.csv")
    p.add_argument("--out-csv",
                   default="artifacts/dfeas_oes_divergent_cases.csv")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--high-oes-threshold", type=float, default=0.7)
    p.add_argument("--high-dfeas-threshold", type=float, default=1.0)
    p.add_argument("--low-oes-threshold", type=float, default=0.2)
    p.add_argument("--low-dfeas-threshold", type=float, default=0.1)
    p.add_argument("--pred", default=None,
                   help="Only consider rows where pred matches this label "
                        "(e.g. 'base' / 'solver_rl' / 'Vitruvion'). "
                        "Default: all model preds (drop_rate=-1).")
    args = p.parse_args()

    d_rows = load_validated(args.dfeas_csv)
    o_rows = load_validated(args.oes_csv)

    by_key_d: dict[tuple, dict] = {}
    for r in d_rows:
        if r.get("drop_rate") not in ("-1", "-1.0"):
            continue
        if args.pred and r.get("pred") != args.pred:
            continue
        by_key_d[(r["rotation_id"], r.get("pred"))] = r

    by_key_o: dict[tuple, dict] = {}
    for r in o_rows:
        if r.get("drop_rate") not in ("-1", "-1.0"):
            continue
        if args.pred and r.get("pred") != args.pred:
            continue
        by_key_o[(r["rotation_id"], r.get("pred"))] = r

    joined: list[dict] = []
    for key, dr in by_key_d.items():
        oer = by_key_o.get(key)
        if oer is None:
            continue
        d_feas = _f(dr.get("min_d_feas"))
        oes = _f(oer.get("max_histcad_oes"))
        gain = _f(dr.get("median_d_feas_gain"))
        leak = _f(dr.get("median_d_feas_leak"))
        if d_feas is None or oes is None:
            continue
        joined.append({
            "rotation_id": key[0], "pred": key[1],
            "batch": dr.get("batch"), "filename": dr.get("filename"),
            "refs": dr.get("refs"), "heldout": dr.get("heldout"),
            "d_feas": d_feas, "oes": oes,
            "gain": gain, "leak": leak,
        })
    print(f"Joined cases: {len(joined)}")

    high = [r for r in joined
            if r["oes"] >= args.high_oes_threshold
            and r["d_feas"] >= args.high_dfeas_threshold]
    low = [r for r in joined
           if r["oes"] <= args.low_oes_threshold
           and r["d_feas"] <= args.low_dfeas_threshold]

    high.sort(key=lambda r: (-(r["oes"] + r["d_feas"]), r["rotation_id"]))
    low.sort(key=lambda r: ((r["oes"] - r["d_feas"]), r["rotation_id"]))

    out_full = args.out_csv if os.path.isabs(args.out_csv) \
        else os.path.join(REPO_ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    with open(out_full, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "rank", "batch", "filename", "rotation_id",
                    "refs", "heldout", "pred", "OES", "d_feas", "gain", "leak"])
        for rank, r in enumerate(high[: args.top_k], start=1):
            w.writerow(["high_OES_high_dfeas", rank, r["batch"], r["filename"],
                        r["rotation_id"], r["refs"], r["heldout"], r["pred"],
                        r["oes"], r["d_feas"], r["gain"], r["leak"]])
        for rank, r in enumerate(low[: args.top_k], start=1):
            w.writerow(["low_OES_low_dfeas", rank, r["batch"], r["filename"],
                        r["rotation_id"], r["refs"], r["heldout"], r["pred"],
                        r["oes"], r["d_feas"], r["gain"], r["leak"]])
    print(f"Saved: {out_full}")

    print("\nTop high-OES / high-d_feas (OES finds edit success, d_feas finds leak):")
    for r in high[: min(args.top_k, 5)]:
        print(f"  {r['batch']}/{r['filename']:40s} {r['heldout']:8s} pred={r['pred']:10s}"
              f"  OES={r['oes']:.3f}  d_feas={r['d_feas']:.3f}"
              f"  gain={r['gain']!r}  leak={r['leak']!r}")
    print("\nTop low-OES / low-d_feas (OES penalises fully constrained, d_feas is OK):")
    for r in low[: min(args.top_k, 5)]:
        print(f"  {r['batch']}/{r['filename']:40s} {r['heldout']:8s} pred={r['pred']:10s}"
              f"  OES={r['oes']:.3f}  d_feas={r['d_feas']:.3f}")


if __name__ == "__main__":
    main()
