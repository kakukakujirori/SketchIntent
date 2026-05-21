"""Aggregate per-variant ranking table: fully_constrained_rate, OES, ER, F1,
d_feas, gain, leak.

Merges four CSVs by (rotation_id, pred) when available, otherwise by
(rotation_id, variant):

- `--dfeas-csv` (constrained_response_scores_*.csv): min_d_feas, gain, leak
   for drop_rate=0 (treated as `heldout`) and drop_rate=-1 model rows.
- `--oes-csv` (source_conditioned_dropout_scores.csv): max_histcad_oes,
  median_histcad_er.
- `--controls-csv` (v3_controls_scores.csv): min_d_feas for source_self /
  heldout / dim_drop50 / rel_drop50 etc.
- `--f1-csv` (constraint_token_f1.csv): combined / dim / rel F1.

Fully constrained rate is computed as the fraction of distinct files for
which the named model variant has any F1 row, divided by total distinct
files seen in heldout-drop0 rows. (Model rows are only emitted when the
sketch loads with DoF=2 / Converged, which is the load gate's definition
of fully constrained mod global placement.)

Output: `--out-csv` (default `artifacts/model_ranking_table.csv`).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_csv(path: str) -> list[dict]:
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
    if sv is None:
        return True
    return sv in (True, "True", "true", "1")


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def variant_for_dfeas_row(row: dict) -> str | None:
    try:
        dr = float(row.get("drop_rate", ""))
    except ValueError:
        return None
    if dr == -1.0:
        return row.get("pred")
    if dr == 0.0:
        return "heldout"
    return None


def variant_for_oes_row(row: dict) -> str | None:
    return variant_for_dfeas_row(row)


def variant_for_controls_row(row: dict) -> str | None:
    v = row.get("variant")
    if not v:
        return None
    return v


def variant_for_f1_row(row: dict) -> str | None:
    return variant_for_dfeas_row(row)


def aggregate(rows: list[dict],
              variant_fn,
              fields: list[str],
              eps_filter: float | None = None) -> dict[str, dict[str, float]]:
    """Group rows by variant and compute medians of named fields."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    file_count: dict[str, set] = defaultdict(set)
    for r in rows:
        if not validated(r):
            continue
        if eps_filter is not None and r.get("eps") not in (None, ""):
            try:
                if abs(float(r["eps"]) - eps_filter) > 1e-18:
                    continue
            except ValueError:
                continue
        v = variant_fn(r)
        if v is None:
            continue
        bf = r.get("batch"), r.get("filename")
        if bf != (None, None):
            file_count[v].add(bf)
        for fld in fields:
            x = r.get(fld)
            if _is_finite(x):
                buckets[v][fld].append(float(x))
    out: dict[str, dict[str, float]] = {}
    for v, by_field in buckets.items():
        out[v] = {fld: median(vs) for fld, vs in by_field.items()}
        out[v]["_n_rows"] = max(len(vs) for vs in by_field.values()) if by_field else 0
        out[v]["_files"] = len(file_count[v])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dfeas-csv",
                   default="artifacts/constrained_response_scores_full.csv")
    p.add_argument("--oes-csv",
                   default="artifacts/source_conditioned_dropout_scores.csv")
    p.add_argument("--controls-csv",
                   default="artifacts/v3_controls_scores.csv")
    p.add_argument("--f1-csv",
                   default="artifacts/constraint_token_f1.csv")
    p.add_argument("--eps", type=float, default=1e-9,
                   help="eps filter for controls CSV rows.")
    p.add_argument("--out-csv",
                   default="artifacts/model_ranking_table.csv")
    args = p.parse_args()

    dfeas_rows = load_csv(args.dfeas_csv)
    oes_rows = load_csv(args.oes_csv)
    controls_rows = load_csv(args.controls_csv)
    f1_rows = load_csv(args.f1_csv)

    dfeas_agg = aggregate(
        dfeas_rows, variant_for_dfeas_row,
        ["min_d_feas", "median_d_feas_gain", "median_d_feas_leak"],
    )
    oes_agg = aggregate(
        oes_rows, variant_for_oes_row,
        ["max_histcad_oes", "median_histcad_er"],
    )
    controls_agg = aggregate(
        controls_rows, variant_for_controls_row,
        ["min_d_feas", "median_d_feas_gain", "median_d_feas_leak"],
        eps_filter=args.eps,
    )
    f1_agg = aggregate(
        f1_rows, variant_for_f1_row,
        ["f1", "dim_f1", "rel_f1"],
    )

    # Universe of variants from all sources.
    all_variants: list[str] = []
    for src in (dfeas_agg, oes_agg, controls_agg, f1_agg):
        for v in src:
            if v not in all_variants:
                all_variants.append(v)
    preferred = ["heldout", "source_self", "dim_drop50", "dim_drop100",
                 "rel_drop50", "rel_drop100"]
    head = [v for v in preferred if v in all_variants]
    rest = sorted(v for v in all_variants if v not in head)
    ordering = head + rest

    # Fully constrained rate: model_files / heldout_files
    heldout_files = f1_agg.get("heldout", {}).get("_files", 0) \
        or dfeas_agg.get("heldout", {}).get("_files", 0)

    full = args.out_csv if os.path.isabs(args.out_csv) \
        else os.path.join(REPO_ROOT, args.out_csv)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "n_files",
            "fully_constrained_rate",
            "OES", "ER",
            "F1", "dim_F1", "rel_F1",
            "d_feas", "d_feas_gain", "d_feas_leak",
        ])
        for v in ordering:
            d_d = dfeas_agg.get(v, {})
            d_o = oes_agg.get(v, {})
            d_c = controls_agg.get(v, {})
            d_f = f1_agg.get(v, {})
            n_files = max(
                int(d_d.get("_files", 0) or 0),
                int(d_o.get("_files", 0) or 0),
                int(d_c.get("_files", 0) or 0),
                int(d_f.get("_files", 0) or 0),
            )
            if v in ("heldout", "source_self") or v.startswith("dim_drop") \
                    or v.startswith("rel_drop"):
                fc_rate = ""
            elif heldout_files:
                fc_rate = (d_f.get("_files", 0) or d_d.get("_files", 0)) / heldout_files
            else:
                fc_rate = ""
            # Prefer controls-side d_feas for variants that exist there (more
            # comprehensive variants like source_self, dim_drop*, rel_drop*);
            # fall back to drop-rate-based dfeas CSV.
            d_feas_val = d_c.get("min_d_feas") if "min_d_feas" in d_c else d_d.get("min_d_feas")
            gain_val = d_c.get("median_d_feas_gain") if "median_d_feas_gain" in d_c \
                else d_d.get("median_d_feas_gain")
            leak_val = d_c.get("median_d_feas_leak") if "median_d_feas_leak" in d_c \
                else d_d.get("median_d_feas_leak")
            w.writerow([
                v, n_files,
                fc_rate,
                d_o.get("max_histcad_oes"),
                d_o.get("median_histcad_er"),
                d_f.get("f1"),
                d_f.get("dim_f1"),
                d_f.get("rel_f1"),
                d_feas_val,
                gain_val,
                leak_val,
            ])
    print(f"Saved: {full}")


if __name__ == "__main__":
    main()
