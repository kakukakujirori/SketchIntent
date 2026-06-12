from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import norm, wilcoxon

from _bootstrap import bootstrap

bootstrap()

from fusion360_annotation.eda_code.runtime_warnings import MODEL_LABELS, has_model_rows  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

ANGLES = ["theta_full", "theta_rel"]
SIMS = ["asgn_sim", "asgn_min", "set_sim", "omp_sim"]
METRICS = ANGLES + SIMS
VARIANT_ORDER = [
    "source_self",
    "heldout",
    "dim_drop25",
    "dim_drop50",
    "dim_drop75",
    "dim_drop100",
    "rel_drop50",
    "rel_drop100",
    "base",
    "solver_rl",
]
MODELS = ["base", "solver_rl"]
PAPER_VARIANTS = [
    ("source_self", "Source-self"),
    ("heldout", "Human held-out"),
    ("dim_drop25", "Dim. drop 25%"),
    ("dim_drop50", "Dim. drop 50%"),
    ("dim_drop75", "Dim. drop 75%"),
    ("dim_drop100", "Dim. drop 100%"),
    ("rel_drop50", "Rel. drop 50%"),
    ("rel_drop100", "Rel. drop 100%"),
    ("base", "base Vitruvion"),
    ("solver_rl", "solver_rl"),
]


def read_csvs(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_model_status.csv"):
            continue
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    return rows


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aggregate(rows, metrics, best=True):
    acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["rid"], row["variant"])
        for metric in metrics:
            v = fnum(row.get(metric))
            if v is not None:
                acc[key][metric].append(v)
    out = {}
    for key, values_by_metric in acc.items():
        out[key] = {}
        for metric, vals in values_by_metric.items():
            if best:
                out[key][metric] = min(vals) if metric in ANGLES else max(vals)
            else:
                out[key][metric] = st.fmean(vals)
    return out


def median_table(lv, metrics, title):
    print(f"\n=== {title} ===")
    print(f"  {'variant':14s} {'n':>5s}" + "".join(f" {m:>11s}" for m in metrics))
    for var in VARIANT_ORDER:
        vals_by_m = []
        n = 0
        for metric in metrics:
            xs = [d[metric] for (_rid, vv), d in lv.items() if vv == var and metric in d]
            n = max(n, len(xs))
            vals_by_m.append(f" {st.median(xs):11.3f}" if xs else f" {'NA':>11s}")
        if n:
            print(f"  {var:14s} {n:5d}" + "".join(vals_by_m))


def metric_values(lv, variant: str, metric: str) -> list[float]:
    return [
        d[metric]
        for (_rid, vv), d in lv.items()
        if vv == variant and metric in d
    ]


def metric_median(lv, variant: str, metric: str) -> float | None:
    xs = metric_values(lv, variant, metric)
    return st.median(xs) if xs else None


def fmt_metric(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}"


def print_paper_table(motion_rows: list[dict], oes_rows: list[dict], token_rows: list[dict]) -> None:
    motion_best = aggregate(motion_rows, METRICS, best=True) if motion_rows else {}
    oes_best = aggregate(oes_rows, ["oes"], best=True) if oes_rows else {}
    token_best = aggregate(token_rows, ["token_f1"], best=True) if token_rows else {}
    columns = [
        ("theta_full", motion_best, "theta_full"),
        ("theta_rel", motion_best, "theta_rel"),
        ("S_place", motion_best, "asgn_sim"),
        ("OES", oes_best, "oes"),
        ("F1", token_best, "token_f1"),
    ]

    print("\n=== paper table (multi-ref medians; columns match paper/sec/5_experiments.tex) ===")
    print(
        f"  {'Variant':18s} {'n':>5s}"
        f" {'theta_full':>11s} {'theta_rel':>11s} {'S_place':>9s}"
        f" {'OES':>8s} {'F1':>8s}"
    )
    for variant, label in PAPER_VARIANTS:
        n = max(
            len(metric_values(motion_best, variant, metric))
            for metric in ("theta_full", "theta_rel", "asgn_sim")
        )
        if n == 0:
            n = max(
                len(metric_values(lv, variant, metric))
                for _name, lv, metric in columns
            )
        values = [
            fmt_metric(metric_median(lv, variant, metric))
            for _name, lv, metric in columns
        ]
        print(f"  {label:18s} {n:5d}" + "".join(f" {v:>11s}" for v in values[:3]) + "".join(f" {v:>8s}" for v in values[3:]))
    print("  note: n is the motion count when available; OES/F1 use their available rows.")


def paired_tests(lv, metrics, vs="heldout", title=""):
    print(f"\n=== paired Wilcoxon vs {vs}{title} (z>0 = human favored) ===")
    for model in MODELS:
        held = {rid: d for (rid, vv), d in lv.items() if vv == vs}
        mod = {rid: d for (rid, vv), d in lv.items() if vv == model}
        line = f"  {model:10s}"
        for metric in metrics:
            common = [r for r in held if r in mod and metric in held[r] and metric in mod[r]]
            sgn = +1 if metric in ANGLES else -1
            diffs = [sgn * (mod[r][metric] - held[r][metric]) for r in common]
            nz = [d for d in diffs if abs(d) > 1e-12]
            hb = sum(1 for d in diffs if d > 1e-12)
            mb = sum(1 for d in diffs if d < -1e-12)
            if len(nz) < 8:
                line += f"  {metric}:NA"
                continue
            _stat, p = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
            z = norm.isf(p / 2) * (1 if st.fmean(diffs) > 0 else -1)
            line += f"  {metric}: n={len(common)} h={hb} m={mb} z={z:+.2f} p={p:.1e}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    args = ap.parse_args()
    rd = args.out_dir

    motion = read_csvs(os.path.join(rd, "motion_eval*.csv"))
    oes = read_csvs(os.path.join(rd, "oes_eval*.csv"))
    token = read_csvs(os.path.join(rd, "token_f1*.csv"))
    if motion and not has_model_rows(motion):
        print(
            f"[warn] motion_eval*.csv contains no model variant rows ({', '.join(MODEL_LABELS)}); "
            "model comparison sections will be NA.",
            file=sys.stderr,
        )
    if oes and not has_model_rows(oes):
        print(
            f"[warn] oes_eval*.csv contains no model variant rows ({', '.join(MODEL_LABELS)}); "
            "OES model comparison sections will be empty.",
            file=sys.stderr,
        )
    if token and not has_model_rows(token):
        print(
            f"[warn] token_f1*.csv contains no model variant rows ({', '.join(MODEL_LABELS)}); "
            "paper-table model F1 cells will be empty.",
            file=sys.stderr,
        )
    if motion or oes or token:
        print_paper_table(motion, oes, token)

    if motion:
        print(
            f"[motion] rows={len(motion)}  "
            f"rotations={len({r['rid'] for r in motion})}  "
            f"files={len({(r['batch'], r['filename']) for r in motion})}"
        )
        lv_best = aggregate(motion, METRICS, best=True)
        median_table(lv_best, ["theta_full", "theta_rel", "asgn_sim"], "controls (multi-ref best, ours only)")
        paired_tests(lv_best, ["theta_full", "theta_rel", "asgn_sim"])
        median_table(lv_best, SIMS, "placement variants (multi-ref best, appendix)")
        lv_mean = aggregate(motion, METRICS, best=False)
        median_table(lv_mean, ["theta_full", "theta_rel", "asgn_sim"], "controls (mean-ref, appendix)")
        floor = [d["asgn_sim"] for (_rid, vv), d in lv_best.items() if vv == "heldout" and "asgn_sim" in d]
        if floor:
            print(f"\nasgn_sim human floor: median={st.median(floor):.3f} mean={st.fmean(floor):.3f} (n={len(floor)})")

    if oes:
        print(f"\n[oes] rows={len(oes)}")
        acc = defaultdict(list)
        for row in oes:
            v = fnum(row.get("oes"))
            if v is not None:
                acc[(row["rid"], row["variant"])].append(v)
        lv_max = {k: {"oes": max(v)} for k, v in acc.items()}
        print("\n=== OES dropout sweep (median of multi-ref max) ===")
        for var in [
            "source_self",
            "heldout",
            "dim_drop25",
            "dim_drop50",
            "dim_drop75",
            "dim_drop100",
            "rel_drop50",
            "rel_drop100",
            "base",
            "solver_rl",
        ]:
            xs = [d["oes"] for (_rid, vv), d in lv_max.items() if vv == var]
            if xs:
                print(f"  {var:14s} n={len(xs):4d} median={st.median(xs):.3f} mean={st.fmean(xs):.3f}")

        print("\n=== OES paired: model vs heldout (z>0 = human favored) ===")
        for model in MODELS:
            held = {rid: d["oes"] for (rid, vv), d in lv_max.items() if vv == "heldout"}
            mod = {rid: d["oes"] for (rid, vv), d in lv_max.items() if vv == model}
            common = sorted(set(held) & set(mod))
            diffs = [held[r] - mod[r] for r in common]
            nz = [d for d in diffs if abs(d) > 1e-12]
            hb = sum(1 for d in diffs if d > 1e-12)
            mb = sum(1 for d in diffs if d < -1e-12)
            if len(nz) >= 8:
                _s, p = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
                z = norm.isf(p / 2) * (1 if st.fmean(diffs) > 0 else -1)
                print(f"  {model:10s} n={len(common)} human-better={hb} model-better={mb} z={z:+.2f} p={p:.1e}")

    status_files = glob.glob(os.path.join(rd, "motion_eval*_model_status.csv"))
    fc_files = {m: set() for m in MODELS}
    if status_files:
        srows = []
        for path in status_files:
            with open(path) as f:
                srows.extend(csv.DictReader(f))
        seen = {(r["batch"], r["filename"], r["label"]): r for r in srows}
        print("\n=== model admission / FC (DoF<=2) rates ===")
        if not any(label in MODEL_LABELS for (_batch, _filename, label) in seen):
            print(f"  no model status rows found for {', '.join(MODEL_LABELS)}")
        for model in MODELS:
            rs = [(k, r) for k, r in seen.items() if k[2] == model]
            n = len(rs)
            loaded = [r for _k, r in rs if r["status"] == "loaded"]
            missing = sum(1 for _k, r in rs if r["status"] == "missing")
            fc = 0
            for k, r in rs:
                if r["dof"] not in ("", "None", None) and float(r["dof"]) <= 2:
                    fc += 1
                    fc_files[model].add((k[0], k[1]))
            if n:
                print(f"  {model:10s} files={n} loaded={len(loaded)} ({100 * len(loaded) / n:.1f}%) missing_pred={missing} FC(DoF<=2)={fc} ({100 * fc / n:.1f}%)")

    if motion and status_files:
        print("\n=== gate sensitivity (theta_full / theta_rel / asgn_sim medians) ===")
        lv_best = aggregate(motion, METRICS, best=True)
        rid_file = {r["rid"]: (r["batch"], r["filename"]) for r in motion}
        for model in MODELS:
            rows_m = {rid: d for (rid, vv), d in lv_best.items() if vv == model}
            strict = {rid: d for rid, d in rows_m.items() if rid_file.get(rid) in fc_files[model]}
            for tag, sel in (("relaxed", rows_m), ("strict", strict)):
                meds = []
                for metric in ["theta_full", "theta_rel", "asgn_sim"]:
                    xs = [d[metric] for d in sel.values() if metric in d]
                    meds.append(f"{metric}={st.median(xs):.3f}" if xs else f"{metric}=NA")
                print(f"  {model:10s} {tag:8s} n={len(sel):4d}  " + "  ".join(meds))

    eps_dir = os.path.join(rd, "eps_sweep")
    if os.path.isdir(eps_dir) and motion:
        print("\n=== eps robustness (batches subset; medians per variant) ===")
        sub_batches = None
        for path in sorted(glob.glob(os.path.join(eps_dir, "*.csv"))):
            if path.endswith("_model_status.csv"):
                continue
            rows_e = list(csv.DictReader(open(path)))
            if not rows_e:
                continue
            if sub_batches is None:
                sub_batches = {r["batch"] for r in rows_e}
                base_rows = [r for r in motion if r["batch"] in sub_batches]
                lv0 = aggregate(base_rows, METRICS, best=True)
                line = "  eps=1e-9   "
                for var in ("heldout", "base", "solver_rl"):
                    for metric in ("theta_full", "theta_rel", "asgn_sim"):
                        xs = [d[metric] for (_rid, vv), d in lv0.items() if vv == var and metric in d]
                        if xs:
                            line += f" {var[:4]}.{metric[6:10]}={st.median(xs):.4f}"
                print(line)
            lv_e = aggregate(rows_e, METRICS, best=True)
            tag = os.path.basename(path).replace("motion_eval_", "").replace(".csv", "")
            line = f"  {tag:10s} "
            for var in ("heldout", "base", "solver_rl"):
                for metric in ("theta_full", "theta_rel", "asgn_sim"):
                    xs = [d[metric] for (_rid, vv), d in lv_e.items() if vv == var and metric in d]
                    if xs:
                        line += f" {var[:4]}.{metric[6:10]}={st.median(xs):.4f}"
            print(line)

    agree = read_csvs(os.path.join(rd, "agreement.csv"))
    if agree:
        jac = [v for v in (fnum(r["jaccard"]) for r in agree) if v is not None]
        f1s = [v for v in (fnum(r["token_f1"]) for r in agree) if v is not None]
        print(f"\n=== human agreement (pairs={len(agree)}) ===")
        print(f"  placement Jaccard mean={st.fmean(jac):.3f} median={st.median(jac):.3f}")
        print(f"  token F1 mean={st.fmean(f1s):.3f} median={st.median(f1s):.3f}")


if __name__ == "__main__":
    main()
