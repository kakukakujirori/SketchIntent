"""FC-matched paired statistics for Ours vs a baseline at one operating point.

Consumes two evaluate.py JSONs that include the per-sketch dumps
(composition.per_sketch and reference.per_sketch). Pairs sketches by their
batch-qualified id, then reports, per metric: n paired, each model's median,
the median paired delta (ours - base), a Wilcoxon signed-rank p-value, and a
bootstrap 95% CI of the mean paired delta.

    conda run -n freecad python paper/scripts/paired_stats_fc_matched.py \
        --ours artifacts/fiverr_eval/persketch_ours_step52200.json \
        --base artifacts/fiverr_eval/persketch_dpc1_step20000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# metric -> (json section, "lower" or "higher" is better)
COMP_METRICS = {"dimensional_ratio": "lower", "raw_distance_ratio": "lower"}
REF_METRICS = {
    "placement_sim": "higher",
    "motion_full_angle": "lower",
    "motion_relational_angle": "lower",
    "leak_dimensional_angle": "lower",
}


def comp_map(d: dict, metric: str) -> dict[str, float]:
    ps = d["composition"].get("per_sketch", {})
    return {k: v[metric] for k, v in ps.items() if metric in v}


def ref_map(d: dict, metric: str) -> dict[str, float]:
    ps = d["reference"].get("per_sketch", [])
    return {r["_id"]: r[metric] for r in ps if metric in r and "_id" in r}


def paired(ours: dict[str, float], base: dict[str, float]):
    ids = sorted(set(ours) & set(base))
    a = np.array([ours[i] for i in ids], float)
    b = np.array([base[i] for i in ids], float)
    return ids, a, b


def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(d: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        if len(nz) == 0:
            return float("nan")
        return float(wilcoxon(nz, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def run(metric: str, direction: str, ours: dict, base: dict) -> str:
    ids, a, b = paired(ours, base)
    if len(ids) == 0:
        return f"{metric:<26} no paired sketches"
    d = a - b  # ours - base
    lo, hi = boot_ci(d)
    p = wilcoxon_p(d)
    mean_d = float(d.mean())
    # Significance: bootstrap CI excludes 0 (mean-based) used as the headline test,
    # since the paired-diff median is often 0 under heavy ties.
    sig = (lo > 0) or (hi < 0)
    favored = (mean_d < 0) == (direction == "lower")
    mark = ("favors Ours" if favored else "favors base") + (" *" if sig else "")
    return (
        f"{metric:<26} n={len(ids):3d}  "
        f"ours_med={np.median(a):.4f}  base_med={np.median(b):.4f}  "
        f"meanΔ={mean_d:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
        f"wilcoxon_p={p:.2e}  [{mark}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    args = ap.parse_args()
    od = json.loads(args.ours.read_text())
    bd = json.loads(args.base.read_text())

    print(f"OURS = {args.ours.name}  (FC {od['solver']['fc_rate']:.3f})")
    print(f"BASE = {args.base.name}  (FC {bd['solver']['fc_rate']:.3f})")
    print("delta = ours - base; Wilcoxon signed-rank two-sided on paired sketches\n")
    for m, dirn in COMP_METRICS.items():
        print(run(m, dirn, comp_map(od, m), comp_map(bd, m)))
    for m, dirn in REF_METRICS.items():
        print(run(m, dirn, ref_map(od, m), ref_map(bd, m)))


if __name__ == "__main__":
    main()
