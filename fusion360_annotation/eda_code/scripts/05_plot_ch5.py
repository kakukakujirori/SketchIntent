from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.config import DIMENSIONAL_DISPLAY_TYPES

HERE = Path(__file__).resolve().parents[1]
REL_COLOR = "#2c7fb8"
DIM_COLOR = "#d95f0e"


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) * sx * sy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--y", default="d_sub_angle", choices=["d_sub_angle", "d_H_angle"])
    ap.add_argument("--layout", default="wide", choices=["wide", "column"])
    args = ap.parse_args()
    out_dir = Path(args.out)
    rows = json.loads((out_dir / "ch5_pairs.json").read_text())

    pts = [(r["token_f1"], r[args.y]) for r in rows if r.get("token_f1") is not None and r.get(args.y) is not None]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(pts)
    r = pearson(xs, ys)
    ang_lo, f1_lo = 0.1, 0.5
    low_f1 = [p for p in pts if p[0] < f1_lo]
    frac = sum(p[1] <= ang_lo for p in low_f1) / len(low_f1) if low_f1 else float("nan")

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for row in rows:
        for t, (na, nb, nm) in (row.get("per_type") or {}).items():
            agg[t][0] += na
            agg[t][1] += nb
            agg[t][2] += nm
    type_f1 = {}
    for t, (sa, sb, sm) in agg.items():
        type_f1[t] = (2.0 * sm / (sa + sb)) if (sa + sb) > 0 else 0.0
    rel_types = sorted([t for t in type_f1 if t not in DIMENSIONAL_DISPLAY_TYPES], key=lambda t: -(agg[t][0] + agg[t][1]))
    dim_types = sorted([t for t in type_f1 if t in DIMENSIONAL_DISPLAY_TYPES], key=lambda t: -(agg[t][0] + agg[t][1]))
    ordered = rel_types + dim_types

    column = args.layout == "column"
    if column:
        fig, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(3.5, 4.6), gridspec_kw={"height_ratios": [1.0, 0.85]})
        fs_label, fs_title, fs_tick, fs_box, fs_bar = 8, 8.5, 6.5, 7, 0
        dot = 6
    else:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
        fs_label, fs_title, fs_tick, fs_box, fs_bar = 10, 10, 8, 9, 6
        dot = 12

    ax_l.scatter(xs, ys, s=dot, alpha=0.40, color=REL_COLOR, edgecolor="none")
    ax_l.axhline(ang_lo, color="gray", lw=0.8, ls="--")
    ax_l.axvline(f1_lo, color="gray", lw=0.8, ls="--")
    ax_l.set_xlabel("Inter-annotator token-F1  (surface agreement)", fontsize=fs_label)
    ax_l.set_ylabel(r"$\theta_{\mathrm{motion}}^{\mathrm{full}}$  (motion disagreement)" if column else r"$\theta_{\mathrm{motion}}^{\mathrm{full}}$  (motion-subspace disagreement)", fontsize=fs_label)
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(-0.02, max(0.5, max(ys) * 1.05))
    ax_l.tick_params(labelsize=fs_tick)
    ax_l.text(0.03, 0.97, f"n={n} pairs\nPearson r={r:.2f}", transform=ax_l.transAxes, va="top", ha="left", fontsize=fs_box, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
    ax_l.set_title("(a) token agreement vs motion agreement", fontsize=fs_title)
    ax_l.spines[["top", "right"]].set_visible(False)

    gap = 1
    x = list(range(len(rel_types))) + [len(rel_types) - 1 + gap + 1 + i for i in range(len(dim_types))]
    vals = [type_f1[t] for t in ordered]
    colors = [REL_COLOR] * len(rel_types) + [DIM_COLOR] * len(dim_types)
    ax_r.bar(x, vals, color=colors, width=0.8)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(ordered, rotation=45, ha="right", fontsize=fs_tick)
    ax_r.set_ylabel("Per-type agreement (micro-F1)" if column else "Inter-annotator agreement (micro-F1)", fontsize=fs_label)
    ax_r.set_ylim(0, 1.0)
    ax_r.tick_params(axis="y", labelsize=fs_tick)
    if fs_bar:
        for xi, v in zip(x, vals):
            ax_r.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=fs_bar)
    ax_r.set_title("(b) per-type inter-annotator agreement", fontsize=fs_title)
    ax_r.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    suffix = "_column" if column else ""
    pdf = out_dir / f"ch5_token_vs_motion{suffix}.pdf"
    png = out_dir / f"ch5_token_vs_motion{suffix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    stats = {
        "n_pairs_plotted": n,
        "y_metric": args.y,
        "pearson_r": r,
        "token_f1_mean": st.mean(xs),
        f"{args.y}_mean": st.mean(ys),
        f"{args.y}_median": st.median(ys),
        f"{args.y}_frac_near_zero(<= {ang_lo})": sum(y <= ang_lo for y in ys) / n,
        "low_f1_and_motion_agree_frac": frac,
        "per_type_f1": {t: type_f1[t] for t in ordered},
    }
    (out_dir / "ch5_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved {pdf}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
