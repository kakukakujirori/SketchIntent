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
from matplotlib.patches import Patch  # noqa: E402

from src.eval.config import DIMENSIONAL_DISPLAY_TYPES

HERE = Path(__file__).resolve().parents[1]
REL_COLOR = "#2c7fb8"
DIM_COLOR = "#d95f0e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    recs = json.loads((out_dir / "records.json").read_text())
    n_sketches = len({r["sketch_id"] for r in recs})

    con = collections.Counter()
    rel = dim = 0
    for r in recs:
        for k, v in r["constraint_counts"].items():
            con[k] += v
        rel += r["n_relational"]
        dim += r["n_dimensional"]
    total = sum(con.values())

    rel_types = sorted([k for k in con if k not in DIMENSIONAL_DISPLAY_TYPES], key=lambda k: con[k])
    dim_types = sorted([k for k in con if k in DIMENSIONAL_DISPLAY_TYPES], key=lambda k: con[k])
    ordered = rel_types + dim_types
    rel_pct = 100 * rel / (rel + dim)
    dim_pct = 100 * dim / (rel + dim)

    gap = 0.6
    y = list(range(len(dim_types)))
    y += [len(dim_types) - 1 + gap + 1 + i for i in range(len(rel_types))]
    plot_types = dim_types + rel_types
    pcts = [100 * con[k] / total for k in plot_types]
    colors = [DIM_COLOR] * len(dim_types) + [REL_COLOR] * len(rel_types)

    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.barh(y, pcts, color=colors, height=0.74)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_types, fontsize=7)
    ax.set_xlabel("Share of all constraints (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for yi, p in zip(y, pcts):
        ax.text(p + 0.4, yi, f"{p:.1f}", va="center", ha="left", fontsize=6)
    ax.axhline(len(dim_types) - 1 + (gap + 1) / 2, color="0.8", lw=0.6)
    ax.set_xlim(0, max(pcts) * 1.08)
    ax.set_ylim(-0.7, y[-1] + 0.7)
    ax.legend(
        handles=[
            Patch(color=REL_COLOR, label=f"Relational  {rel_pct:.0f}%"),
            Patch(color=DIM_COLOR, label=f"Dimensional  {dim_pct:.0f}%"),
        ],
        loc="center right",
        fontsize=7,
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.3,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    pdf = out_dir / "ch3_constraint_distribution.pdf"
    png = out_dir / "ch3_constraint_distribution.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)

    ncon = [r["n_constraints"] for r in recs]
    nprim = [r["n_primitives"] for r in recs]
    ncon_construction = [r["n_construction"] for r in recs]
    dofs = [r["dof"] for r in recs]
    stats = {
        "n_sketches_complete": n_sketches,
        "n_annotations": len(recs),
        "fully_constrained_rate": sum(d == 0 for d in dofs) / len(dofs),
        "relational_pct": rel_pct,
        "dimensional_pct": dim_pct,
        "constraints_per_annotation": {
            "mean": st.mean(ncon),
            "median": st.median(ncon),
            "min": min(ncon),
            "max": max(ncon),
        },
        "primitives_per_annotation": {
            "mean": st.mean(nprim),
            "median": st.median(nprim),
            "min": min(nprim),
            "max": max(nprim),
        },
        "annotations_with_construction": sum(c > 0 for c in ncon_construction),
        "constraint_type_pct": {k: 100 * con[k] / total for k in ordered},
    }
    (out_dir / "ch3_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Saved {pdf}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
