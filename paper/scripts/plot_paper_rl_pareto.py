"""Paper-ready RL reward Pareto figure for Sec.6.

Reuses the JSON parsing in plot_fiverr_eval_pareto.py. Draws FC-rate on the x-axis
against the three headline/independent axes (dimensional ratio, full motion angle,
S_place), connecting checkpoints from the same run into a trajectory curve so the
Solver-RL (DimParsimony) baseline frontier and the Ours trajectory are comparable
at matched FC.

    conda run -n freecad python paper/scripts/plot_paper_rl_pareto.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from plot_fiverr_eval_pareto import row_from_json  # type: ignore

ART = Path("artifacts/fiverr_eval")

# Each series: label, color, marker, connect, [(json, step)] ordered by step.
SERIES = [
    (
        "Solver-RL (DP c1)",
        "#1f77b4",
        "o",
        True,
        [(f"dpc1_step{s}.json", s) for s in (5000, 10000, 15000, 20000, 25000, 30000, 35000)],
    ),
    (
        "Solver-RL (DP c0.3)",
        "#6baed6",
        "v",
        True,
        [(f"dpc03_step{s}.json", s) for s in (5000, 10000, 15000, 20000, 25000, 30000)],
    ),
    (
        "Solver-RL (DP c3)",
        "#08519c",
        "^",
        True,
        [(f"dpc3_step{s}.json", s) for s in (5000, 10000, 15000, 20000, 25000)],
    ),
    (
        "Solver $+$ M (no DP)",
        "#ff7f0e",
        "D",
        True,
        [(f"sm_step{s}.json", s) for s in (5000, 10000, 15000, 20000, 25000, 30000)],
    ),
    (
        "Solver $+$ L (no DP)",
        "#9467bd",
        "P",
        True,
        [(f"sl_step{s}.json", s) for s in (5000, 10000, 15000, 20000, 25000, 30000)],
    ),
    (
        "Ours (M$+$L), in progress",
        "#fb9a99",
        "o",
        True,
        [(f"oursretry_step{s}.json", s) for s in (5000, 10000, 15000)],
    ),
    (
        "Ours (M$+$L), converged",
        "#d62728",
        "*",
        False,
        [("ours_step52200.json", 52200), ("ours_step59800.json", 59800)],
    ),
]

AXES = [
    ("dimensional_ratio", "Dimensional ratio (lower better)"),
    ("motion_full_angle_mean", r"$\theta_{motion}^{full}$ (lower better)"),
    ("placement_sim_mean", r"$S_{place}$ (higher better)"),
]

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 16, 14, 12, 12


def load_series():
    out = []
    for label, color, marker, connect, items in SERIES:
        rows = []
        for fname, step in items:
            p = ART / fname
            if not p.exists():
                continue
            r = row_from_json(label, p)
            r["_step"] = step
            rows.append(r)
        if rows:
            rows.sort(key=lambda r: r["fc_rate"])
            out.append((label, color, marker, connect, rows))
    return out


def main():
    series = load_series()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=180)
    for ax, (key, title) in zip(axes, AXES):
        for label, color, marker, connect, rows in series:
            xs = [r["fc_rate"] for r in rows]
            ys = [r[key] for r in rows]
            if connect and len(rows) > 1:
                ax.plot(xs, ys, "-", color=color, alpha=0.6, lw=1.6, zorder=2)
            ax.scatter(
                xs, ys, s=130 if marker == "*" else 55, c=color, marker=marker,
                edgecolors="white", linewidths=0.8, zorder=3,
                label=label if ax is axes[0] else None,
            )
        ax.set_title(title, fontsize=TITLE_FS)
        ax.set_xlabel("Fully constrained rate (higher better)", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.12, 0.80)
    axes[0].legend(fontsize=LEGEND_FS, loc="best", framealpha=0.92)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"rl_reward_pareto.{ext}")
    fig.savefig("paper/figs/rl_reward_pareto.pdf")
    print(ART / "rl_reward_pareto.png")
    print("paper/figs/rl_reward_pareto.pdf")


if __name__ == "__main__":
    main()
