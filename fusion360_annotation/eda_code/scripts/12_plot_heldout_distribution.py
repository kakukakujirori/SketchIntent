from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

DISTANCE_METRICS = {"theta_full", "theta_rel", "theta_leak"}
METRICS = ["theta_full", "theta_rel", "theta_leak", "placement_sim"]
PLOT_LABELS = {
    "theta_full": r"$\theta_{\mathrm{motion}}^{\mathrm{full}}$",
    "theta_rel": r"$\theta_{\mathrm{motion}}^{\mathrm{rel}}$",
    "theta_leak": r"$\theta_{\mathrm{leak}}^{\mathrm{dim}}$",
    "placement_sim": r"$S_{\mathrm{place}}$",
}


def aggregate_multiref_best(rows: list[dict], variant: str) -> dict[str, list[float]]:
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        if row.get("variant") != variant:
            continue
        rid = row["rid"]
        for metric in METRICS:
            value = row.get(metric)
            if value not in (None, ""):
                grouped[rid][metric].append(float(value))

    values = {metric: [] for metric in METRICS}
    for by_metric in grouped.values():
        for metric in METRICS:
            xs = by_metric.get(metric)
            if not xs:
                continue
            if metric in DISTANCE_METRICS:
                values[metric].append(min(xs))
            else:
                values[metric].append(max(xs))
    return values


def percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        raise ValueError("empty input")
    pos = (len(sorted_xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def summarize(values: dict[str, list[float]], near_zero_eps: float) -> dict[str, dict]:
    out = {}
    for metric, xs in values.items():
        sx = sorted(xs)
        near_zero = sum(x <= near_zero_eps for x in sx)
        exact_zero = sum(x == 0 for x in sx)
        out[metric] = {
            "n": len(sx),
            "mean": st.fmean(sx),
            "median": st.median(sx),
            "q1": percentile(sx, 0.25),
            "q3": percentile(sx, 0.75),
            "min": sx[0],
            "max": sx[-1],
            "exact_zero": exact_zero,
            "exact_zero_pct": 100 * exact_zero / len(sx),
            "near_zero_eps": near_zero_eps,
            "near_zero": near_zero,
            "near_zero_pct": 100 * near_zero / len(sx),
        }
    return out


def plot(values: dict[str, list[float]], stats: dict[str, dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 3.25), constrained_layout=True)
    bins = [i / 20 for i in range(21)]

    def fmt(value: float) -> str:
        return f"{value:.1e}" if abs(value) < 0.01 else f"{value:.2f}"

    for ax, metric in zip(axs.ravel(), METRICS):
        xs = values[metric]
        color = "#4C78A8" if metric in DISTANCE_METRICS else "#54A24B"
        ax.hist(xs, bins=bins, color=color, edgecolor="white", linewidth=0.45)
        ax.axvline(stats[metric]["median"], color="black", lw=1.15, ls="--")
        ax.axvline(stats[metric]["mean"], color="#D95F02", lw=1.15, ls=":")
        ax.set_xlim(0, 1)
        ax.set_title(PLOT_LABELS[metric])
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.spines[["top", "right"]].set_visible(False)

        mean = stats[metric]["mean"]
        median = stats[metric]["median"]
        note = f"mean {fmt(mean)}\nmedian {fmt(median)}"
        note_x = 0.18 if metric == "placement_sim" else 0.40
        ax.text(
            note_x,
            0.94,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-csv", default=str(HERE / "out" / "motion_eval.csv"))
    ap.add_argument("--variant", default="heldout")
    ap.add_argument("--near-zero-eps", type=float, default=1e-6)
    ap.add_argument(
        "--output",
        default=str(REPO_ROOT / "paper" / "figs" / "appendix_heldout_human_distribution.pdf"),
    )
    ap.add_argument(
        "--stats-output",
        default=str(HERE / "out" / "heldout_human_distribution_stats.json"),
    )
    args = ap.parse_args()

    with Path(args.motion_csv).open() as f:
        rows = list(csv.DictReader(f))
    values = aggregate_multiref_best(rows, args.variant)
    stats = summarize(values, args.near_zero_eps)

    output = Path(args.output)
    plot(values, stats, output)
    Path(args.stats_output).write_text(json.dumps(stats, indent=2))

    print(f"wrote {output}")
    print(f"wrote {args.stats_output}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
