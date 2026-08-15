"""Correlate the pre-edit paper metrics with the finite-edit discrepancy of experiment A.

Primary analysis: Spearman rho per (pair kind, magnitude kind, magnitude), computed over
successful trials only. Solver failures are never assigned a substitute distance; they are
reported in failure_summary.csv instead. Confidence intervals come from a sketch-cluster
bootstrap because trials of one sketch are not independent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuttal_exp.common import (  # noqa: E402
    percentile_ci,
    sketch_cluster_bootstrap,
    write_environment,
)

METRICS = ("theta_full", "theta_rel", "theta_leak", "placement_distance")
NEAR_ZERO_DEFAULT = 1e-10
NEAR_ZERO_SENSITIVITY = (1e-12, 1e-9, 1e-8)
MIN_VALID_REPLICATES = 1500
MIN_ROWS_FOR_CORRELATION = 20

CORRELATION_COLUMNS = [
    "analysis", "pair_kind", "magnitude_kind", "magnitude_value", "metric",
    "n_rows", "n_sketches", "rho", "ci_low", "ci_high", "n_valid_replicates",
    "ci_unstable", "pearson_r",
]
FAILURE_COLUMNS = [
    "pair_kind", "magnitude_kind", "magnitude_value",
    "n_designed", "n_invalid_target_domain", "n_baseline_failure_pairs",
    "n_both_success", "n_P_only_failure", "n_G_only_failure", "n_both_failure",
    "n_near_zero_both_response", "both_success_rate",
]


def load_trials(run_dir: Path) -> pd.DataFrame:
    trials = pd.read_csv(run_dir / "trials.csv")
    trials["skip_reason"] = trials["skip_reason"].fillna("")
    for column in ("solve_ok_p", "solve_ok_g"):
        trials[column] = trials[column].fillna(False).astype(bool)
    return trials


def mark_near_zero(trials: pd.DataFrame, threshold: float) -> pd.Series:
    norms = trials[["response_norm_p", "response_norm_g"]].fillna(np.inf)
    return (norms["response_norm_p"] < threshold) & (norms["response_norm_g"] < threshold)


def analysable(trials: pd.DataFrame, threshold: float = NEAR_ZERO_DEFAULT) -> pd.DataFrame:
    both_ok = trials["solve_ok_p"] & trials["solve_ok_g"]
    return trials[both_ok & ~mark_near_zero(trials, threshold) & trials["d_finite"].notna()]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def correlation_record(
    frame: pd.DataFrame, metric: str, *, analysis: str, n_replicates: int, seed: int
) -> tuple[dict, np.ndarray] | None:
    subset = frame[frame[metric].notna()]
    if len(subset) < MIN_ROWS_FOR_CORRELATION:
        return None
    x = subset[metric].to_numpy(dtype=float)
    y = subset["d_finite"].to_numpy(dtype=float)
    clusters = subset["sketch_id"].to_numpy()

    replicates = sketch_cluster_bootstrap(
        clusters, lambda rows: _spearman(x[rows], y[rows]),
        n_replicates=n_replicates, seed=seed,
    )
    ci_low, ci_high, n_valid = percentile_ci(replicates)
    pearson = float("nan") if np.all(x == x[0]) or np.all(y == y[0]) else float(pearsonr(x, y).statistic)
    return dict(
        analysis=analysis,
        pair_kind=subset["pair_kind"].iloc[0],
        magnitude_kind=subset["magnitude_kind"].iloc[0],
        magnitude_value=subset["magnitude_value"].iloc[0],
        metric=metric,
        n_rows=len(subset),
        n_sketches=subset["sketch_id"].nunique(),
        rho=_spearman(x, y),
        ci_low=ci_low,
        ci_high=ci_high,
        n_valid_replicates=n_valid,
        ci_unstable=n_valid < MIN_VALID_REPLICATES,
        pearson_r=pearson,
    ), replicates


def _grouped_records(frame, analysis, n_replicates, seed):
    rows, replicate_frames = [], []
    grouping = ["pair_kind", "magnitude_kind", "magnitude_value"]
    for keys, group in frame.groupby(grouping, sort=True):
        for metric in METRICS:
            result = correlation_record(
                group, metric, analysis=analysis, n_replicates=n_replicates, seed=seed
            )
            if result is None:
                continue
            record, replicates = result
            rows.append(record)
            replicate_frames.append(pd.DataFrame(dict(
                analysis=analysis, pair_kind=keys[0], magnitude_kind=keys[1],
                magnitude_value=keys[2], metric=metric,
                replicate=np.arange(len(replicates)), rho=replicates,
            )))
    return rows, replicate_frames


def by_driver_type(frame: pd.DataFrame, n_replicates: int, seed: int) -> pd.DataFrame:
    """Exploratory: is a pooled correlation driven by the mix of driver types?"""
    rows = []
    for driver_type, group in frame.groupby("driver_type", sort=True):
        records, _ = _grouped_records(group, f"driver_type:{driver_type}", n_replicates, seed)
        rows.extend(records)
    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


def median_aggregated(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (sketch, pair, magnitude): drivers and signs collapsed by median."""
    keys = ["sketch_id", "pair_id", "pair_kind", "magnitude_kind", "magnitude_value"]
    return frame.groupby(keys, as_index=False)[["d_finite", *METRICS]].median()


def unordered_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one direction per unordered annotation pair; theta_leak is dropped as asymmetric."""
    ordered = frame["annotation_p"] < frame["annotation_g"]
    kept = frame[ordered].copy()
    kept["theta_leak"] = np.nan
    return kept


def failure_summary(trials: pd.DataFrame, pair_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline_failures = (
        pair_metrics[pair_metrics["status"] == "baseline_solve_failure"]
        .groupby("pair_kind")
        .size()
    )
    rows = []
    grouping = ["pair_kind", "magnitude_kind", "magnitude_value"]
    for (pair_kind, magnitude_kind, magnitude_value), group in trials.groupby(grouping, sort=True):
        invalid = group["skip_reason"] == "invalid_target_domain"
        attempted = group[~invalid]
        ok_p, ok_g = attempted["solve_ok_p"], attempted["solve_ok_g"]
        both_success = int((ok_p & ok_g).sum())
        rows.append(dict(
            pair_kind=pair_kind,
            magnitude_kind=magnitude_kind,
            magnitude_value=magnitude_value,
            n_designed=len(group),
            n_invalid_target_domain=int(invalid.sum()),
            n_baseline_failure_pairs=int(baseline_failures.get(pair_kind, 0)),
            n_both_success=both_success,
            n_P_only_failure=int((~ok_p & ok_g).sum()),
            n_G_only_failure=int((ok_p & ~ok_g).sum()),
            n_both_failure=int((~ok_p & ~ok_g).sum()),
            n_near_zero_both_response=int(mark_near_zero(attempted, NEAR_ZERO_DEFAULT).sum()),
            both_success_rate=both_success / len(attempted) if len(attempted) else float("nan"),
        ))
    return pd.DataFrame(rows, columns=FAILURE_COLUMNS)


def success_by_metric_quartile(trials: pd.DataFrame) -> pd.DataFrame:
    """Guards against pairs with large metrics being selectively dropped by solver failures."""
    attempted = trials[trials["skip_reason"] != "invalid_target_domain"].copy()
    attempted["both_success"] = attempted["solve_ok_p"] & attempted["solve_ok_g"]
    rows = []
    for (pair_kind, magnitude_kind, magnitude_value), group in attempted.groupby(
        ["pair_kind", "magnitude_kind", "magnitude_value"], sort=True
    ):
        for metric in METRICS:
            values = group[metric]
            if values.notna().sum() < MIN_ROWS_FOR_CORRELATION or values.nunique() < 4:
                continue
            quartile = pd.qcut(values, 4, labels=False, duplicates="drop")
            for level, bucket in group.groupby(quartile, observed=True):
                rows.append(dict(
                    pair_kind=pair_kind, magnitude_kind=magnitude_kind,
                    magnitude_value=magnitude_value, metric=metric, quartile=int(level),
                    n=len(bucket), both_success_rate=float(bucket["both_success"].mean()),
                ))
    return pd.DataFrame(rows)


def near_zero_sensitivity(trials: pd.DataFrame, n_replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for threshold in NEAR_ZERO_SENSITIVITY:
        frame = analysable(trials, threshold)
        records, _ = _grouped_records(frame, f"near_zero_{threshold:g}", n_replicates, seed)
        rows.extend(records)
    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _magnitude_axis(magnitude_kind: str) -> tuple[str, float]:
    return ("edit magnitude (% of bbox diagonal)", 100.0) if magnitude_kind == "length" else (
        "edit magnitude (degrees)", 1.0
    )


def plot_rho_vs_magnitude(correlations: pd.DataFrame, magnitude_kind: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = correlations[
        (correlations["analysis"] == "primary") & (correlations["magnitude_kind"] == magnitude_kind)
    ]
    if frame.empty:
        return
    pair_kinds = sorted(frame["pair_kind"].unique())
    label, scale = _magnitude_axis(magnitude_kind)
    fig, axes = plt.subplots(1, len(pair_kinds), figsize=(6 * len(pair_kinds), 4.2), squeeze=False)
    for ax, pair_kind in zip(axes[0], pair_kinds):
        panel = frame[frame["pair_kind"] == pair_kind]
        for metric in METRICS:
            series = panel[panel["metric"] == metric].sort_values("magnitude_value")
            if series.empty:
                continue
            x = series["magnitude_value"] * scale
            ax.plot(x, series["rho"], marker="o", label=metric)
            ax.fill_between(x, series["ci_low"], series["ci_high"], alpha=0.15)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel(r"Spearman $\rho$ vs. $D_{finite}$")
        ax.set_title(f"{pair_kind} pairs (n_sketches={panel['n_sketches'].max()})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_success_rate(failures: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, failures["magnitude_kind"].nunique(), figsize=(11, 4.2), squeeze=False
    )
    for ax, magnitude_kind in zip(axes[0], sorted(failures["magnitude_kind"].unique())):
        panel = failures[failures["magnitude_kind"] == magnitude_kind]
        label, scale = _magnitude_axis(magnitude_kind)
        for pair_kind, group in panel.groupby("pair_kind"):
            series = group.sort_values("magnitude_value")
            ax.plot(series["magnitude_value"] * scale, series["both_success_rate"],
                    marker="o", label=pair_kind)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(label)
        ax.set_ylabel("both-side solve success rate")
        ax.set_title(magnitude_kind)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter(frame: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    human = frame[frame["pair_kind"] == "human"]
    if human.empty:
        return
    magnitudes = sorted(human[human["magnitude_kind"] == "length"]["magnitude_value"].unique())
    if not magnitudes:
        return
    for magnitude in (magnitudes[0], magnitudes[len(magnitudes) // 2], magnitudes[-1]):
        panel = human[(human["magnitude_kind"] == "length") & (human["magnitude_value"] == magnitude)]
        for metric in METRICS:
            subset = panel[panel[metric].notna()]
            if len(subset) < MIN_ROWS_FOR_CORRELATION:
                continue
            fig, ax = plt.subplots(figsize=(4.4, 4.0))
            ax.hexbin(subset[metric], subset["d_finite"], gridsize=30, mincnt=1, cmap="viridis")
            ax.set_xlabel(metric)
            ax.set_ylabel(r"$D_{finite}$")
            ax.set_title(f"length {magnitude:g} (n={len(subset)})")
            fig.tight_layout()
            fig.savefig(figures_dir / f"scatter_{metric}_length{magnitude:g}.pdf")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="run_dir", default="artifacts/rebuttal/finite_edit")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    write_environment(
        run_dir,
        sys.argv,
        [Path(__file__), Path(__file__).with_name("common.py")],
        filename="analysis_environment.json",
    )
    trials = load_trials(run_dir)
    pair_metrics = pd.read_csv(run_dir / "pair_metrics.csv")
    primary = analysable(trials)

    records, replicate_frames = _grouped_records(primary, "primary", args.replicates, args.seed)
    for analysis, frame in (
        ("median_aggregated", median_aggregated(primary)),
        ("unordered_only", unordered_only(primary)),
    ):
        extra, _ = _grouped_records(frame, analysis, args.replicates, args.seed)
        records.extend(extra)

    correlations = pd.concat(
        [
            pd.DataFrame(records, columns=CORRELATION_COLUMNS),
            near_zero_sensitivity(trials, args.replicates, args.seed),
            by_driver_type(primary, args.replicates, args.seed),
        ],
        ignore_index=True,
    )
    correlations.to_csv(run_dir / "correlations.csv", index=False)
    if replicate_frames:
        pd.concat(replicate_frames, ignore_index=True).to_parquet(
            run_dir / "bootstrap_replicates.parquet", index=False
        )

    failures = failure_summary(trials, pair_metrics)
    failures.to_csv(run_dir / "failure_summary.csv", index=False)
    if (failures["both_success_rate"] < 0.95).any():
        success_by_metric_quartile(trials).to_csv(
            run_dir / "success_by_metric_quartile.csv", index=False
        )

    if not args.no_figures:
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_rho_vs_magnitude(correlations, "length", figures_dir / "rho_vs_length_magnitude.pdf")
        plot_rho_vs_magnitude(correlations, "angle", figures_dir / "rho_vs_angle_magnitude.pdf")
        plot_success_rate(failures, figures_dir / "success_rate_vs_magnitude.pdf")
        plot_scatter(primary, figures_dir)

    unstable = int(correlations["ci_unstable"].sum())
    print(f"analysable trials: {len(primary)} / {len(trials)}")
    print(f"correlation rows: {len(correlations)} ({unstable} with unstable CI)")
    print(correlations[correlations["analysis"] == "primary"].to_string(index=False))
    print(f"wrote -> {run_dir}")


if __name__ == "__main__":
    main()
