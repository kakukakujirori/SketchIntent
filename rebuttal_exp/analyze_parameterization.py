"""How far do the paper metrics move when the primitive chart changes?

Headline results are the paired |d' - d| distribution per scenario and whether the
variant ordering of the Sec. 5 table survives; the rank correlation is reported
alongside them rather than as the claim. C0 is an integration gate, not a result: if the
identity chart does not reproduce the native numbers, nothing else is trustworthy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ablation"))  # ablation/aggregate.py imports _bootstrap

from ablation.aggregate import ANGLES, VARIANT_ORDER, aggregate  # noqa: E402
from rebuttal_exp.common import percentile_ci, sketch_cluster_bootstrap  # noqa: E402
from rebuttal_exp.primitive_charts import C0, NATIVE  # noqa: E402

METRICS = ("theta_full", "theta_rel", "theta_leak", "placement_sim")
KEY = ["rid", "variant", "gt"]
IDENTITY_RTOL = 1e-8
IDENTITY_ATOL = 1e-10
RESOLVED_METRIC_FLOOR = 1e-6
ZERO_ANGLE_NOISE = 1e-6
"""The padded angle is sqrt(1 - cos2/max). When two subspaces coincide, cos2/max is
1 - O(machine eps) and the square root lifts that to O(1e-8), so a metric value near zero
carries a sqrt(eps) noise floor no pipeline can beat. The identity gate is therefore split:
values above RESOLVED_METRIC_FLOOR must agree to machine precision, values at the floor
only have to stay inside that noise band."""
WORST_CASES = 20

SUMMARY_COLUMNS = [
    "scenario", "metric", "n_pairs", "n_sketches",
    "median_absolute_delta", "q90_absolute_delta", "q95_absolute_delta", "max_absolute_delta",
    "median_relative_delta", "q90_relative_delta", "max_relative_delta",
    "rank_columns", "n_rank_changed", "n_moved", "n_moved_with_rank_change",
    "spearman", "spearman_ci_low", "spearman_ci_high", "n_valid_replicates",
]
RANK_COLUMNS = {
    "theta_full": ("k_full_pred", "k_full_gt"),
    "theta_rel": ("k_rel_pred", "k_rel_gt"),
    "placement_sim": ("n_dim_pred", "n_dim_gt"),
    # theta_leak constructs U_P = orth(R_pred); n_dim_pred instead counts non-zero
    # observable response columns and is not the dimension used by this metric.
    "theta_leak": ("k_response_pred",),
}
"""Subspace dimensions behind each metric. The null-space and range cutoffs are absolute,
so a chart that rescales singular values can move a borderline direction across them.
Splitting the moved records by whether a dimension also changed separates that numerical
boundary effect from genuine parameterization dependence."""
MOVED_TOL = 1e-6
RANKING_COLUMNS = [
    "scenario", "metric", "n_variants", "ordering_preserved", "method_ordering_preserved",
    "native_order", "transformed_order", "winner_flip_rate", "n_flip_comparisons",
]
CONTROL_PREFIXES = ("dim_drop", "rel_drop")
NON_METHOD_VARIANTS = ("source_self",)


def ordered_variants(pairs: pd.DataFrame) -> list[str]:
    """Sec. 5 order, then any variant the ablation list predates (e.g. the proposed model)."""
    present = set(pairs["variant"].unique())
    known = [v for v in VARIANT_ORDER if v in present]
    return known + sorted(present - set(VARIANT_ORDER))


def is_method_variant(variant: str) -> bool:
    """Held-out human and the model runs: the rows the paper's conclusions rest on."""
    return not variant.startswith(CONTROL_PREFIXES) and variant not in NON_METHOD_VARIANTS


def load_pairs(run_dir: Path) -> pd.DataFrame:
    pairs = pd.read_parquet(run_dir / "transformed_pair_metrics.parquet")
    return pairs[pairs["status"] == "ok"].copy()


def paired_frame(pairs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Rows where both native and `scenario` produced a metric for the same record."""
    native = pairs[pairs["scenario"] == NATIVE].set_index(KEY)
    other = pairs[pairs["scenario"] == scenario].set_index(KEY)
    common = native.index.intersection(other.index)
    carried = list(METRICS) + sorted({c for cols in RANK_COLUMNS.values() for c in cols})
    joined = native.loc[common, ["sketch_id", *carried]].join(
        other.loc[common, carried], rsuffix="_transformed"
    )
    return joined.reset_index()


def paired_rows(pairs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Both scenarios' rows, restricted to the records both of them produced.

    A transformed scenario only covers sketches holding its target primitive, so the two
    orderings must be built on the same records; comparing an ordering over all sketches
    with one over the arc-bearing subset would measure the sample, not the chart.
    """
    keys = pd.MultiIndex.from_frame(pairs[KEY])
    native = keys[pairs["scenario"].to_numpy() == NATIVE]
    other = keys[pairs["scenario"].to_numpy() == scenario]
    common = native.intersection(other)
    return pairs[keys.isin(common) & pairs["scenario"].isin([NATIVE, scenario])]


def identity_report(pairs: pd.DataFrame) -> pd.DataFrame:
    """C0 gate: the identity chart must reproduce the native metrics.

    Resolved values are held to machine precision; values sitting at the metric's own
    sqrt(eps) noise floor only have to stay within that band.
    """
    joined = paired_frame(pairs, C0)
    rows = []
    for metric in METRICS:
        native = joined[metric].to_numpy(dtype=float)
        transformed = joined[f"{metric}_transformed"].to_numpy(dtype=float)
        delta = np.abs(native - transformed)
        resolved = native > RESOLVED_METRIC_FLOOR
        mismatched = ~np.isclose(native, transformed, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL)
        rows.append(dict(
            metric=metric,
            n_pairs=len(joined),
            n_resolved=int(resolved.sum()),
            n_mismatch_resolved=int((mismatched & resolved).sum()),
            max_delta_resolved=_max_or_zero(delta[resolved]),
            n_near_zero=int((~resolved).sum()),
            n_mismatch_near_zero=int((mismatched & ~resolved).sum()),
            max_delta_near_zero=_max_or_zero(delta[~resolved]),
        ))
    return pd.DataFrame(rows)


def identity_gate_failed(gate: pd.DataFrame) -> bool:
    return bool(
        int(gate["n_mismatch_resolved"].sum())
        or (gate["max_delta_near_zero"] > ZERO_ANGLE_NOISE).any()
    )


def _max_or_zero(values: np.ndarray) -> float:
    return float(values.max()) if values.size else 0.0


def rank_columns_match(pairs: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [
        "k_full_pred", "k_full_gt", "k_rel_pred", "k_rel_gt",
        "k_response_pred", "k_response_gt", "n_dim_pred", "n_dim_gt",
    ]
    native = pairs[pairs["scenario"] == NATIVE].set_index(KEY)
    other = pairs[pairs["scenario"] == C0].set_index(KEY)
    common = native.index.intersection(other.index)
    rows = []
    for column in rank_cols:
        left = native.loc[common, column].to_numpy()
        right = other.loc[common, column].to_numpy()
        rows.append(dict(column=column, n=len(common), n_mismatch=int((left != right).sum())))
    return pd.DataFrame(rows)


def delta_summary(pairs: pd.DataFrame, scenario: str, *, replicates: int, seed: int) -> pd.DataFrame:
    joined = paired_frame(pairs, scenario)
    rows = []
    for metric in METRICS:
        native = joined[metric].to_numpy(dtype=float)
        transformed = joined[f"{metric}_transformed"].to_numpy(dtype=float)
        absolute = np.abs(transformed - native)
        relative = absolute / np.maximum(np.abs(native), 1e-12)
        clusters = joined["sketch_id"].to_numpy()

        def statistic(rows_idx, x=native, y=transformed):
            a, b = x[rows_idx], y[rows_idx]
            if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
                return float("nan")
            return float(spearmanr(a, b).statistic)

        replicates_array = sketch_cluster_bootstrap(
            clusters, statistic, n_replicates=replicates, seed=seed
        )
        ci_low, ci_high, n_valid = percentile_ci(replicates_array)
        rank_changed = _rank_changed(joined, metric)
        moved = absolute > MOVED_TOL
        rows.append(dict(
            scenario=scenario, metric=metric,
            n_pairs=len(joined), n_sketches=joined["sketch_id"].nunique(),
            rank_columns="+".join(RANK_COLUMNS[metric]),
            n_rank_changed=int(rank_changed.sum()),
            n_moved=int(moved.sum()),
            n_moved_with_rank_change=int((moved & rank_changed).sum()),
            median_absolute_delta=float(np.median(absolute)) if absolute.size else float("nan"),
            q90_absolute_delta=_quantile(absolute, 0.90),
            q95_absolute_delta=_quantile(absolute, 0.95),
            max_absolute_delta=float(absolute.max()) if absolute.size else float("nan"),
            median_relative_delta=float(np.median(relative)) if relative.size else float("nan"),
            q90_relative_delta=_quantile(relative, 0.90),
            max_relative_delta=float(relative.max()) if relative.size else float("nan"),
            spearman=statistic(np.arange(len(joined))),
            spearman_ci_low=ci_low, spearman_ci_high=ci_high, n_valid_replicates=n_valid,
        ))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def _rank_changed(joined: pd.DataFrame, metric: str) -> np.ndarray:
    changed = np.zeros(len(joined), dtype=bool)
    for column in RANK_COLUMNS[metric]:
        changed |= (joined[column] != joined[f"{column}_transformed"]).to_numpy()
    return changed


def aggregated_variant_medians(pairs: pd.DataFrame, scenario: str, variants: list[str]) -> dict:
    """Sec. 5 aggregation applied independently within one scenario."""
    rows = pairs[pairs["scenario"] == scenario].to_dict("records")
    collapsed = aggregate(rows, list(METRICS), best=True)
    medians: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        per_variant = {}
        for variant in variants:
            values = [d[metric] for (_rid, v), d in collapsed.items() if v == variant and metric in d]
            if values:
                per_variant[variant] = float(np.median(values))
        medians[metric] = per_variant
    return medians


def variant_ordering(medians: dict, metric: str) -> list[str]:
    """Best-first ordering; angles are distances and similarities are not."""
    per_variant = medians[metric]
    reverse = metric not in ANGLES
    return [v for v, _ in sorted(per_variant.items(), key=lambda kv: kv[1], reverse=reverse)]


def winner_flip_rate(pairs: pd.DataFrame, scenario: str, metric: str) -> tuple[float, int]:
    """How often a per-sketch pairwise variant comparison reverses under the chart."""
    native = aggregate(pairs[pairs["scenario"] == NATIVE].to_dict("records"), [metric], best=True)
    other = aggregate(pairs[pairs["scenario"] == scenario].to_dict("records"), [metric], best=True)
    by_rid: dict[str, dict[str, tuple[float, float]]] = {}
    for (rid, variant), values in native.items():
        if metric in values and (rid, variant) in other and metric in other[(rid, variant)]:
            by_rid.setdefault(rid, {})[variant] = (values[metric], other[(rid, variant)][metric])

    flips = comparisons = 0
    for variants in by_rid.values():
        labels = sorted(variants)
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                n_left, t_left = variants[left]
                n_right, t_right = variants[right]
                if n_left == n_right or t_left == t_right:
                    continue
                comparisons += 1
                if (n_left < n_right) != (t_left < t_right):
                    flips += 1
    return (flips / comparisons if comparisons else float("nan")), comparisons


def ranking_report(pairs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    paired = paired_rows(pairs, scenario)
    variants = ordered_variants(paired)
    native_medians = aggregated_variant_medians(paired, NATIVE, variants)
    other_medians = aggregated_variant_medians(paired, scenario, variants)
    rows = []
    for metric in METRICS:
        native_order = variant_ordering(native_medians, metric)
        transformed_order = variant_ordering(other_medians, metric)
        methods = [v for v in native_order if is_method_variant(v)]
        transformed_methods = [v for v in transformed_order if is_method_variant(v)]
        flip_rate, comparisons = winner_flip_rate(paired, scenario, metric)
        rows.append(dict(
            scenario=scenario, metric=metric, n_variants=len(native_order),
            ordering_preserved=native_order == transformed_order,
            method_ordering_preserved=methods == transformed_methods,
            native_order=">".join(native_order),
            transformed_order=">".join(transformed_order),
            winner_flip_rate=flip_rate, n_flip_comparisons=comparisons,
        ))
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def worst_cases(pairs: pd.DataFrame, diagnostics: pd.DataFrame, scenario: str) -> pd.DataFrame:
    joined = paired_frame(pairs, scenario)
    rows = []
    for metric in METRICS:
        frame = joined.copy()
        frame["metric"] = metric
        frame["absolute_delta"] = (frame[f"{metric}_transformed"] - frame[metric]).abs()
        frame["native_value"] = frame[metric]
        frame["transformed_value"] = frame[f"{metric}_transformed"]
        top = frame.sort_values(["absolute_delta", "rid", "variant", "gt"],
                                ascending=[False, True, True, True]).head(WORST_CASES)
        rows.append(top[["sketch_id", "rid", "variant", "gt", "metric",
                         "native_value", "transformed_value", "absolute_delta"]])
    worst = pd.concat(rows, ignore_index=True)
    worst["scenario"] = scenario
    return worst.merge(_sketch_composition(diagnostics, scenario), on="sketch_id", how="left")


def _sketch_composition(diagnostics: pd.DataFrame, scenario: str) -> pd.DataFrame:
    frame = diagnostics[diagnostics["scenario"] == scenario]
    if frame.empty:
        return pd.DataFrame(columns=["sketch_id"])
    grouped = frame.groupby("sketch_id").agg(
        n_transformed_blocks=("eid", "nunique"),
        worst_circumcircle_condition=("circumcircle_condition", "min"),
        worst_gauge_residual_J_x=("gauge_residual_J_x", "max"),
        worst_gauge_residual_J_R=("gauge_residual_J_R", "max"),
        worst_identity_error=("identity_error", "max"),
    )
    return grouped.reset_index()


def plot_delta(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = sorted(summary["scenario"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(scenarios) * len(METRICS) + 2, 4.2))
    width = 0.8 / max(len(scenarios), 1)
    positions = np.arange(len(METRICS))
    for offset, scenario in enumerate(scenarios):
        frame = summary[summary["scenario"] == scenario].set_index("metric")
        medians = [frame.loc[m, "median_absolute_delta"] if m in frame.index else np.nan
                   for m in METRICS]
        q90 = [frame.loc[m, "q90_absolute_delta"] if m in frame.index else np.nan for m in METRICS]
        maxima = [frame.loc[m, "max_absolute_delta"] if m in frame.index else np.nan
                  for m in METRICS]
        x = positions + offset * width - 0.4 + width / 2
        ax.bar(x, np.maximum(medians, 1e-18), width=width * 0.9, label=f"{scenario} median")
        ax.scatter(x, np.maximum(q90, 1e-18), marker="_", s=90, color="black")
        ax.scatter(x, np.maximum(maxima, 1e-18), marker="v", s=28, color="crimson")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(METRICS)
    ax.set_ylabel(r"$|d' - d|$   (bar: median, $-$: Q90, $\vee$: max)")
    ax.set_title("Metric shift under an alternative primitive chart")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rank_correlation(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = sorted(summary["scenario"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    positions = np.arange(len(METRICS))
    width = 0.8 / max(len(scenarios), 1)
    for offset, scenario in enumerate(scenarios):
        frame = summary[summary["scenario"] == scenario].set_index("metric")
        values = [frame.loc[m, "spearman"] if m in frame.index else np.nan for m in METRICS]
        low = [frame.loc[m, "spearman_ci_low"] if m in frame.index else np.nan for m in METRICS]
        high = [frame.loc[m, "spearman_ci_high"] if m in frame.index else np.nan for m in METRICS]
        x = positions + offset * width - 0.4 + width / 2
        ax.bar(x, values, width=width * 0.9, label=scenario)
        ax.vlines(x, low, high, color="black", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Spearman(native, transformed)")
    ax.set_title("Per-pair rank agreement (secondary to the delta distribution)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="run_dir", default="artifacts/rebuttal/parameterization")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pairs = load_pairs(run_dir)
    diagnostics = pd.read_parquet(run_dir / "transform_diagnostics.parquet")
    scenarios = [s for s in pairs["scenario"].unique() if s != NATIVE]

    if C0 in scenarios:
        gate = identity_report(pairs)
        ranks = rank_columns_match(pairs)
        gate.to_csv(run_dir / "identity_gate.csv", index=False)
        ranks.to_csv(run_dir / "identity_gate_ranks.csv", index=False)
        print("=== C0 identity gate ===")
        print(gate.to_string(index=False))
        print(ranks.to_string(index=False))
        if identity_gate_failed(gate) or int(ranks["n_mismatch"].sum()):
            raise SystemExit("C0 does not reproduce the native metrics; not aggregating C1/C2")
        scenarios = [s for s in scenarios if s != C0]

    if not scenarios:
        print("no transformed scenarios beyond C0 in this run")
        return

    summary = pd.concat(
        [delta_summary(pairs, s, replicates=args.replicates, seed=args.seed) for s in scenarios],
        ignore_index=True,
    )
    ranking = pd.concat([ranking_report(pairs, s) for s in scenarios], ignore_index=True)
    worst = pd.concat([worst_cases(pairs, diagnostics, s) for s in scenarios], ignore_index=True)

    summary.to_csv(run_dir / "summary.csv", index=False)
    ranking.to_csv(run_dir / "ranking_changes.csv", index=False)
    worst.to_csv(run_dir / "worst_cases.csv", index=False)

    if not args.no_figures:
        figures = run_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        plot_delta(summary, figures / "delta_by_transform.pdf")
        plot_rank_correlation(summary, figures / "rank_correlation_by_transform.pdf")

    print("\n=== absolute delta vs native ===")
    print(summary[["scenario", "metric", "n_pairs", "median_absolute_delta",
                   "q90_absolute_delta", "max_absolute_delta", "n_moved",
                   "n_moved_with_rank_change", "spearman"]].to_string(index=False))
    print("\n=== variant ordering ===")
    print(ranking[["scenario", "metric", "ordering_preserved", "method_ordering_preserved",
                   "winner_flip_rate", "n_flip_comparisons"]].to_string(index=False))
    print(f"\nwrote -> {run_dir}")


if __name__ == "__main__":
    main()
