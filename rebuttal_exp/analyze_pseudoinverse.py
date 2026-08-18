"""How far do the paper metrics move when a different admissible response is chosen?

The headline is the paired |d_eta - d_0| distribution per eta and whether the Sec. 5
variant ordering survives each individual draw; the rank correlation is reported next to
them rather than as the claim. eta = 0 is an integration gate, not a result: if the
sampling path does not reproduce ``pair_paper_metrics`` there, nothing else is trustworthy.

Two accounts are reported side by side. ``invariance_coverage`` counts the responses whose
whole null space is invisible to J_phi: there ``J_phi R`` is the same for every admissible
response, so theta_full, theta_rel and S_place cannot move at all and no sampling is done.
``sensitivity_summary`` covers the remainder -- the responses that really do have a choice.
``guard_summary`` accounts for the discarded draws and shows how far the run sits from the
guard threshold.
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

from ablation.aggregate import ANGLES, aggregate  # noqa: E402
from rebuttal_exp.common import (  # noqa: E402
    RESOLVED_METRIC_FLOOR,
    ZERO_ANGLE_NOISE,
    percentile_ci,
    sketch_cluster_bootstrap,
)
from rebuttal_exp.variants import is_method_variant, ordered_variants  # noqa: E402

METRICS = ("theta_full", "theta_rel", "theta_leak", "placement_sim")
KEY = ["rid", "variant", "gt"]
METRIC_RANK = {
    "theta_full": "k_full_pred",
    "theta_rel": "k_rel_pred",
    "theta_leak": "k_response_pred",
    "placement_sim": "n_dim_pred",
}
"""The Pred-side subspace dimension behind each metric. A different response can change it,
and a metric that moved because its dimension jumped is a different phenomenon from one that
moved while the dimension held."""
RANK_COLUMNS = sorted(set(METRIC_RANK.values()))
BASELINE_ETA = 0.0
GATE_RTOL = 1e-8
GATE_ATOL = 1e-10
DRAW_KEY = ["sketch_id", "pred_label", "eta", "sample"]
"""One perturbation realization. The response, not the evaluation record, is the unit: a
single R_eta is drawn per Pred annotation and then scored against every record that Pred
appears in."""
MOVED_TOL = 1e-6
ORDERING_TOL = 1e-6
WORST_CASES = 20
FREE_RANK_BINS = [(0, 0), (1, 2), (3, 5), (6, 10**9)]
CI_SAMPLE = 0
"""Bootstrap CIs are built on one fixed sample index; the spread across the other draws is
reported separately as spearman_median/min_over_samples rather than folded into the CI."""
GUARD_FLOOR_LADDER = (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5)
"""Candidate guard floors. How arbitrary the chosen floor is cannot be asserted, only
measured: every floor that discards the same draws produces a bit-identical run, so the
counts along this ladder say exactly how far the threshold can move for free."""

SENSITIVITY_COLUMNS = [
    "eta", "metric", "n_records", "n_sketches", "n_samples", "n_rows",
    "mean_absolute_delta", "median_absolute_delta", "q90_absolute_delta",
    "q95_absolute_delta", "max_absolute_delta",
    "mean_relative_delta", "median_relative_delta", "q90_relative_delta", "max_relative_delta",
    "signed_delta_q05", "signed_delta_median", "signed_delta_q95",
    "n_moved", "fraction_moved", "rank_column", "fraction_rank_changed",
    "fraction_moved_with_rank_change",
    "median_within_record_sd", "median_within_record_range", "max_within_record_range",
    "spearman_median_over_samples", "spearman_min_over_samples",
    "spearman_ci_low", "spearman_ci_high", "n_valid_replicates",
]
RANKING_COLUMNS = [
    "eta", "metric", "n_samples", "fraction_ordering_preserved",
    "fraction_method_ordering_preserved", "n_comparable_pairs",
    "median_inverted_pairs", "max_inverted_pairs", "median_winner_flip_rate",
    "max_winner_flip_rate", "native_order", "worst_order",
]
"""`fraction_ordering_preserved` is binary per sample, so one flipped pair scores the same as
a scrambled table. `n_comparable_pairs` (the variant pairs the baseline separates by more
than ORDERING_TOL) and the inverted counts say how much of the ordering actually moved."""


# ---------------------------------------------------------------------------
# Loading and pairing
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    native = pd.read_parquet(run_dir / "native_pair_metrics.parquet")
    native = native[native["status"] == "ok"].copy()
    samples = pd.read_parquet(run_dir / "perturbation_metrics.parquet")
    return native, samples


def paired_samples(native: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Every perturbed row alongside the unperturbed value of the same record."""
    carried = list(METRICS) + RANK_COLUMNS
    baseline = native.set_index(KEY)[["pred_label", *carried]]
    baseline.columns = ["pred_label"] + [f"{name}_0" for name in carried]
    return samples.join(baseline, on=KEY, how="inner")


def identity_report(native: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """eta = 0 gate: the sampling path must reproduce the paper path exactly."""
    baseline = samples[samples["eta"] == BASELINE_ETA]
    joined = native.set_index(KEY).join(baseline.set_index(KEY), rsuffix="_sampled", how="inner")
    rows = []
    for metric in METRICS:
        left = joined[metric].to_numpy(dtype=float)
        right = joined[f"{metric}_sampled"].to_numpy(dtype=float)
        delta = np.abs(left - right)
        resolved = left > RESOLVED_METRIC_FLOOR
        mismatched = ~np.isclose(left, right, rtol=GATE_RTOL, atol=GATE_ATOL)
        rows.append(dict(
            metric=metric, n_pairs=len(joined),
            n_resolved=int(resolved.sum()),
            n_mismatch_resolved=int((mismatched & resolved).sum()),
            max_delta_resolved=_max_or_zero(delta[resolved]),
            n_near_zero=int((~resolved).sum()),
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


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def sensitivity_summary(paired: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    rows = []
    for eta, group in paired.groupby("eta", sort=True):
        if eta == BASELINE_ETA:
            continue
        for metric in METRICS:
            rows.append(_sensitivity_row(group, eta, metric,
                                         replicates=replicates, seed=seed))
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)


def _sensitivity_row(group: pd.DataFrame, eta: float, metric: str,
                     *, replicates: int, seed: int) -> dict:
    perturbed = group[metric].to_numpy(dtype=float)
    baseline = group[f"{metric}_0"].to_numpy(dtype=float)
    signed = perturbed - baseline
    absolute = np.abs(signed)
    relative = absolute / np.maximum(np.abs(baseline), 1e-12)
    rank_column = METRIC_RANK[metric]
    rank_changed = (group[rank_column] != group[f"{rank_column}_0"]).to_numpy()

    per_record = group.assign(_value=perturbed).groupby(KEY)["_value"]
    spreads = per_record.agg(["std", "min", "max"])
    ranges = (spreads["max"] - spreads["min"]).to_numpy(dtype=float)

    per_sample = []
    for _index, sample in group.groupby("sample", sort=True):
        per_sample.append(_spearman(sample[f"{metric}_0"].to_numpy(dtype=float),
                                    sample[metric].to_numpy(dtype=float)))
    per_sample = np.array(per_sample, dtype=float)

    fixed = group[group["sample"] == CI_SAMPLE]
    ci_low, ci_high, n_valid = _bootstrap_spearman(fixed, metric, replicates, seed)

    return dict(
        eta=eta, metric=metric,
        n_records=group.groupby(KEY).ngroups, n_sketches=group["sketch_id"].nunique(),
        n_samples=group["sample"].nunique(), n_rows=len(group),
        mean_absolute_delta=float(absolute.mean()) if absolute.size else float("nan"),
        median_absolute_delta=float(np.median(absolute)) if absolute.size else float("nan"),
        q90_absolute_delta=_quantile(absolute, 0.90),
        q95_absolute_delta=_quantile(absolute, 0.95),
        max_absolute_delta=float(absolute.max()) if absolute.size else float("nan"),
        mean_relative_delta=float(relative.mean()) if relative.size else float("nan"),
        median_relative_delta=float(np.median(relative)) if relative.size else float("nan"),
        q90_relative_delta=_quantile(relative, 0.90),
        max_relative_delta=float(relative.max()) if relative.size else float("nan"),
        signed_delta_q05=_quantile(signed, 0.05),
        signed_delta_median=float(np.median(signed)) if signed.size else float("nan"),
        signed_delta_q95=_quantile(signed, 0.95),
        n_moved=int((absolute > MOVED_TOL).sum()),
        fraction_moved=float((absolute > MOVED_TOL).mean()) if absolute.size else float("nan"),
        rank_column=METRIC_RANK[metric],
        fraction_rank_changed=float(rank_changed.mean()) if rank_changed.size else float("nan"),
        fraction_moved_with_rank_change=float(((absolute > MOVED_TOL) & rank_changed).mean())
        if absolute.size else float("nan"),
        median_within_record_sd=float(np.nanmedian(spreads["std"].to_numpy(dtype=float))),
        median_within_record_range=float(np.median(ranges)) if ranges.size else float("nan"),
        max_within_record_range=float(ranges.max()) if ranges.size else float("nan"),
        spearman_median_over_samples=float(np.nanmedian(per_sample)) if per_sample.size
        else float("nan"),
        spearman_min_over_samples=float(np.nanmin(per_sample)) if per_sample.size
        else float("nan"),
        spearman_ci_low=ci_low, spearman_ci_high=ci_high, n_valid_replicates=n_valid,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _bootstrap_spearman(frame: pd.DataFrame, metric: str, replicates: int, seed: int):
    if frame.empty:
        return float("nan"), float("nan"), 0
    baseline = frame[f"{metric}_0"].to_numpy(dtype=float)
    perturbed = frame[metric].to_numpy(dtype=float)

    def statistic(rows):
        return _spearman(baseline[rows], perturbed[rows])

    return percentile_ci(sketch_cluster_bootstrap(
        frame["sketch_id"].to_numpy(), statistic, n_replicates=replicates, seed=seed
    ))


def _response_index(responses: pd.DataFrame) -> pd.DataFrame:
    """response_metadata.csv is keyed by annotation label; records call it pred_label."""
    return responses.rename(columns={"label": "pred_label"}).set_index(
        ["sketch_id", "pred_label"]
    )


def stratify_by_free_rank(paired: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity split by how many observable directions the response is free in."""
    keys = ["sketch_id", "pred_label"]
    ranks = _response_index(responses)[["observable_free_rank", "raw_null_dim"]]
    joined = paired.join(ranks, on=keys, how="left")
    joined["free_rank_bin"] = joined["observable_free_rank"].map(_free_rank_bin)
    rows = []
    for (eta, bin_name), group in joined.groupby(["eta", "free_rank_bin"], sort=True):
        if eta == BASELINE_ETA:
            continue
        for metric in METRICS:
            absolute = (group[metric] - group[f"{metric}_0"]).abs().to_numpy(dtype=float)
            rows.append(dict(
                eta=eta, free_rank_bin=bin_name, metric=metric,
                n_rows=len(group), n_sketches=group["sketch_id"].nunique(),
                mean_absolute_delta=float(absolute.mean()) if absolute.size else np.nan,
                median_absolute_delta=float(np.median(absolute)) if absolute.size else np.nan,
                q90_absolute_delta=_quantile(absolute, 0.90),
                max_absolute_delta=float(absolute.max()) if absolute.size else np.nan,
            ))
    return pd.DataFrame(rows)


def _free_rank_bin(rank) -> str:
    if pd.isna(rank):
        return "unknown"
    for low, high in FREE_RANK_BINS:
        if low <= rank <= high:
            return f"{low}" if low == high else f"{low}-{high if high < 10**9 else '+'}"
    return "unknown"


# ---------------------------------------------------------------------------
# Exact invariance and the draw guard
# ---------------------------------------------------------------------------

def invariance_coverage(responses: pd.DataFrame) -> pd.DataFrame:
    """How often the minimum-norm choice provably cannot matter.

    ``exactly_invariant`` marks ``J_phi N = 0``: every admissible response has the same
    ``J_phi R``, so theta_full, theta_rel and S_place are identical for all of them and the
    response is not sampled. Only theta_leak, built from ``range(R)`` in raw state space,
    stays response-dependent there. The share is reported per population because it tracks
    how tightly constrained the annotation is, not anything about the metric.
    """
    usable = responses[responses["skip_reason"].isna() | (responses["skip_reason"] == "")]
    rows = []
    for name, group in [("all", usable)] + list(usable.groupby("label_kind", sort=True)):
        invariant = group["exactly_invariant"].astype(bool)
        rows.append(dict(
            label_kind=name, n_responses=len(group),
            n_sketches=group["sketch_id"].nunique(),
            n_exactly_invariant=int(invariant.sum()),
            fraction_exactly_invariant=float(invariant.mean()) if len(group) else np.nan,
            median_raw_null_dim=float(group["raw_null_dim"].median()),
            median_observable_free_rank=float(group["observable_free_rank"].median()),
            median_observable_gain_max=float(group["observable_gain_max"].median()),
        ))
    return pd.DataFrame(rows)


def guard_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Discarded-draw accounting, and how far the guard floor could move for free.

    Takes the paired frame rather than the raw samples because the draw key needs
    ``pred_label``, which only the native join carries.

    The ``would_guard_at_*`` columns are the load-bearing ones. A draw is discarded exactly
    when its smallest column gain falls below the floor, so two floors that would discard the
    same draws produce bit-identical runs. Reporting the whole ladder replaces a claim about
    the threshold being arbitrary with the measurement of how arbitrary it actually is.
    """
    rows = []
    for eta, group in paired.groupby("eta", sort=True):
        # One draw is one (response, eta, sample). Its rows repeat over every evaluation
        # record that shares the Pred side -- both GT references, and, for a model or human
        # label, every reference pair it appears under. Deduplicating on the record key
        # instead would count one model draw three times.
        draws = group.drop_duplicates(DRAW_KEY)
        gains = draws["min_observable_gain"].to_numpy(dtype=float)
        finite = gains[np.isfinite(gains)]
        row = dict(
            eta=eta, n_draws=len(draws), n_rows=len(group),
            n_columns_guarded=int(draws["n_columns_guarded"].sum()),
            n_draws_with_a_guarded_column=int((draws["n_columns_guarded"] > 0).sum()),
            min_observable_gain=float(finite.min()) if finite.size else np.nan,
            q01_observable_gain=_quantile(finite, 0.01),
        )
        row.update({f"would_guard_at_{floor:.0e}": int((finite < floor).sum())
                    for floor in GUARD_FLOOR_LADDER})
        rows.append(row)
    return pd.DataFrame(rows)


def floor_sensitivity(paired: pd.DataFrame) -> pd.DataFrame:
    """Do the reported numbers actually move when the guard floor does?

    ``guard_summary`` bounds the population a different floor would discard; this bounds the
    consequence, which is what the reported medians and Q90s rest on. Each candidate floor is
    applied post hoc by dropping the draws it would have rejected and recomputing.

    Only floors at or above the run's own floor can be evaluated: draws below it were never
    generated, so a lower floor cannot be simulated from this run's output.
    """
    rows = []
    for floor in GUARD_FLOOR_LADDER:
        kept = paired[paired["min_observable_gain"] >= floor]
        if kept.empty:
            continue
        for eta, group in kept.groupby("eta", sort=True):
            for metric in METRICS:
                absolute = (group[metric] - group[f"{metric}_0"]).abs().to_numpy(dtype=float)
                rows.append(dict(
                    gain_floor=floor, eta=eta, metric=metric, n_rows=len(group),
                    mean_absolute_delta=float(absolute.mean()) if absolute.size else np.nan,
                    median_absolute_delta=float(np.median(absolute)) if absolute.size else np.nan,
                    q90_absolute_delta=_quantile(absolute, 0.90),
                ))
    return pd.DataFrame(rows)


def floor_sensitivity_worst_shift(frame: pd.DataFrame) -> float:
    """Largest relative move of any reported statistic across the evaluable floors."""
    if frame.empty:
        return float("nan")
    worst = 0.0
    for column in ("median_absolute_delta", "q90_absolute_delta"):
        pivot = frame.pivot_table(index=["eta", "metric"], columns="gain_floor", values=column)
        reference = pivot[pivot.columns.min()]
        spread = pivot.sub(reference, axis=0).abs().div(reference.abs().where(reference != 0), axis=0)
        if spread.notna().any().any():
            worst = max(worst, float(np.nanmax(spread.to_numpy())))
    return worst


# ---------------------------------------------------------------------------
# Ordering stability
# ---------------------------------------------------------------------------

def collapsed_medians(rows: list[dict], variants: list[str]) -> tuple[dict, dict]:
    """Sec. 5 reference aggregation, then the per-variant median."""
    collapsed = aggregate(rows, list(METRICS), best=True)
    medians: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        per_variant = {}
        for variant in variants:
            values = [d[metric] for (_rid, v), d in collapsed.items()
                      if v == variant and metric in d]
            if values:
                per_variant[variant] = float(np.median(values))
        medians[metric] = per_variant
    return medians, collapsed


def variant_ordering(medians: dict, metric: str) -> list[str]:
    """Best-first, with sub-noise differences treated as ties.

    Angles are distances and similarities are not. Values are quantised at ORDERING_TOL
    before sorting and ties break on the variant name: below that scale the padded angle is
    dominated by its own sqrt(eps) floor, and sorting on it would report noise as a
    reordering.
    """
    sign = 1.0 if metric in ANGLES else -1.0
    return [v for v, _ in sorted(
        medians[metric].items(),
        key=lambda kv: (round(sign * kv[1] / ORDERING_TOL), kv[0]),
    )]


def ordering_violations(native_medians: dict, other_medians: dict, metric: str,
                        variants: list[str]) -> tuple[int, int]:
    """Inverted variant pairs, counted only where the baseline separates the two.

    Comparing sorted lists for equality would score a reordering of tied variants as a
    broken ordering, and theta_leak in particular has an aggregate median of exactly zero
    for many variants. Only pairs the baseline resolves can be preserved or broken.
    """
    native, other = native_medians[metric], other_medians[metric]
    inverted = comparable = 0
    for i, left in enumerate(variants):
        for right in variants[i + 1:]:
            if not {left, right} <= native.keys() or not {left, right} <= other.keys():
                continue
            baseline_gap = native[left] - native[right]
            if abs(baseline_gap) <= ORDERING_TOL:
                continue
            comparable += 1
            if baseline_gap * (other[left] - other[right]) < 0:
                inverted += 1
    return inverted, comparable


def flip_rate(native_collapsed: dict, other_collapsed: dict, metric: str) -> tuple[float, int]:
    """How often a per-sketch pairwise variant comparison reverses under this response."""
    by_rid: dict[str, dict[str, tuple[float, float]]] = {}
    for (rid, variant), values in native_collapsed.items():
        other = other_collapsed.get((rid, variant))
        if metric in values and other is not None and metric in other:
            by_rid.setdefault(rid, {})[variant] = (values[metric], other[metric])

    flips = comparisons = 0
    for variants in by_rid.values():
        labels = sorted(variants)
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                n_left, t_left = variants[left]
                n_right, t_right = variants[right]
                if abs(n_left - n_right) <= ORDERING_TOL or abs(t_left - t_right) <= ORDERING_TOL:
                    continue
                comparisons += 1
                if (n_left < n_right) != (t_left < t_right):
                    flips += 1
    return (flips / comparisons if comparisons else float("nan")), comparisons


def ranking_report(native: pd.DataFrame, samples: pd.DataFrame,
                   *, max_samples: int) -> pd.DataFrame:
    """Ordering stability, with each draw compared on the records it actually produced.

    Exactly-invariant responses are never sampled, and a numerically inadmissible draw can
    be rejected by the post-perturbation residual gate.  Ranking that subset against an
    ordering built on all native records would measure population drift rather than the
    response choice, so the native aggregation is rebuilt for every draw.
    """
    observed: dict[tuple, list] = {}
    rows = []
    for (eta, sample), group in samples.groupby(["eta", "sample"], sort=True):
        if eta == BASELINE_ETA or sample >= max_samples:
            continue
        covered = pd.MultiIndex.from_frame(group[KEY].drop_duplicates())
        paired_native = native[pd.MultiIndex.from_frame(native[KEY]).isin(covered)]
        variants = ordered_variants(paired_native)
        native_medians, native_collapsed = collapsed_medians(
            paired_native.to_dict("records"), variants
        )
        native_orders = {m: variant_ordering(native_medians, m) for m in METRICS}
        methods = {
            m: [v for v in native_orders[m] if is_method_variant(v)] for m in METRICS
        }
        medians, collapsed = collapsed_medians(group.to_dict("records"), variants)
        for metric in METRICS:
            order = variant_ordering(medians, metric)
            rate, _n = flip_rate(native_collapsed, collapsed, metric)
            inverted, comparable = ordering_violations(
                native_medians, medians, metric, variants
            )
            method_inverted, _n_method = ordering_violations(
                native_medians, medians, metric, methods[metric]
            )
            observed.setdefault((eta, metric), []).append(
                (inverted == 0, method_inverted == 0, rate,
                 ">".join(order), ">".join(native_orders[metric]),
                 inverted, comparable)
            )

    for (eta, metric), entries in sorted(observed.items()):
        preserved = np.array([e[0] for e in entries], dtype=float)
        method_preserved = np.array([e[1] for e in entries], dtype=float)
        rates = np.array([e[2] for e in entries], dtype=float)
        inverted = np.array([e[5] for e in entries], dtype=float)
        worst = max(entries, key=lambda e: (e[5], e[2]))
        rows.append(dict(
            eta=eta, metric=metric, n_samples=len(entries),
            fraction_ordering_preserved=float(preserved.mean()),
            fraction_method_ordering_preserved=float(method_preserved.mean()),
            n_comparable_pairs=int(min(e[6] for e in entries)),
            median_inverted_pairs=float(np.median(inverted)),
            max_inverted_pairs=float(inverted.max()),
            median_winner_flip_rate=float(np.nanmedian(rates)),
            max_winner_flip_rate=float(np.nanmax(rates)),
            native_order=worst[4],
            worst_order=worst[3],
        ))
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


def finite_anchor_summary(anchor: pd.DataFrame) -> pd.DataFrame:
    """How close the real solver's small finite response is to the minimum-norm prediction.

    Split by annotation kind and by null dimension: the minimum-norm convention is a claim
    about which branch an incremental solve picks, and the room to pick a different one is
    exactly ``dim ker(J_x)``.
    """
    usable = anchor[anchor["cosine"].notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    usable["null_dim_bin"] = pd.cut(
        usable["raw_null_dim"], [-1, 0, 5, 10, 20, 10**9],
        labels=["0", "1-5", "6-10", "11-20", "21+"],
    )
    rows = []
    for group_name, group in [("all", usable)] + [
        (f"kind={kind}", frame) for kind, frame in usable.groupby("label_kind", sort=True)
    ] + [
        (f"null_dim={name}", frame)
        for name, frame in usable.groupby("null_dim_bin", sort=True, observed=True)
    ]:
        cosine = group["cosine"].to_numpy(dtype=float)
        error = group["normalized_error"].to_numpy(dtype=float)
        rows.append(dict(
            stratum=group_name, n=len(group), n_sketches=group["sketch_id"].nunique(),
            cosine_median=float(np.median(cosine)),
            cosine_q10=_quantile(cosine, 0.10), cosine_q01=_quantile(cosine, 0.01),
            fraction_above_0_99=float((cosine > 0.99).mean()),
            normalized_error_median=float(np.median(error)),
            normalized_error_q90=_quantile(error, 0.90),
        ))
    return pd.DataFrame(rows)


def worst_cases(paired: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    keys = ["sketch_id", "pred_label"]
    ranks = _response_index(responses)[["raw_null_dim", "observable_free_rank",
                                        "observable_gain_max"]]
    carried = ["sketch_id", "pred_label", "rid", "variant", "gt", "eta", "sample",
               "raw_perturbation_ratio", "constraint_residual", "nullspace_residual",
               "min_observable_gain", "n_columns_guarded"]
    frames = []
    for metric in METRICS:
        signed = paired[metric] - paired[f"{metric}_0"]
        top = signed.abs().nlargest(WORST_CASES).index
        frames.append(paired.loc[top, carried].assign(
            metric=metric,
            baseline_value=paired.loc[top, f"{metric}_0"],
            perturbed_value=paired.loc[top, metric],
            signed_delta=signed.loc[top],
            absolute_delta=signed.loc[top].abs(),
        ))
    worst = pd.concat(frames, ignore_index=True)
    return worst.join(ranks, on=keys, how="left")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_delta_vs_eta(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for metric in METRICS:
        per_metric = summary[summary["metric"] == metric].sort_values("eta")
        ax.plot(per_metric["eta"], np.maximum(per_metric["median_absolute_delta"], 1e-18),
                marker="o", label=metric)
        ax.plot(per_metric["eta"], np.maximum(per_metric["q90_absolute_delta"], 1e-18),
                marker="_", linestyle=":", color=ax.lines[-1].get_color())
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$  (observable-space size of the added free motion)")
    ax.set_ylabel(r"$|d_\eta - d_0|$  (solid: median, dotted: Q90)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rank_correlation(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for metric in METRICS:
        per_metric = summary[summary["metric"] == metric].sort_values("eta")
        ax.plot(per_metric["eta"], per_metric["spearman_median_over_samples"],
                marker="o", label=metric)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Spearman(baseline, perturbed), median over samples")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_by_free_rank(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    bins = [b for b in frame["free_rank_bin"].unique() if b != "unknown"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for bin_name in sorted(bins):
        for metric in ("theta_full", "placement_sim"):
            per_bin = frame[(frame["free_rank_bin"] == bin_name)
                            & (frame["metric"] == metric)].sort_values("eta")
            ax.plot(per_bin["eta"], np.maximum(per_bin["median_absolute_delta"], 1e-18),
                    marker="o", linestyle="-" if metric == "theta_full" else "--",
                    label=f"{metric} | free rank {bin_name}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"median $|d_\eta - d_0|$")
    ax.set_title("Sensitivity by observable free rank")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="run_dir", default="artifacts/rebuttal/pseudoinverse")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ordering-samples", type=int, default=20,
                        help="samples per eta used for the ordering analysis")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    native, samples = load_run(run_dir)
    responses = pd.read_csv(run_dir / "response_metadata.csv")

    coverage = invariance_coverage(responses)
    coverage.to_csv(run_dir / "invariance_coverage.csv", index=False)
    print("=== exact invariance (J_phi N = 0: the choice provably cannot matter) ===")
    print(coverage.to_string(index=False))

    gate = identity_report(native, samples)
    gate.to_csv(run_dir / "eta0_gate.csv", index=False)
    print("=== eta = 0 gate ===")
    print(gate.to_string(index=False))
    if identity_gate_failed(gate):
        raise SystemExit("eta = 0 does not reproduce pair_paper_metrics; not aggregating")

    paired = paired_samples(native, samples)
    paired = paired[paired["eta"] != BASELINE_ETA]
    if paired.empty:
        print("no perturbed samples in this run")
        return

    summary = sensitivity_summary(paired, replicates=args.replicates, seed=args.seed)
    stratified = stratify_by_free_rank(paired, responses)
    guards = guard_summary(paired)
    floors = floor_sensitivity(paired)
    ranking = ranking_report(native, samples, max_samples=args.ordering_samples)
    worst = worst_cases(paired, responses)

    summary.to_csv(run_dir / "sensitivity_summary.csv", index=False)
    stratified.to_csv(run_dir / "sensitivity_by_observable_free_rank.csv", index=False)
    guards.to_csv(run_dir / "guard_summary.csv", index=False)
    floors.to_csv(run_dir / "floor_sensitivity.csv", index=False)
    ranking.to_csv(run_dir / "ranking_changes.csv", index=False)
    worst.to_csv(run_dir / "worst_cases.csv", index=False)

    if not args.no_figures:
        figures = run_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        plot_delta_vs_eta(summary, figures / "metric_delta_vs_eta.pdf")
        plot_rank_correlation(summary, figures / "rank_correlation_vs_eta.pdf")
        plot_by_free_rank(stratified, figures / "sensitivity_by_observable_free_rank.pdf")

    print("\n=== sensitivity (responses that can move) ===")
    print(summary[["eta", "metric", "mean_absolute_delta", "median_absolute_delta",
                   "q90_absolute_delta", "max_absolute_delta", "fraction_moved",
                   "spearman_median_over_samples"]].to_string(index=False))
    print("\n=== draw guard ===")
    print(guards.to_string(index=False))
    total = int(guards["n_draws"].sum())
    ladder = " ".join(
        f"{floor:.0e}:{int(guards[f'would_guard_at_{floor:.0e}'].sum())}"
        for floor in GUARD_FLOOR_LADDER
    )
    print(f"  draws discarded at each candidate floor (of {total}): {ladder}")
    shift = floor_sensitivity_worst_shift(floors)
    print(f"  largest relative move of any reported median/Q90 across those floors: {shift:.2%}")
    print("\n=== ordering ===")
    print(ranking[["eta", "metric", "fraction_method_ordering_preserved",
                   "fraction_ordering_preserved", "n_comparable_pairs",
                   "median_inverted_pairs", "median_winner_flip_rate"]].to_string(index=False))
    anchor_path = run_dir / "finite_anchor.csv"
    if anchor_path.exists():
        anchor = finite_anchor_summary(pd.read_csv(anchor_path))
        anchor.to_csv(run_dir / "finite_anchor_summary.csv", index=False)
        print("\n=== finite solver anchor (auxiliary) ===")
        print(anchor.to_string(index=False))

    print(f"\nwrote -> {run_dir}")


if __name__ == "__main__":
    main()
