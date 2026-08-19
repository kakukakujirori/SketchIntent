"""Does the multi-reference aggregation let one prediction borrow different humans per metric?

Sec. 5 scores a prediction against K human references and keeps, independently per metric, the
most favourable one. A prediction could therefore match the relational behaviour of annotator A
and the dimensional placement of annotator B without being close to either complete annotation.

Nothing is recomputed here. ``motion_eval.csv`` already stores one row per (rid, variant,
reference), i.e. the per-reference scores *before* the min/max aggregation in
``11_analyze_paper_numbers.py``; the convergence, geometry-preservation and leave-one-out
filtering are inherited with it. This script only re-aggregates that file, so its
``independent(all)`` column reproduces Table 1 exactly and is asserted to.

Three things are measured: how often the per-metric best references disagree (Exp. 1), what a
rank-only joint selection picks and how often it is ambiguous (Exp. 2), how much each metric
loses on that shared reference (Exp. 3), and what Table 1 looks like under it (Exp. 4).

Two protocol points that shape every number below.

``atol``. Best-reference sets are tie-aware: ``|s - best| <= atol`` with an absolute tolerance
and no relative term, because the question is whether two references differ by more than the
metric's own numerical resolution. The primary value is ``RESOLVED_METRIC_FLOOR = 1e-6``, which
common.py already documents as the sqrt(eps) noise floor of the padded angle -- it is not a
threshold invented for this experiment. ``tolerance_sweep.csv`` reports {0, 1e-9, 1e-6, 1e-3}.

Rank-sum ties. With K = 2 a 2-2 split across the four metrics leaves the two references with the
same rank sum, and rank information alone cannot break it. Those samples are flagged
``ambiguous_joint_selection`` and never resolved by reference index or by raw-score averaging.
They are also not dropped: the envelope columns bound every metric over the admissible tie
resolutions. The envelope is a *per-metric* bound -- worst and best are attained by different tie
resolutions, so the envelope columns must not be read as one simultaneously realizable Table 1.
The apples-to-apples comparison is independent(unique) vs shared(unique), which holds the sample
set fixed and varies only the aggregation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rebuttal_exp.common import (  # noqa: E402
    RESOLVED_METRIC_FLOOR,
    finalize_run_config,
    percentile_ci,
    prepare_output_dir,
    sketch_cluster_bootstrap,
    write_csv,
    write_environment,
)

ANGLES = ("theta_full", "theta_rel", "theta_leak")
PLACE = "S_place"
METRICS = ANGLES + (PLACE,)
LOWER_IS_BETTER = {m: (m in ANGLES) for m in METRICS}
PLACE_SOURCE_COLUMNS = ("placement_sim", "asgn_sim")
"""S_place was renamed placement_sim after the Table 1 run; both spell the same quantity."""

PRIMARY_ATOL = RESOLVED_METRIC_FLOOR
TOLERANCE_SWEEP = (0.0, 1e-9, RESOLVED_METRIC_FLOOR, 1e-3)
RANK_SUM_TOL = 1e-12
"""Rank sums are sums of averaged integer ranks, so equality is exact; the tolerance only
guards the float representation."""
REGRET_SIGN_TOL = 1e-12
FLIP_TOL = 1e-6

SOURCES = ("heldout", "base", "solver_rl")
"""The three rows the reviewer's question is about. Everything is computed for all variants."""
VARIANT_ORDER = (
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
)
MODELS = ("base", "solver_rl")
PAIR_PROBES = (
    ("theta_rel", PLACE),
    ("theta_full", PLACE),
    ("theta_full", "theta_rel"),
    ("theta_rel", "theta_leak"),
)
"""rel x place is the primary probe: it is exactly 'relational behaviour and dimensional
placement prefer incompatible annotations'."""

TABLE1_REFERENCE = {
    "heldout": (0.000, 0.000, 0.000, 0.831, 858),
    "base": (0.839, 0.707, 0.029, 0.283, 552),
    "solver_rl": (0.488, 0.471, 0.304, 0.628, 810),
}
"""paper/sec/5_experiments.tex tab:controls, for the reproduction gate."""

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """One evaluated prediction: (rid, variant) scored against every available reference."""

    rid: str
    batch: str
    filename: str
    variant: str
    refs: tuple[str, ...]
    values: dict[str, dict[str, float]]  # metric -> reference -> score

    @property
    def cluster(self) -> str:
        return f"{self.batch}/{self.filename}"


def load_samples(path: Path) -> tuple[list[Sample], dict, str]:
    """Group motion_eval rows into per-(rid, variant) samples, counting every exclusion."""
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())

    place_column = next((c for c in PLACE_SOURCE_COLUMNS if c in fieldnames), None)
    if place_column is None:
        raise SystemExit(f"{path} has none of {PLACE_SOURCE_COLUMNS}; cannot read S_place")
    source_columns = {**{m: m for m in ANGLES}, PLACE: place_column}

    groups: dict[tuple[str, str], dict] = {}
    n_incomplete = 0
    for row in rows:
        # register the group before parsing, so a sample whose rows are all unusable still
        # shows up in the exclusion counts instead of vanishing
        group = groups.setdefault(
            (row["rid"], row["variant"]),
            dict(batch=row["batch"], filename=row["filename"], scores={}),
        )
        try:
            scores = {m: float(row[col]) for m, col in source_columns.items()}
        except (TypeError, ValueError, KeyError):
            n_incomplete += 1
            continue
        if not all(np.isfinite(v) for v in scores.values()):
            n_incomplete += 1
            continue
        group["scores"][row["gt"]] = scores

    samples, n_too_few_refs = [], 0
    for (rid, variant), group in groups.items():
        refs = tuple(sorted(group["scores"]))
        if len(refs) < 2:
            n_too_few_refs += 1
            continue
        values = {m: {k: group["scores"][k][m] for k in refs} for m in METRICS}
        samples.append(
            Sample(
                rid=rid,
                batch=group["batch"],
                filename=group["filename"],
                variant=variant,
                refs=refs,
                values=values,
            )
        )
    samples.sort(key=lambda s: (s.variant, s.rid))

    counts = dict(
        source_csv=str(path),
        rows_total=len(rows),
        rows_with_incomplete_metrics=n_incomplete,
        rows_used=len(rows) - n_incomplete,
        groups_total=len(groups),
        groups_dropped_fewer_than_two_refs=n_too_few_refs,
        samples=len(samples),
        samples_by_variant={
            v: sum(1 for s in samples if s.variant == v) for v in sorted({s.variant for s in samples})
        },
        references_per_sample={
            str(k): v
            for k, v in sorted(
                {len(s.refs): sum(1 for t in samples if len(t.refs) == len(s.refs)) for s in samples}.items()
            )
        },
        s_place_source_column=place_column,
    )
    return samples, counts, place_column


# ---------------------------------------------------------------------------
# Tie-aware primitives (K >= 2)
# ---------------------------------------------------------------------------

def best_score(scores: dict[str, float], metric: str) -> float:
    return min(scores.values()) if LOWER_IS_BETTER[metric] else max(scores.values())


def best_reference_set(scores: dict[str, float], metric: str, atol: float) -> frozenset[str]:
    """{k : s[k] is numerically equal to the best score}, absolute tolerance, rtol = 0."""
    target = best_score(scores, metric)
    return frozenset(k for k, v in scores.items() if abs(v - target) <= atol)


def best_reference_sets(sample: Sample, atol: float) -> dict[str, frozenset[str]]:
    return {m: best_reference_set(sample.values[m], m, atol) for m in METRICS}


def average_ranks(scores: dict[str, float], metric: str, atol: float) -> dict[str, float]:
    """Rank 1 = best; tied values share the average of the positions they occupy.

    Ties are grouped by single linkage along the sorted order: a reference joins the current
    group when it lies within ``atol`` of the group's last member. Tolerance ties are not
    transitive, so at K >= 3 the chaining is a choice; it is exact for the K = 2 data here.
    """
    sign = 1.0 if LOWER_IS_BETTER[metric] else -1.0
    order = sorted(scores, key=lambda k: (sign * scores[k], k))
    groups: list[list[str]] = []
    for key in order:
        if groups and abs(scores[key] - scores[groups[-1][-1]]) <= atol:
            groups[-1].append(key)
        else:
            groups.append([key])
    ranks, position = {}, 1
    for group in groups:
        shared = position + (len(group) - 1) / 2
        for key in group:
            ranks[key] = shared
        position += len(group)
    return ranks


def rank_table(sample: Sample, atol: float) -> dict[str, dict[str, float]]:
    return {m: average_ranks(sample.values[m], m, atol) for m in METRICS}


def rank_sums(ranks: dict[str, dict[str, float]], refs: tuple[str, ...]) -> dict[str, float]:
    """R_k, used only to select a shared reference. Never a CAD metric."""
    return {k: sum(ranks[m][k] for m in METRICS) for k in refs}


def joint_selection(sums: dict[str, float]) -> tuple[str | None, tuple[str, ...]]:
    """Unique rank-sum winner, or (None, tied winners) when rank information cannot decide."""
    smallest = min(sums.values())
    winners = tuple(sorted(k for k, v in sums.items() if abs(v - smallest) <= RANK_SUM_TOL))
    return (winners[0] if len(winners) == 1 else None), winners


def regret(sample: Sample, metric: str, reference: str) -> float:
    scores = sample.values[metric]
    if LOWER_IS_BETTER[metric]:
        return scores[reference] - min(scores.values())
    return max(scores.values()) - scores[reference]


def envelope(sample: Sample, metric: str, candidates: tuple[str, ...]) -> tuple[float, float]:
    """(worst, best) value of one metric over the admissible rank-optimal references."""
    vals = [sample.values[metric][k] for k in candidates]
    if LOWER_IS_BETTER[metric]:
        return max(vals), min(vals)
    return min(vals), max(vals)


# ---------------------------------------------------------------------------
# Per-sample analysis
# ---------------------------------------------------------------------------

def analyse_sample(sample: Sample, atol: float) -> dict:
    best = best_reference_sets(sample, atol)
    ranks = rank_table(sample, atol)
    sums = rank_sums(ranks, sample.refs)
    k_joint, winners = joint_selection(sums)
    coherent = bool(frozenset.intersection(*best.values()))
    record = dict(
        sample=sample,
        best=best,
        ranks=ranks,
        rank_sums=sums,
        k_joint=k_joint,
        rank_optimal=winners,
        ambiguous=k_joint is None,
        coherent=coherent,
        unique_best={m: len(best[m]) == 1 for m in METRICS},
        incompatible={pair: not (best[pair[0]] & best[pair[1]]) for pair in PAIR_PROBES},
        independent={m: best_score(sample.values[m], m) for m in METRICS},
    )
    record["shared"] = {m: sample.values[m][k_joint] for m in METRICS} if k_joint else None
    record["regret"] = {m: regret(sample, m, k_joint) for m in METRICS} if k_joint else None
    record["envelope"] = {m: envelope(sample, m, winners) for m in METRICS}
    return record


def analyse(samples: list[Sample], atol: float) -> list[dict]:
    records = [analyse_sample(s, atol) for s in samples]
    for record in records:
        if record["regret"] is None:
            continue
        for metric, value in record["regret"].items():
            if value < -REGRET_SIGN_TOL:
                raise AssertionError(
                    f"negative regret {value:.3e} for {metric} on "
                    f"{record['sample'].rid}/{record['sample'].variant}"
                )
    return records


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def by_variant(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for record in records:
        out.setdefault(record["sample"].variant, []).append(record)
    return out


def ordered_variants(groups: dict[str, list[dict]]) -> list[str]:
    known = [v for v in VARIANT_ORDER if v in groups]
    return known + sorted(v for v in groups if v not in VARIANT_ORDER)


def rate_with_ci(records: list[dict], flags: list[bool]) -> tuple[float, float, float, int]:
    if not records:
        return float("nan"), float("nan"), float("nan"), 0
    values = np.asarray(flags, dtype=float)
    clusters = [r["sample"].cluster for r in records]
    replicates = sketch_cluster_bootstrap(
        clusters,
        lambda rows: float(values[rows].mean()),
        n_replicates=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    low, high, n_valid = percentile_ci(replicates)
    return float(values.mean()), low, high, n_valid


def median_difference_ci(records: list[dict], metric: str) -> tuple[float, float, int]:
    """Paired CI for median(shared) - median(independent) on the unique-k_joint subset."""
    unique = [r for r in records if r["k_joint"]]
    if len(unique) < 8:
        return float("nan"), float("nan"), 0
    shared = np.asarray([r["shared"][metric] for r in unique])
    independent = np.asarray([r["independent"][metric] for r in unique])
    replicates = sketch_cluster_bootstrap(
        [r["sample"].cluster for r in unique],
        lambda rows: float(np.median(shared[rows]) - np.median(independent[rows])),
        n_replicates=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    low, high, n_valid = percentile_ci(replicates)
    return low, high, n_valid


def quantile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


# ---------------------------------------------------------------------------
# Experiment 1: best-reference consistency
# ---------------------------------------------------------------------------

EXP1_COLUMNS = [
    "variant", "atol", "n",
    "coherent_rate", "coherent_ci_low", "coherent_ci_high",
    "incoherent_rate",
    "rel_vs_place_incompatible_rate", "rel_vs_place_ci_low", "rel_vs_place_ci_high",
    "full_vs_place_incompatible_rate", "full_vs_rel_incompatible_rate",
    "rel_vs_leak_incompatible_rate",
    "n_all_unique_best", "unique_all_four_agree_rate", "unique_exactly_three_agree_rate",
    "unique_split_rate",
    "ambiguous_joint_selection_rate", "n_unique_joint", "n_ambiguous_joint",
    "n_bootstrap_replicates",
]


def exp1_table(records: list[dict], atol: float) -> list[dict]:
    rows = []
    groups = by_variant(records)
    for variant in ordered_variants(groups):
        group = groups[variant]
        n = len(group)
        coherent, c_low, c_high, n_valid = rate_with_ci(group, [r["coherent"] for r in group])
        probe = PAIR_PROBES[0]
        rel_place, rp_low, rp_high, _ = rate_with_ci(group, [r["incompatible"][probe] for r in group])
        unique_best = [r for r in group if all(r["unique_best"].values())]
        agree_counts = []
        for record in unique_best:
            picks = [next(iter(record["best"][m])) for m in METRICS]
            agree_counts.append(max(picks.count(p) for p in set(picks)))
        n_unique = len(unique_best)
        rows.append(
            dict(
                variant=variant,
                atol=atol,
                n=n,
                coherent_rate=coherent,
                coherent_ci_low=c_low,
                coherent_ci_high=c_high,
                incoherent_rate=1.0 - coherent,
                rel_vs_place_incompatible_rate=rel_place,
                rel_vs_place_ci_low=rp_low,
                rel_vs_place_ci_high=rp_high,
                full_vs_place_incompatible_rate=st.fmean(
                    [float(r["incompatible"][("theta_full", PLACE)]) for r in group]
                ),
                full_vs_rel_incompatible_rate=st.fmean(
                    [float(r["incompatible"][("theta_full", "theta_rel")]) for r in group]
                ),
                rel_vs_leak_incompatible_rate=st.fmean(
                    [float(r["incompatible"][("theta_rel", "theta_leak")]) for r in group]
                ),
                n_all_unique_best=n_unique,
                unique_all_four_agree_rate=(
                    sum(1 for c in agree_counts if c == 4) / n_unique if n_unique else float("nan")
                ),
                unique_exactly_three_agree_rate=(
                    sum(1 for c in agree_counts if c == 3) / n_unique if n_unique else float("nan")
                ),
                unique_split_rate=(
                    sum(1 for c in agree_counts if c <= 2) / n_unique if n_unique else float("nan")
                ),
                ambiguous_joint_selection_rate=st.fmean([float(r["ambiguous"]) for r in group]),
                n_unique_joint=sum(1 for r in group if r["k_joint"]),
                n_ambiguous_joint=sum(1 for r in group if r["ambiguous"]),
                n_bootstrap_replicates=n_valid,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Experiment 3: joint-reference regret
# ---------------------------------------------------------------------------

EXP3_COLUMNS = [
    "variant", "metric", "atol", "n_unique_joint",
    "fraction_zero_within_atol", "fraction_exactly_zero",
    "median", "mean", "p75", "p90", "p95", "max",
]


def exp3_table(records: list[dict], atol: float) -> list[dict]:
    rows = []
    groups = by_variant(records)
    for variant in ordered_variants(groups):
        unique = [r for r in groups[variant] if r["k_joint"]]
        for metric in METRICS:
            vals = [r["regret"][metric] for r in unique]
            n = len(vals)
            rows.append(
                dict(
                    variant=variant,
                    metric=metric,
                    atol=atol,
                    n_unique_joint=n,
                    fraction_zero_within_atol=(
                        sum(1 for v in vals if abs(v) <= atol) / n if n else float("nan")
                    ),
                    fraction_exactly_zero=(sum(1 for v in vals if v == 0.0) / n if n else float("nan")),
                    median=st.median(vals) if vals else float("nan"),
                    mean=st.fmean(vals) if vals else float("nan"),
                    p75=quantile(vals, 75),
                    p90=quantile(vals, 90),
                    p95=quantile(vals, 95),
                    max=max(vals) if vals else float("nan"),
                )
            )
    return rows


SPREAD_COLUMNS = [
    "variant", "metric", "atol", "n_all_samples",
    "fraction_zero_within_atol", "median", "mean", "p75", "p90", "p95", "max",
]


def reference_spread_table(records: list[dict], atol: float) -> list[dict]:
    """How much the reference choice can move a metric, on every sample.

    Exp. 3's regret is defined only where a unique k_joint exists, so it cannot speak for the
    ambiguous samples. This is the selection-free companion: max_k s[m,k] - min_k s[m,k], the
    full range a metric spans over the references, computed on all samples regardless of how
    (or whether) a shared reference could be chosen. Regret is bounded above by this spread.
    """
    rows = []
    groups = by_variant(records)
    for variant in ordered_variants(groups):
        group = groups[variant]
        for metric in METRICS:
            vals = [
                max(r["sample"].values[metric].values()) - min(r["sample"].values[metric].values())
                for r in group
            ]
            n = len(vals)
            rows.append(
                dict(
                    variant=variant,
                    metric=metric,
                    atol=atol,
                    n_all_samples=n,
                    fraction_zero_within_atol=sum(1 for v in vals if v <= atol) / n if n else float("nan"),
                    median=st.median(vals) if vals else float("nan"),
                    mean=st.fmean(vals) if vals else float("nan"),
                    p75=quantile(vals, 75),
                    p90=quantile(vals, 90),
                    p95=quantile(vals, 95),
                    max=max(vals) if vals else float("nan"),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Experiment 4: independent vs shared Table 1
# ---------------------------------------------------------------------------

EXP4_COLUMNS = [
    "variant", "metric", "atol",
    "n_all", "n_unique_joint", "n_ambiguous_joint", "ambiguous_fraction",
    "independent_all", "independent_unique_subset", "shared_unique_subset",
    "shared_minus_independent_unique_subset",
    "shared_minus_independent_ci_low", "shared_minus_independent_ci_high",
    "envelope_worst_all", "envelope_best_all",
]


def exp4_table(records: list[dict], atol: float) -> list[dict]:
    rows = []
    groups = by_variant(records)
    for variant in ordered_variants(groups):
        group = groups[variant]
        unique = [r for r in group if r["k_joint"]]
        for metric in METRICS:
            independent_all = [r["independent"][metric] for r in group]
            independent_unique = [r["independent"][metric] for r in unique]
            shared_unique = [r["shared"][metric] for r in unique]
            worst = [r["envelope"][metric][0] for r in group]
            best = [r["envelope"][metric][1] for r in group]
            low, high, _ = median_difference_ci(group, metric)
            rows.append(
                dict(
                    variant=variant,
                    metric=metric,
                    atol=atol,
                    n_all=len(group),
                    n_unique_joint=len(unique),
                    n_ambiguous_joint=len(group) - len(unique),
                    ambiguous_fraction=(len(group) - len(unique)) / len(group) if group else float("nan"),
                    independent_all=st.median(independent_all) if independent_all else float("nan"),
                    independent_unique_subset=st.median(independent_unique) if unique else float("nan"),
                    shared_unique_subset=st.median(shared_unique) if unique else float("nan"),
                    shared_minus_independent_unique_subset=(
                        st.median(shared_unique) - st.median(independent_unique) if unique else float("nan")
                    ),
                    shared_minus_independent_ci_low=low,
                    shared_minus_independent_ci_high=high,
                    envelope_worst_all=st.median(worst) if worst else float("nan"),
                    envelope_best_all=st.median(best) if best else float("nan"),
                )
            )
    return rows


SUBSET_COLUMNS = [
    "variant", "metric", "atol", "n_unique_joint", "n_ambiguous_joint",
    "independent_median_unique_subset", "independent_median_ambiguous_subset",
    "unique_minus_ambiguous",
]


def subset_diagnostic(records: list[dict], atol: float) -> list[dict]:
    """Is the unique-k_joint subset a biased sample?

    Dropping the ambiguous samples is a selection, not a random subsample: both sides are scored
    under the *same* independent aggregation here, so any difference is the selection alone. This
    is what separates 'the shared reference moved the metric' from 'the retained sketches were
    different ones', and it is why the envelope columns are computed over all samples.
    """
    rows = []
    groups = by_variant(records)
    for variant in ordered_variants(groups):
        unique = [r for r in groups[variant] if r["k_joint"]]
        ambiguous = [r for r in groups[variant] if r["ambiguous"]]
        for metric in METRICS:
            left = [r["independent"][metric] for r in unique]
            right = [r["independent"][metric] for r in ambiguous]
            rows.append(
                dict(
                    variant=variant,
                    metric=metric,
                    atol=atol,
                    n_unique_joint=len(left),
                    n_ambiguous_joint=len(right),
                    independent_median_unique_subset=st.median(left) if left else float("nan"),
                    independent_median_ambiguous_subset=st.median(right) if right else float("nan"),
                    unique_minus_ambiguous=(
                        st.median(left) - st.median(right) if left and right else float("nan")
                    ),
                )
            )
    return rows


def check_table1_reproduction(rows: list[dict]) -> list[str]:
    """independent(all) must reproduce tab:controls, or the loader disagrees with Sec. 5."""
    lookup = {(r["variant"], r["metric"]): r for r in rows}
    problems = []
    for variant, expected in TABLE1_REFERENCE.items():
        for metric, value in zip(METRICS, expected[:4]):
            row = lookup.get((variant, metric))
            if row is None:
                problems.append(f"{variant}/{metric}: missing")
                continue
            if abs(row["independent_all"] - value) > 5e-4:
                problems.append(
                    f"{variant}/{metric}: independent_all={row['independent_all']:.4f} "
                    f"vs Table 1 {value:.3f}"
                )
            if row["n_all"] != expected[4]:
                problems.append(f"{variant}/{metric}: n={row['n_all']} vs Table 1 {expected[4]}")
    return problems


# ---------------------------------------------------------------------------
# Conclusion checks: paired tests and per-sketch model flips
# ---------------------------------------------------------------------------

PAIRED_COLUMNS = [
    "comparison", "protocol", "metric", "n", "n_first_better", "n_second_better", "n_tied",
    "z", "p_value",
]


def _paired_stats(first: list[float], second: list[float], metric: str) -> dict:
    """z > 0 means ``first`` is better (lower angle / higher S_place)."""
    sign = 1.0 if LOWER_IS_BETTER[metric] else -1.0
    diffs = [sign * (b - a) for a, b in zip(first, second)]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    out = dict(
        n=len(diffs),
        n_first_better=sum(1 for d in diffs if d > 1e-12),
        n_second_better=sum(1 for d in diffs if d < -1e-12),
        n_tied=len(diffs) - len(nonzero),
        z=float("nan"),
        p_value=float("nan"),
    )
    if len(nonzero) >= 8:
        _stat, p = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
        out["p_value"] = float(p)
        out["z"] = float(norm.isf(p / 2) * (1 if st.fmean(diffs) > 0 else -1))
    return out


def _values_by_rid(records: list[dict], variant: str, protocol: str) -> dict[str, dict[str, float]]:
    out = {}
    for record in records:
        if record["sample"].variant != variant:
            continue
        if protocol == "independent":
            out[record["sample"].rid] = record["independent"]
        elif record["k_joint"]:
            out[record["sample"].rid] = record["shared"]
    return out


def paired_tests(records: list[dict]) -> list[dict]:
    rows = []
    for protocol in ("independent", "shared"):
        held = _values_by_rid(records, "heldout", protocol)
        for model in MODELS:
            pred = _values_by_rid(records, model, protocol)
            common = sorted(set(held) & set(pred))
            for metric in METRICS:
                stats = _paired_stats(
                    [held[r][metric] for r in common], [pred[r][metric] for r in common], metric
                )
                rows.append(dict(comparison=f"heldout_vs_{model}", protocol=protocol, metric=metric, **stats))
        base = _values_by_rid(records, "base", protocol)
        solver = _values_by_rid(records, "solver_rl", protocol)
        common = sorted(set(base) & set(solver))
        for metric in METRICS:
            stats = _paired_stats(
                [base[r][metric] for r in common], [solver[r][metric] for r in common], metric
            )
            rows.append(dict(comparison="base_vs_solver_rl", protocol=protocol, metric=metric, **stats))
    return rows


FLIP_COLUMNS = [
    "metric", "n_common_rids", "n_flips", "flip_rate",
    "n_independent_base_better", "n_independent_solver_better", "n_independent_tie",
    "n_shared_base_better", "n_shared_solver_better", "n_shared_tie",
]


def _verdict(a: float, b: float, metric: str) -> str:
    if abs(a - b) <= FLIP_TOL:
        return "tie"
    better_is_lower = LOWER_IS_BETTER[metric]
    return "base" if ((a < b) == better_is_lower) else "solver_rl"


def model_flips(records: list[dict]) -> list[dict]:
    """Does the shared protocol change which model wins on an individual sketch?"""
    base_ind = _values_by_rid(records, "base", "independent")
    solver_ind = _values_by_rid(records, "solver_rl", "independent")
    base_sh = _values_by_rid(records, "base", "shared")
    solver_sh = _values_by_rid(records, "solver_rl", "shared")
    common = sorted(set(base_ind) & set(solver_ind) & set(base_sh) & set(solver_sh))
    rows = []
    for metric in METRICS:
        independent = [_verdict(base_ind[r][metric], solver_ind[r][metric], metric) for r in common]
        shared = [_verdict(base_sh[r][metric], solver_sh[r][metric], metric) for r in common]
        flips = sum(
            1
            for a, b in zip(independent, shared)
            if a != b and a != "tie" and b != "tie"
        )
        rows.append(
            dict(
                metric=metric,
                n_common_rids=len(common),
                n_flips=flips,
                flip_rate=flips / len(common) if common else float("nan"),
                n_independent_base_better=independent.count("base"),
                n_independent_solver_better=independent.count("solver_rl"),
                n_independent_tie=independent.count("tie"),
                n_shared_base_better=shared.count("base"),
                n_shared_solver_better=shared.count("solver_rl"),
                n_shared_tie=shared.count("tie"),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Per-sample dumps
# ---------------------------------------------------------------------------

PER_SAMPLE_COLUMNS = (
    ["rid", "batch", "filename", "variant", "n_refs", "refs", "atol"]
    + [f"values_{m}" for m in METRICS]
    + [f"best_set_{m}" for m in METRICS]
    + ["coherent", "coherent_reference_set"]
    + [f"incompatible_{a}_vs_{b}" for a, b in PAIR_PROBES]
    + [f"rank_{m}" for m in METRICS]
    + ["rank_sums", "rank_optimal_references", "k_joint", "ambiguous_joint_selection"]
    + [f"independent_{m}" for m in METRICS]
    + [f"shared_{m}" for m in METRICS]
    + [f"regret_{m}" for m in METRICS]
    + [f"envelope_worst_{m}" for m in METRICS]
    + [f"envelope_best_{m}" for m in METRICS]
)


def _json_cell(mapping) -> str:
    return json.dumps(mapping, sort_keys=True)


def per_sample_rows(records: list[dict], atol: float) -> list[dict]:
    rows = []
    for record in records:
        sample = record["sample"]
        row = dict(
            rid=sample.rid,
            batch=sample.batch,
            filename=sample.filename,
            variant=sample.variant,
            n_refs=len(sample.refs),
            refs=_json_cell(list(sample.refs)),
            atol=atol,
            coherent=record["coherent"],
            coherent_reference_set=_json_cell(sorted(frozenset.intersection(*record["best"].values()))),
            rank_sums=_json_cell(record["rank_sums"]),
            rank_optimal_references=_json_cell(list(record["rank_optimal"])),
            k_joint=record["k_joint"] or "",
            ambiguous_joint_selection=record["ambiguous"],
        )
        for metric in METRICS:
            row[f"values_{metric}"] = _json_cell(sample.values[metric])
            row[f"best_set_{metric}"] = _json_cell(sorted(record["best"][metric]))
            row[f"rank_{metric}"] = _json_cell(record["ranks"][metric])
            row[f"independent_{metric}"] = record["independent"][metric]
            row[f"shared_{metric}"] = record["shared"][metric] if record["shared"] else ""
            row[f"regret_{metric}"] = record["regret"][metric] if record["regret"] else ""
            worst, best = record["envelope"][metric]
            row[f"envelope_worst_{metric}"] = worst
            row[f"envelope_best_{metric}"] = best
        for first, second in PAIR_PROBES:
            row[f"incompatible_{first}_vs_{second}"] = record["incompatible"][(first, second)]
        rows.append(row)
    return rows


PER_REFERENCE_COLUMNS = (
    ["rid", "batch", "filename", "variant", "reference", "atol"]
    + list(METRICS)
    + [f"rank_{m}" for m in METRICS]
    + [f"in_best_set_{m}" for m in METRICS]
    + ["rank_sum", "is_k_joint", "is_rank_optimal", "ambiguous_joint_selection"]
)


def per_reference_rows(records: list[dict], atol: float) -> list[dict]:
    rows = []
    for record in records:
        sample = record["sample"]
        for reference in sample.refs:
            row = dict(
                rid=sample.rid,
                batch=sample.batch,
                filename=sample.filename,
                variant=sample.variant,
                reference=reference,
                atol=atol,
                rank_sum=record["rank_sums"][reference],
                is_k_joint=reference == record["k_joint"],
                is_rank_optimal=reference in record["rank_optimal"],
                ambiguous_joint_selection=record["ambiguous"],
            )
            for metric in METRICS:
                row[metric] = sample.values[metric][reference]
                row[f"rank_{metric}"] = record["ranks"][metric][reference]
                row[f"in_best_set_{metric}"] = reference in record["best"][metric]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Tolerance sweep
# ---------------------------------------------------------------------------

SWEEP_COLUMNS = [
    "atol", "variant", "n", "coherent_rate", "rel_vs_place_incompatible_rate",
    "ambiguous_joint_selection_rate", "n_unique_joint",
    "metric", "independent_all", "independent_unique_subset", "shared_unique_subset",
    "envelope_worst_all", "envelope_best_all",
]


def sweep_rows(samples: list[Sample], atol: float) -> list[dict]:
    records = analyse(samples, atol)
    exp1 = {r["variant"]: r for r in exp1_table(records, atol)}
    rows = []
    for row in exp4_table(records, atol):
        if row["variant"] not in SOURCES:
            continue
        head = exp1[row["variant"]]
        rows.append(
            dict(
                atol=atol,
                variant=row["variant"],
                n=head["n"],
                coherent_rate=head["coherent_rate"],
                rel_vs_place_incompatible_rate=head["rel_vs_place_incompatible_rate"],
                ambiguous_joint_selection_rate=head["ambiguous_joint_selection_rate"],
                n_unique_joint=head["n_unique_joint"],
                metric=row["metric"],
                independent_all=row["independent_all"],
                independent_unique_subset=row["independent_unique_subset"],
                shared_unique_subset=row["shared_unique_subset"],
                envelope_worst_all=row["envelope_worst_all"],
                envelope_best_all=row["envelope_best_all"],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def report(
    exp1: list[dict],
    exp3: list[dict],
    exp4: list[dict],
    subset: list[dict],
    paired: list[dict],
    flips: list[dict],
) -> None:
    print("\n=== Exp 1: best-reference consistency (tie-aware) ===")
    print(f"  {'variant':14s} {'n':>5s} {'coherent':>9s} {'95% CI':>16s} {'rel x place':>12s} {'ambiguous':>10s}")
    for row in exp1:
        if row["variant"] not in SOURCES:
            continue
        ci = f"[{row['coherent_ci_low']:.3f},{row['coherent_ci_high']:.3f}]"
        print(
            f"  {row['variant']:14s} {row['n']:5d} {row['coherent_rate']:9.3f} {ci:>16s}"
            f" {row['rel_vs_place_incompatible_rate']:12.3f} {row['ambiguous_joint_selection_rate']:10.3f}"
        )

    print("\n=== Exp 3: regret on the shared reference (unique k_joint) ===")
    print(f"  {'variant':14s} {'metric':11s} {'n':>5s} {'zero':>7s} {'median':>8s} {'mean':>8s} {'p90':>8s} {'p95':>8s}")
    for row in exp3:
        if row["variant"] not in SOURCES:
            continue
        print(
            f"  {row['variant']:14s} {row['metric']:11s} {row['n_unique_joint']:5d}"
            f" {row['fraction_zero_within_atol']:7.3f} {row['median']:8.3f} {row['mean']:8.3f}"
            f" {row['p90']:8.3f} {row['p95']:8.3f}"
        )

    print("\n=== Exp 4: independent vs shared Table 1 (medians) ===")
    print("  envelope columns bound each metric separately; they are not one realizable table")
    print(
        f"  {'variant':14s} {'metric':11s} {'ind(all)':>9s} {'ind(uni)':>9s} {'shared':>9s}"
        f" {'delta CI':>18s} {'worst':>8s} {'best':>8s} {'n_uni/n':>10s}"
    )
    for row in exp4:
        if row["variant"] not in SOURCES:
            continue
        ci = f"[{row['shared_minus_independent_ci_low']:+.3f},{row['shared_minus_independent_ci_high']:+.3f}]"
        counts = f"{row['n_unique_joint']}/{row['n_all']}"
        print(
            f"  {row['variant']:14s} {row['metric']:11s} {row['independent_all']:9.3f}"
            f" {row['independent_unique_subset']:9.3f} {row['shared_unique_subset']:9.3f}"
            f" {ci:>18s} {row['envelope_worst_all']:8.3f} {row['envelope_best_all']:8.3f} {counts:>10s}"
        )

    print("\n=== unique-k_joint subset selection (both sides independently aggregated) ===")
    print(f"  {'variant':14s} {'metric':11s} {'unique':>8s} {'ambiguous':>10s} {'delta':>8s}")
    for row in subset:
        if row["variant"] not in SOURCES:
            continue
        print(
            f"  {row['variant']:14s} {row['metric']:11s}"
            f" {row['independent_median_unique_subset']:8.3f}"
            f" {row['independent_median_ambiguous_subset']:10.3f}"
            f" {row['unique_minus_ambiguous']:+8.3f}"
        )

    print("\n=== paired Wilcoxon (z > 0 = first listed is better) ===")
    for row in paired:
        if np.isnan(row["z"]):
            continue
        print(
            f"  {row['comparison']:22s} {row['protocol']:11s} {row['metric']:11s} n={row['n']:4d}"
            f" first={row['n_first_better']:4d} second={row['n_second_better']:4d}"
            f" z={row['z']:+7.2f} p={row['p_value']:.1e}"
        )

    print("\n=== secondary: per-sketch base vs solver_rl verdict flips ===")
    for row in flips:
        print(
            f"  {row['metric']:11s} n={row['n_common_rids']:4d} flips={row['n_flips']:4d}"
            f" ({row['flip_rate']:.3f})  independent base/solver/tie="
            f"{row['n_independent_base_better']}/{row['n_independent_solver_better']}/{row['n_independent_tie']}"
            f"  shared={row['n_shared_base_better']}/{row['n_shared_solver_better']}/{row['n_shared_tie']}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_MOTION_EVAL = REPO_ROOT / "fusion360_annotation_" / "eda_code" / "out" / "motion_eval.csv"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-eval", default=str(DEFAULT_MOTION_EVAL))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "rebuttal" / "shared_reference"))
    ap.add_argument("--atol", type=float, default=PRIMARY_ATOL)
    ap.add_argument("--sweep", default=",".join(repr(a) for a in TOLERANCE_SWEEP))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--allow-table1-mismatch",
        action="store_true",
        help="continue even if independent(all) does not reproduce tab:controls",
    )
    args = ap.parse_args()

    source = Path(args.motion_eval).resolve()
    out_dir = Path(args.out)
    sweep = [float(x) for x in args.sweep.split(",") if x.strip()]

    samples, counts, place_column = load_samples(source)
    if not samples:
        raise SystemExit(f"no usable samples in {source}")

    config = dict(
        motion_eval=str(source),
        motion_eval_sha256=file_digest(source),
        motion_eval_mtime_ns=source.stat().st_mtime_ns,
        s_place_source_column=place_column,
        atol=args.atol,
        tolerance_sweep=sweep,
        metrics=list(METRICS),
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    prepare_output_dir(out_dir, config, force=args.force)
    write_environment(out_dir, sys.argv, [Path(__file__)])

    records = analyse(samples, args.atol)
    exp1 = exp1_table(records, args.atol)
    exp3 = exp3_table(records, args.atol)
    spread = reference_spread_table(records, args.atol)
    exp4 = exp4_table(records, args.atol)
    subset = subset_diagnostic(records, args.atol)
    paired = paired_tests(records)
    flips = model_flips(records)

    problems = check_table1_reproduction(exp4)
    if problems:
        message = "independent(all) does not reproduce tab:controls:\n  " + "\n  ".join(problems)
        if not args.allow_table1_mismatch:
            raise SystemExit(message)
        print(f"[warn] {message}")

    write_csv(out_dir / "per_sample.csv", per_sample_rows(records, args.atol), PER_SAMPLE_COLUMNS)
    write_csv(out_dir / "per_reference_scores.csv", per_reference_rows(records, args.atol), PER_REFERENCE_COLUMNS)
    write_csv(out_dir / "exp1_best_reference_consistency.csv", exp1, EXP1_COLUMNS)
    write_csv(out_dir / "exp3_regret_stats.csv", exp3, EXP3_COLUMNS)
    write_csv(out_dir / "exp3b_reference_spread.csv", spread, SPREAD_COLUMNS)
    write_csv(out_dir / "exp4_table_independent_vs_shared.csv", exp4, EXP4_COLUMNS)
    write_csv(out_dir / "exp4_subset_diagnostic.csv", subset, SUBSET_COLUMNS)
    write_csv(out_dir / "exp4_paired_tests.csv", paired, PAIRED_COLUMNS)
    write_csv(out_dir / "secondary_model_flips.csv", flips, FLIP_COLUMNS)

    sweep_out = []
    for atol in sweep:
        sweep_out.extend(sweep_rows(samples, atol))
    write_csv(out_dir / "tolerance_sweep.csv", sweep_out, SWEEP_COLUMNS)

    counts.update(
        atol=args.atol,
        table1_reproduced=not problems,
        unique_joint_by_variant={
            v: sum(1 for r in group if r["k_joint"]) for v, group in by_variant(records).items()
        },
        ambiguous_joint_by_variant={
            v: sum(1 for r in group if r["ambiguous"]) for v, group in by_variant(records).items()
        },
    )
    (out_dir / "sample_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    finalize_run_config(out_dir, n_samples=len(samples), table1_reproduced=not problems)

    report(exp1, exp3, exp4, subset, paired, flips)
    print(f"\nwrote results -> {out_dir}")


if __name__ == "__main__":
    main()
