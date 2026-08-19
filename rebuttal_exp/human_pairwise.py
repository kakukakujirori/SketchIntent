"""How much human-human variation does the best-reference aggregation hide?

Appendix Fig. 8 reports the full held-out-human distributions, but under the Sec. 5
multi-reference aggregation: ``min`` over the two references for the three angles and ``max``
for ``S_place``. It is therefore a best-of-2 statistic and does not characterise human-human
disagreement directly. This script reports every within-sketch human pair on its own, with no
min/max anywhere.

Nothing is recomputed. ``06_run_motion_eval.py`` enumerates ``combinations(names, 2)`` with the
third annotation held out and scores the held-out annotation against *each* reference
separately, one row per (rid, variant, reference). With exactly three annotations per sketch the
``variant=heldout`` rows are therefore already all six ordered human pairs per sketch -- i.e. all
three unordered pairs with both ``theta_leak`` directions. Re-reading that file inherits the
normalization, observable space, Jacobians, SVD thresholds, construction-geometry handling,
FreeCAD gating and tolerances of Table 1 with the file itself, which the Fig. 8 reproduction gate
below asserts mechanically.

Three protocol points that shape every number.

Symmetry. theta_full, theta_rel and S_place are symmetric in (Hi, Hj) by construction -- principal
angles and a Hungarian matching of |<u_i, u_j>| do not care which side is called Pred. Each
unordered pair therefore contributes exactly one value, taken from the canonical direction
(annotator initials in sorted order), and the observed asymmetry is carried per row as a
diagnostic. It must stay under RESOLVED_METRIC_FLOOR or the run aborts.

Direction. theta_leak is *not* symmetric: it evaluates GT's relational constraints at Pred's
state, so ``leak(Hi; Hj) != leak(Hj; Hi)`` -- by up to 0.845 on this data. Both directions are
kept verbatim and both enter the summary, so its N is twice that of the symmetric metrics. It is
never symmetrized into a new metric.

N. The symmetric metrics have one observation per unordered pair (858) and theta_leak has one per
ordered pair (1716). Fig. 8 also has 858 observations, but they are the *minimum* of two of these
pairs, not the pairs themselves; that is the entire difference between the two protocols.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rebuttal_exp.common import (  # noqa: E402
    RESOLVED_METRIC_FLOOR,
    finalize_run_config,
    prepare_output_dir,
    write_csv,
    write_environment,
)
from rebuttal_exp.shared_reference import (  # noqa: E402
    ANGLES,
    METRICS,
    PLACE,
    DEFAULT_MOTION_EVAL,
    file_digest,
    load_samples,
    quantile,
)

SYMMETRIC = ("theta_full", "theta_rel", PLACE)
"""Principal angles and the Hungarian placement matching do not depend on argument order."""
DIRECTIONAL = "theta_leak"

HELDOUT = "heldout"
DEFAULT_CROSS_CHECK = REPO_ROOT / "ablation" / "out" / "motion_eval.csv"

FIG8_REFERENCE = {
    "theta_full": (0.167, 2.4e-8),
    "theta_rel": (0.178, 2.6e-8),
    "theta_leak": (0.090, 7.5e-15),
    PLACE: (0.823, 0.831),
}
FIG8_N = 858
FIG8_MEAN_ATOL = 2e-3
FIG8_MEDIAN_RTOL = 0.25
"""paper/sec/8_suppl.tex fig:appendix_heldout_distribution. The means are compared absolutely;
the medians sit on the sqrt(eps) noise floor of the padded angle, where two independent runs of
the same pipeline already differ by a few percent, so they are compared relatively."""


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def query_annotator(rid: str) -> str:
    """The held-out annotation of a leave-one-out rid ``{batch}:{base}:{refs}__q={heldout}``."""
    head, _, query = rid.rpartition("__q=")
    if not head or not query:
        raise ValueError(f"rid does not carry a held-out annotator: {rid!r}")
    return query


@dataclass(frozen=True)
class HumanPair:
    """One unordered pair of annotations of the same sketch, scored in both directions."""

    batch: str
    filename: str
    annotator_i: str
    annotator_j: str
    symmetric: dict[str, float]  # metric -> canonical-direction value
    asymmetry: dict[str, float]  # metric -> |v(i,j) - v(j,i)|
    leak_i_to_j: float
    leak_j_to_i: float

    @property
    def sketch_id(self) -> str:
        return self.filename[:-6] if self.filename.lower().endswith(".fcstd") else self.filename

    @property
    def cluster(self) -> tuple[str, str]:
        return (self.batch, self.filename)


def directed_scores(samples) -> tuple[dict, dict]:
    """Map (batch, filename) -> (pred, gt) -> {metric: value} over the held-out rows only."""
    directed: dict[tuple[str, str], dict[tuple[str, str], dict[str, float]]] = {}
    counts = dict(heldout_samples=0, directed_observations=0, self_comparisons=0)
    for sample in samples:
        if sample.variant != HELDOUT:
            continue
        counts["heldout_samples"] += 1
        pred = query_annotator(sample.rid)
        for ref in sample.refs:
            if pred == ref:
                counts["self_comparisons"] += 1
                continue
            key = (sample.batch, sample.filename)
            directed.setdefault(key, {})[(pred, ref)] = {m: sample.values[m][ref] for m in METRICS}
            counts["directed_observations"] += 1
    return directed, counts


def build_pairs(directed: dict, *, floor: float = RESOLVED_METRIC_FLOOR) -> tuple[list[HumanPair], dict]:
    """Collapse the directed observations into unordered pairs, keeping both leak directions."""
    pairs: list[HumanPair] = []
    counts = dict(pairs=0, pairs_missing_a_direction=0, max_asymmetry={m: 0.0 for m in SYMMETRIC})
    for (batch, filename), by_direction in sorted(directed.items()):
        seen: set[frozenset[str]] = set()
        for (pred, ref) in sorted(by_direction):
            unordered = frozenset((pred, ref))
            if unordered in seen:
                continue
            seen.add(unordered)
            i, j = sorted(unordered)
            forward = by_direction.get((i, j))
            backward = by_direction.get((j, i))
            if forward is None or backward is None:
                counts["pairs_missing_a_direction"] += 1
                continue
            symmetric = {m: forward[m] for m in SYMMETRIC}
            asymmetry = {m: abs(forward[m] - backward[m]) for m in SYMMETRIC}
            for m, delta in asymmetry.items():
                counts["max_asymmetry"][m] = max(counts["max_asymmetry"][m], delta)
            pairs.append(
                HumanPair(
                    batch=batch,
                    filename=filename,
                    annotator_i=i,
                    annotator_j=j,
                    symmetric=symmetric,
                    asymmetry=asymmetry,
                    leak_i_to_j=forward[DIRECTIONAL],
                    leak_j_to_i=backward[DIRECTIONAL],
                )
            )
            counts["pairs"] += 1
    violations = {m: v for m, v in counts["max_asymmetry"].items() if v > floor}
    if violations:
        raise SystemExit(
            "a metric declared symmetric is not symmetric within the numerical floor "
            f"({floor:g}): {violations}. Either the loader mixed up the argument order or the "
            "metric definition changed; do not paper over this with an average."
        )
    return pairs, counts


def pair_values(pairs: list[HumanPair]) -> dict[str, list[float]]:
    """All-pairs observations: one per unordered pair, both directions for theta_leak."""
    values = {m: [p.symmetric[m] for p in pairs] for m in SYMMETRIC}
    values[DIRECTIONAL] = [v for p in pairs for v in (p.leak_i_to_j, p.leak_j_to_i)]
    return {m: values[m] for m in METRICS}


def best_reference_values(samples) -> dict[str, list[float]]:
    """Fig. 8's protocol on the same rows: min over references for angles, max for S_place."""
    values: dict[str, list[float]] = {m: [] for m in METRICS}
    for sample in samples:
        if sample.variant != HELDOUT:
            continue
        for m in METRICS:
            scores = list(sample.values[m].values())
            values[m].append(min(scores) if m in ANGLES else max(scores))
    return values


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "protocol", "metric", "n", "mean", "median", "q1", "q3", "iqr", "p90",
    "min", "max", "near_zero_eps", "frac_below_eps", "frac_below_1e_3", "near_identical_frac",
]


def summarize(values: list[float], metric: str, protocol: str, eps: float) -> dict:
    n = len(values)
    if n == 0:
        raise SystemExit(f"no observations for {protocol}/{metric}")
    q1, q3 = quantile(values, 25), quantile(values, 75)
    identical = (
        sum(1 for v in values if v > 1 - eps) if metric == PLACE else sum(1 for v in values if v < eps)
    )
    return dict(
        protocol=protocol,
        metric=metric,
        n=n,
        mean=st.fmean(values),
        median=st.median(values),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        p90=quantile(values, 90),
        min=min(values),
        max=max(values),
        near_zero_eps=eps,
        frac_below_eps=sum(1 for v in values if v < eps) / n,
        frac_below_1e_3=sum(1 for v in values if v < 1e-3) / n,
        near_identical_frac=identical / n,
    )


def summary_table(all_pairs: dict, best_ref: dict, eps: float) -> list[dict]:
    rows = []
    for protocol, values in (("all_pairs", all_pairs), ("best_reference", best_ref)):
        for metric in METRICS:
            rows.append(summarize(values[metric], metric, protocol, eps))
    return rows


def check_fig8_reproduction(rows: list[dict]) -> list[str]:
    """Re-applying min/max to the same rows must land on the published Fig. 8 statistics."""
    problems = []
    by_metric = {r["metric"]: r for r in rows if r["protocol"] == "best_reference"}
    for metric, (mean, median) in FIG8_REFERENCE.items():
        row = by_metric.get(metric)
        if row is None:
            problems.append(f"{metric}: absent from the best_reference summary")
            continue
        if row["n"] != FIG8_N:
            problems.append(f"{metric}: n={row['n']}, Fig. 8 has {FIG8_N}")
        if abs(row["mean"] - mean) > FIG8_MEAN_ATOL:
            problems.append(f"{metric}: mean {row['mean']:.5g}, Fig. 8 reports {mean:g}")
        if median and abs(row["median"] - median) > FIG8_MEDIAN_RTOL * median:
            problems.append(f"{metric}: median {row['median']:.4g}, Fig. 8 reports {median:g}")
    return problems


# ---------------------------------------------------------------------------
# Per-pair and per-sketch tables
# ---------------------------------------------------------------------------

PAIR_COLUMNS = [
    "sketch_id", "batch", "annotator_i", "annotator_j", "annotation_i_id", "annotation_j_id",
    "theta_full_motion", "theta_rel_motion", "s_place",
    "theta_dim_leak_i_to_j", "theta_dim_leak_j_to_i",
    "theta_full_asym_abs", "theta_rel_asym_abs", "s_place_asym_abs",
]


def _f(value: float) -> str:
    """repr() round-trips a float exactly; str(1e-15) must not become '0'."""
    return repr(float(value))


def pair_rows(pairs: list[HumanPair], annotations: dict) -> list[dict]:
    rows = []
    for pair in pairs:
        files = annotations.get((pair.batch, pair.filename), {})
        rows.append(
            dict(
                sketch_id=pair.sketch_id,
                batch=pair.batch,
                annotator_i=pair.annotator_i,
                annotator_j=pair.annotator_j,
                annotation_i_id=files.get(pair.annotator_i, ""),
                annotation_j_id=files.get(pair.annotator_j, ""),
                theta_full_motion=_f(pair.symmetric["theta_full"]),
                theta_rel_motion=_f(pair.symmetric["theta_rel"]),
                s_place=_f(pair.symmetric[PLACE]),
                theta_dim_leak_i_to_j=_f(pair.leak_i_to_j),
                theta_dim_leak_j_to_i=_f(pair.leak_j_to_i),
                theta_full_asym_abs=_f(pair.asymmetry["theta_full"]),
                theta_rel_asym_abs=_f(pair.asymmetry["theta_rel"]),
                s_place_asym_abs=_f(pair.asymmetry[PLACE]),
            )
        )
    return rows


PER_SKETCH_COLUMNS = [
    "sketch_id", "batch", "n_pairs", "annotators",
    "max_pairwise_theta_full", "mean_pairwise_theta_full",
    "max_pairwise_theta_rel", "mean_pairwise_theta_rel",
    "min_pairwise_S_place", "mean_pairwise_S_place",
    "max_pairwise_theta_leak", "mean_pairwise_theta_leak",
    "n_pairs_below_eps", "agreement_label",
]


def per_sketch_rows(pairs: list[HumanPair], eps: float) -> list[dict]:
    by_sketch: dict[tuple[str, str], list[HumanPair]] = {}
    for pair in pairs:
        by_sketch.setdefault(pair.cluster, []).append(pair)

    rows = []
    for (batch, filename), group in sorted(by_sketch.items()):
        full = [p.symmetric["theta_full"] for p in group]
        rel = [p.symmetric["theta_rel"] for p in group]
        place = [p.symmetric[PLACE] for p in group]
        leak = [v for p in group for v in (p.leak_i_to_j, p.leak_j_to_i)]
        below = sum(1 for v in full if v < eps)
        if below == len(full):
            label = "consensus"
        elif below == 0:
            label = "divergent"
        else:
            label = "mixed"
        rows.append(
            dict(
                sketch_id=group[0].sketch_id,
                batch=batch,
                n_pairs=len(group),
                annotators="".join(sorted({a for p in group for a in (p.annotator_i, p.annotator_j)})),
                max_pairwise_theta_full=_f(max(full)),
                mean_pairwise_theta_full=_f(st.fmean(full)),
                max_pairwise_theta_rel=_f(max(rel)),
                mean_pairwise_theta_rel=_f(st.fmean(rel)),
                min_pairwise_S_place=_f(min(place)),
                mean_pairwise_S_place=_f(st.fmean(place)),
                max_pairwise_theta_leak=_f(max(leak)),
                mean_pairwise_theta_leak=_f(st.fmean(leak)),
                n_pairs_below_eps=below,
                agreement_label=label,
            )
        )
    return rows


def agreement_label_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"consensus": 0, "mixed": 0, "divergent": 0}
    for row in rows:
        counts[row["agreement_label"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Cross-source stability
# ---------------------------------------------------------------------------

CROSS_COLUMNS = ["metric", "n_primary", "n_secondary", "mean_primary", "mean_secondary",
                 "mean_delta", "median_primary", "median_secondary", "median_delta",
                 "p90_primary", "p90_secondary", "p90_delta"]


def cross_source_rows(primary: dict, secondary: dict, eps: float) -> list[dict]:
    rows = []
    for metric in METRICS:
        a = summarize(primary[metric], metric, "all_pairs", eps)
        b = summarize(secondary[metric], metric, "all_pairs", eps)
        rows.append(
            dict(
                metric=metric,
                n_primary=a["n"], n_secondary=b["n"],
                mean_primary=a["mean"], mean_secondary=b["mean"], mean_delta=b["mean"] - a["mean"],
                median_primary=a["median"], median_secondary=b["median"],
                median_delta=b["median"] - a["median"],
                p90_primary=a["p90"], p90_secondary=b["p90"], p90_delta=b["p90"] - a["p90"],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PLOT_LABELS = {
    "theta_full": r"$\theta_{\mathrm{motion}}^{\mathrm{full}}$",
    "theta_rel": r"$\theta_{\mathrm{motion}}^{\mathrm{rel}}$",
    "theta_leak": r"$\theta_{\mathrm{leak}}^{\mathrm{dim}}$",
    PLACE: r"$S_{\mathrm{place}}$",
}
DISTANCE_COLOR = "#4C78A8"
PLACE_COLOR = "#54A24B"

NOTE_HEADROOM = 1.32
"""Whitespace above the tallest bin so the per-panel mean/median table never sits on data.
Nothing is hidden: every bin is still drawn at its true height."""


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.5,
        }
    )
    return plt


def _fmt(value: float) -> str:
    return f"{value:.1e}" if abs(value) < 0.01 else f"{value:.2f}"


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")


def plot_distribution(values: dict, stats: dict, output: Path) -> None:
    """Appendix Fig. 8's panel, redrawn on the direct all-pair observations."""
    plt = _style()
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 3.25), constrained_layout=True)
    bins = [i / 20 for i in range(21)]
    for ax, metric in zip(axs.ravel(), METRICS):
        color = PLACE_COLOR if metric == PLACE else DISTANCE_COLOR
        ax.hist(values[metric], bins=bins, color=color, edgecolor="white", linewidth=0.45)
        ax.axvline(stats[metric]["median"], color="black", lw=1.15, ls="--")
        ax.axvline(stats[metric]["mean"], color="#D95F02", lw=1.15, ls=":")
        ax.set_xlim(0, 1)
        ax.set_title(PLOT_LABELS[metric])
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.spines[["top", "right"]].set_visible(False)
        note = f"mean {_fmt(stats[metric]['mean'])}\nmedian {_fmt(stats[metric]['median'])}"
        ax.text(
            0.18 if metric == PLACE else 0.40,
            0.94,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
    _save(fig, output)
    plt.close(fig)


def plot_overlay(all_pairs: dict, best_ref: dict, stats: dict, output: Path) -> None:
    """The two protocols on one axis, as a drop-in replacement for Appendix Fig. 8.

    Fractions, not counts: theta_leak has one observation per *ordered* pair and so twice the N
    of the symmetric metrics. The best-reference distribution is drawn as an outline so the
    filled all-pair bars in front of it stay readable; overlapping fills read as stacked.

    Mean and median are printed rather than drawn as vertical lines. Under the best-reference
    protocol the motion medians sit at ~1e-8, i.e. on the y-axis, where a vertical line is
    invisible -- and it is exactly that number, next to the all-pair median, that carries the
    result.
    """
    plt = _style()
    fig, axs = plt.subplots(2, 2, figsize=(4.9, 3.5), constrained_layout=True)
    bins = [i / 20 for i in range(21)]
    for ax, metric in zip(axs.ravel(), METRICS):
        color = PLACE_COLOR if metric == PLACE else DISTANCE_COLOR
        tallest = 0.0
        for values, kwargs in (
            (all_pairs[metric],
             dict(color=color, edgecolor="white", linewidth=0.35, label="all pairs")),
            (best_ref[metric],
             dict(histtype="step", color="#4D4D4D", linewidth=1.0, label="best-reference")),
        ):
            counts, _, _ = ax.hist(
                values, bins=bins, weights=[1 / len(values)] * len(values), **kwargs
            )
            tallest = max(tallest, float(max(counts)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, tallest * NOTE_HEADROOM)
        ax.set_title(PLOT_LABELS[metric])
        ax.set_xlabel("score")
        ax.set_ylabel("fraction")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.02 if metric == PLACE else 0.20,
            0.97,
            _panel_note(stats, metric),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            family="monospace",
            linespacing=1.35,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        )
    handles, labels = axs.ravel()[0].get_legend_handles_labels()
    order = [labels.index("best-reference"), labels.index("all pairs")]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    _save(fig, output)
    plt.close(fig)


def _panel_note(stats: dict, metric: str) -> str:
    """A three-line mini-table: mean and median under each protocol."""
    rows = [("", "mean", "median")]
    for protocol, label in (("best_reference", "best-ref"), ("all_pairs", "all pairs")):
        row = stats[protocol][metric]
        rows.append((label, _fmt(row["mean"]), _fmt(row["median"])))
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    return "\n".join(
        f"{r[0]:<{widths[0]}}  {r[1]:>{widths[1]}}  {r[2]:>{widths[2]}}" for r in rows
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

MARKDOWN_FIELDS = ["protocol", "metric", "n", "mean", "median", "q1", "q3", "iqr", "p90"]


def markdown_table(rows: list[dict]) -> str:
    def cell(row, key):
        value = row[key]
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    lines = ["| " + " | ".join(MARKDOWN_FIELDS) + " |",
             "|" + "|".join(["---"] * len(MARKDOWN_FIELDS)) + "|"]
    for metric in METRICS:
        for row in rows:
            if row["metric"] == metric:
                lines.append("| " + " | ".join(cell(row, k) for k in MARKDOWN_FIELDS) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-eval", default=str(DEFAULT_MOTION_EVAL))
    ap.add_argument(
        "--cross-check-csv",
        default=str(DEFAULT_CROSS_CHECK),
        help="second independent run of the same pipeline; pass '' to skip",
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "rebuttal" / "human_pairwise"))
    ap.add_argument("--near-zero-eps", type=float, default=RESOLVED_METRIC_FLOOR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--allow-fig8-mismatch",
        action="store_true",
        help="continue even if the min/max re-aggregation does not reproduce Fig. 8",
    )
    args = ap.parse_args()

    source = Path(args.motion_eval).resolve()
    out_dir = Path(args.out)
    eps = args.near_zero_eps

    samples, load_counts, place_column = load_samples(source)
    directed, directed_counts = directed_scores(samples)
    if not directed:
        raise SystemExit(f"no variant={HELDOUT} rows in {source}")
    pairs, pair_counts = build_pairs(directed, floor=RESOLVED_METRIC_FLOOR)

    annotations = load_annotation_ids()

    all_pairs = pair_values(pairs)
    best_ref = best_reference_values(samples)
    summary = summary_table(all_pairs, best_ref, eps)

    problems = check_fig8_reproduction(summary)
    if problems:
        message = "the min/max re-aggregation does not reproduce Fig. 8:\n  " + "\n  ".join(problems)
        if not args.allow_fig8_mismatch:
            raise SystemExit(message)
        print(f"[warn] {message}")

    config = dict(
        motion_eval=str(source),
        motion_eval_sha256=file_digest(source),
        motion_eval_mtime_ns=source.stat().st_mtime_ns,
        cross_check_csv=args.cross_check_csv or None,
        s_place_source_column=place_column,
        near_zero_eps=eps,
        symmetry_floor=RESOLVED_METRIC_FLOOR,
        metrics=list(METRICS),
        symmetric_metrics=list(SYMMETRIC),
        directional_metric=DIRECTIONAL,
    )
    prepare_output_dir(out_dir, config, force=args.force)
    write_environment(out_dir, sys.argv, [Path(__file__)])

    rows = pair_rows(pairs, annotations)
    sketches = per_sketch_rows(pairs, eps)
    write_csv(out_dir / "human_pairwise_metrics.csv", rows, PAIR_COLUMNS)
    write_csv(out_dir / "human_pairwise_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(out_dir / "human_pairwise_per_sketch.csv", sketches, PER_SKETCH_COLUMNS)

    cross_rows = []
    cross_path = Path(args.cross_check_csv).resolve() if args.cross_check_csv else None
    if cross_path is not None and cross_path.exists() and cross_path != source:
        other_samples, _, _ = load_samples(cross_path)
        other_directed, _ = directed_scores(other_samples)
        other_pairs, _ = build_pairs(other_directed, floor=RESOLVED_METRIC_FLOOR)
        cross_rows = cross_source_rows(all_pairs, pair_values(other_pairs), eps)
        write_csv(out_dir / "cross_source_stability.csv", cross_rows, CROSS_COLUMNS)
    elif cross_path is not None and not cross_path.exists():
        print(f"[warn] cross-check CSV not found, skipping: {cross_path}")

    checks = sanity_checks(
        pairs=pairs,
        directed=directed,
        directed_counts=directed_counts,
        pair_counts=pair_counts,
        all_pairs=all_pairs,
        summary=summary,
        sketches=sketches,
        fig8_ok=not problems,
        eps=eps,
    )
    (out_dir / "sanity_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True))

    if not args.no_figures:
        stats = {
            protocol: {r["metric"]: r for r in summary if r["protocol"] == protocol}
            for protocol in ("all_pairs", "best_reference")
        }
        figures = out_dir / "figures"
        plot_distribution(all_pairs, stats["all_pairs"], figures / "human_pairwise_distribution.pdf")
        plot_overlay(all_pairs, best_ref, stats, figures / "human_pairwise_vs_bestref.pdf")

    counts = dict(
        load_counts,
        n_valid_sketches=checks["n_valid_sketches"]["value"],
        n_valid_annotations=checks["n_valid_annotations"]["value"],
        n_valid_unordered_pairs=checks["n_valid_unordered_pairs"]["value"],
        n_directional_leak_evaluations=checks["n_directional_leak_evaluations"]["value"],
        agreement_labels=agreement_label_counts(sketches),
    )
    (out_dir / "sample_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    finalize_run_config(out_dir, n_pairs=len(pairs), fig8_reproduced=not problems)

    report(summary, checks, sketches, cross_rows)
    print(f"\nwrote results -> {out_dir}")


def load_annotation_ids() -> dict:
    """(batch, base_fcstd) -> {annotator: annotation filename}; empty if the dataset is absent."""
    try:
        from src.eval.manifest import discover_manifest

        manifest, _ = discover_manifest()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] annotation ids unavailable ({str(exc)[:80]}); id columns will be empty")
        return {}
    return {
        (batch, base): dict(files)
        for batch, bases in manifest.items()
        for base, files in bases.items()
    }


def sanity_checks(*, pairs, directed, directed_counts, pair_counts, all_pairs, summary,
                  sketches, fig8_ok, eps) -> dict:
    """Every item of the spec's checklist, with its measured value and a pass/fail."""
    annotators = {(p.batch, p.filename, a) for p in pairs for a in (p.annotator_i, p.annotator_j)}
    unordered = {(p.batch, p.filename, p.annotator_i, p.annotator_j) for p in pairs}
    positives = [v for m in METRICS for v in all_pairs[m] if v > 0]
    all_pairs_rows = {r["metric"]: r for r in summary if r["protocol"] == "all_pairs"}

    def check(value, ok, note):
        return dict(value=value, ok=bool(ok), note=note)

    return dict(
        n_valid_sketches=check(len({p.cluster for p in pairs}), True, "sketches with usable pairs"),
        n_valid_annotations=check(
            len(annotators), True, "distinct (sketch, annotator) annotations entering a pair"
        ),
        n_valid_unordered_pairs=check(
            len(pairs), len(unordered) == len(pairs), "each unordered pair appears exactly once"
        ),
        n_directional_leak_evaluations=check(
            len(all_pairs["theta_leak"]),
            len(all_pairs["theta_leak"]) == 2 * len(pairs),
            "both directions kept for every pair",
        ),
        no_self_comparisons=check(
            directed_counts["self_comparisons"],
            directed_counts["self_comparisons"] == 0
            and all(p.annotator_i != p.annotator_j for p in pairs),
            "count of pred == gt observations, must be 0",
        ),
        no_multireference_aggregation=check(
            {m: all_pairs_rows[m]["n"] for m in METRICS},
            all(all_pairs_rows[m]["n"] == len(pairs) for m in SYMMETRIC),
            "one observation per unordered pair; no min/max is applied to the all_pairs protocol",
        ),
        symmetry_max_abs=check(
            pair_counts["max_asymmetry"],
            all(v <= RESOLVED_METRIC_FLOOR for v in pair_counts["max_asymmetry"].values()),
            f"max |v(i,j) - v(j,i)| per symmetric metric, floor {RESOLVED_METRIC_FLOOR:g}",
        ),
        directional_metric_kept=check(
            max((abs(p.leak_i_to_j - p.leak_j_to_i) for p in pairs), default=0.0),
            True,
            "max theta_leak direction gap; a nonzero value is why it is not symmetrized",
        ),
        pipeline_inherited=check(
            fig8_ok,
            fig8_ok,
            "re-applying min/max to the same rows reproduces the published Fig. 8 statistics, so "
            "normalization, measurement space, SVD threshold and filtering are the Table 1 ones",
        ),
        no_rounding_to_zero=check(
            min(positives) if positives else None,
            bool(positives) and min(positives) < 1e-12,
            "smallest strictly positive observation; near-zero values survive as floats",
        ),
        pairs_missing_a_direction=check(
            pair_counts["pairs_missing_a_direction"],
            pair_counts["pairs_missing_a_direction"] == 0,
            "unordered pairs dropped for want of one direction",
        ),
        agreement_labels=check(agreement_label_counts(sketches), True, "per-sketch agreement split"),
    )


def report(summary: list[dict], checks: dict, sketches: list[dict], cross_rows: list[dict]) -> None:
    print("\n## Counts\n")
    for key in ("n_valid_sketches", "n_valid_annotations", "n_valid_unordered_pairs",
                "n_directional_leak_evaluations"):
        print(f"  {key:32s} {checks[key]['value']}")
    print(
        "\n  filtering rule (inherited): a sketch enters only with exactly three parseable "
        "annotations,\n  all three passing the canonical_solid base policy at geometry_tol=5e-1, "
        "and both sides of a\n  pair sharing the observable dimension m_phi. Humans are not "
        "solver-gated; models are."
    )

    print("\n## Summary\n")
    print(markdown_table(summary))

    print("\n## Sanity checks\n")
    for name, entry in sorted(checks.items()):
        flag = "ok " if entry["ok"] else "FAIL"
        print(f"  [{flag}] {name:32s} {entry['value']}")

    labels = agreement_label_counts(sketches)
    total = sum(labels.values()) or 1
    print("\n## Per-sketch agreement\n")
    for label in ("consensus", "mixed", "divergent"):
        print(f"  {label:12s} {labels[label]:4d}  ({100 * labels[label] / total:.1f}%)")

    if cross_rows:
        print("\n## Cross-source stability (independent rerun of the same pipeline)\n")
        for row in cross_rows:
            print(
                f"  {row['metric']:11s} mean {row['mean_primary']:.4f} -> "
                f"{row['mean_secondary']:.4f} (delta {row['mean_delta']:+.4f}), "
                f"p90 delta {row['p90_delta']:+.4f}"
            )


if __name__ == "__main__":
    main()
