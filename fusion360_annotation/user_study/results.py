#!/usr/bin/env python3
"""Summarize the Solver-RL vs. Ours user-study results.

The main table uses individual non-tie preferences over valid presented
sketches, matching the reference-geometry metrics that are computed only for
scorable outputs. Votes are counted over all raters. Preference correlations use
per-sketch vote margins and signed metric advantages. Use --include-invalid for
an end-to-end presented-set audit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DIMENSIONAL_CONSTRAINTS = {
    "Angle",
    "Diameter",
    "Distance",
    "DistanceX",
    "DistanceY",
    "Radius",
}

METRICS = (
    ("dim_ratio", "Dim.", True),
    ("motion_full_angle", "theta_full", True),
    ("motion_relational_angle", "theta_rel", True),
    ("leak_dimensional_angle", "theta_leak", True),
    ("placement_sim", "S_place", False),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLVER_METRICS = REPO_ROOT / "artifacts/fiverr_eval/iter1_dpc1_step70000.json"
DEFAULT_OURS_METRICS = REPO_ROOT / "artifacts/fiverr_eval/iter1_oursretry_step70000.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "fusion360_annotation/user_study/results"
DEFAULT_STUDY_ROOT = REPO_ROOT / "fusion360_annotation/user_study"


@dataclass(frozen=True)
class Vote:
    rater: str
    stem: str
    batch: str
    preferred: str


@dataclass
class Agreement:
    hit: int = 0
    total: int = 0
    missing: int = 0
    metric_ties: int = 0

    @property
    def pct(self) -> float:
        return float("nan") if self.total == 0 else 100.0 * self.hit / self.total


@dataclass
class PairWins:
    solver: int = 0
    ours: int = 0
    ties: int = 0
    missing: int = 0


@dataclass
class PreferenceCorrelation:
    pearson: float = float("nan")
    spearman: float = float("nan")
    total: int = 0
    missing: int = 0


@dataclass
class MetricMean:
    solver: float = float("nan")
    ours: float = float("nan")
    total: int = 0
    missing: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--solver-metrics", type=Path, default=DEFAULT_SOLVER_METRICS)
    parser.add_argument("--ours-metrics", type=Path, default=DEFAULT_OURS_METRICS)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include sketches whose exported model outputs were marked invalid.",
    )
    return parser.parse_args()


def model_short(name: str) -> str:
    if "m4.5_l1.5_leakdof" in name:
        return "Ours"
    if "dimpars_c1" in name:
        return "Solver-RL"
    return name


def load_votes(results_dir: Path) -> tuple[list[Vote], str, str]:
    votes: list[Vote] = []
    model1 = model2 = None
    paths = sorted(results_dir.glob("user_study_results_*.json"))
    if not paths:
        raise FileNotFoundError(f"No user_study_results_*.json files in {results_dir}")

    for path in paths:
        data = json.loads(path.read_text())
        meta = data["meta"]
        if model1 is None:
            model1 = meta["model1"]
            model2 = meta["model2"]
        elif (model1, model2) != (meta["model1"], meta["model2"]):
            raise ValueError(f"Model mismatch in {path}")

        rater = path.stem.removeprefix("user_study_results_")
        for item in data["votes"]:
            votes.append(
                Vote(
                    rater=rater,
                    stem=item["stem"],
                    batch=str(item["batch"]),
                    preferred=item["preferred"],
                )
            )

    assert model1 is not None and model2 is not None
    return votes, model1, model2


def stem_from_metric_id(metric_id: str) -> str:
    return metric_id.split("/", 1)[1].rsplit(".", 1)[0]


def load_per_sketch_metrics(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    return {
        stem_from_metric_id(row["_id"]): row
        for row in data["reference"]["per_sketch"]
    }


def dim_ratio_for_file(path: Path) -> float:
    text = path.read_text()
    constraints = re.findall(r"addConstraint\(Sketcher\.Constraint\('([^']+)'", text)
    if not constraints:
        return float("nan")
    n_dim = sum(name in DIMENSIONAL_CONSTRAINTS for name in constraints)
    return n_dim / len(constraints)


def invalid_models_for_stem(
    study_root: Path, model_names: tuple[str, str], stem: str, batch: str
) -> list[str]:
    invalid = []
    for model in model_names:
        path = study_root / model / batch / f"{stem}.py"
        header = "\n".join(path.read_text().splitlines()[:4])
        if "NOTE: Final constraints were marked invalid" in header:
            invalid.append(model)
    return invalid


def filter_valid_votes(
    votes: list[Vote],
    study_root: Path,
    model_names: tuple[str, str],
) -> tuple[list[Vote], dict[str, list[str]]]:
    stem_to_batch = {vote.stem: vote.batch for vote in votes}
    invalid_by_stem = {
        stem: invalid
        for stem, batch in stem_to_batch.items()
        if (invalid := invalid_models_for_stem(study_root, model_names, stem, batch))
    }
    valid_votes = [vote for vote in votes if vote.stem not in invalid_by_stem]
    return valid_votes, invalid_by_stem


def load_dim_ratios(
    study_root: Path, model_names: tuple[str, str], stem_to_batch: dict[str, str]
) -> dict[str, dict[str, float]]:
    ratios: dict[str, dict[str, float]] = {model: {} for model in model_names}
    for stem, batch in stem_to_batch.items():
        for model in model_names:
            path = study_root / model / batch / f"{stem}.py"
            ratios[model][stem] = dim_ratio_for_file(path)
    return ratios


def metric_values(
    metric: str,
    stem: str,
    solver: str,
    ours: str,
    solver_metrics: dict[str, dict[str, float]],
    ours_metrics: dict[str, dict[str, float]],
    dim_ratios: dict[str, dict[str, float]],
) -> tuple[float | None, float | None]:
    if metric == "dim_ratio":
        return dim_ratios[solver].get(stem), dim_ratios[ours].get(stem)
    if stem not in solver_metrics or stem not in ours_metrics:
        return None, None
    return solver_metrics[stem][metric], ours_metrics[stem][metric]


def compare_model(
    solver_value: float,
    ours_value: float,
    lower_is_better: bool,
    solver: str,
    ours: str,
) -> str | None:
    if math.isnan(solver_value) or math.isnan(ours_value):
        return None
    if abs(solver_value - ours_value) < 1e-12:
        return None
    if lower_is_better:
        return solver if solver_value < ours_value else ours
    return solver if solver_value > ours_value else ours


def agreement_for_decisions(
    decisions: list[tuple[str, str]],
    solver: str,
    ours: str,
    solver_metrics: dict[str, dict[str, float]],
    ours_metrics: dict[str, dict[str, float]],
    dim_ratios: dict[str, dict[str, float]],
) -> dict[str, Agreement]:
    out = {key: Agreement() for key, _, _ in METRICS}
    for stem, preferred in decisions:
        for metric, _label, lower_is_better in METRICS:
            solver_value, ours_value = metric_values(
                metric, stem, solver, ours, solver_metrics, ours_metrics, dim_ratios
            )
            if solver_value is None or ours_value is None:
                out[metric].missing += 1
                continue
            picked = compare_model(solver_value, ours_value, lower_is_better, solver, ours)
            if picked is None:
                out[metric].metric_ties += 1
                continue
            out[metric].total += 1
            out[metric].hit += int(picked == preferred)
    return out


def metric_pair_wins(
    stems: list[str],
    solver: str,
    ours: str,
    solver_metrics: dict[str, dict[str, float]],
    ours_metrics: dict[str, dict[str, float]],
    dim_ratios: dict[str, dict[str, float]],
) -> dict[str, PairWins]:
    out = {key: PairWins() for key, _, _ in METRICS}
    for stem in stems:
        for metric, _label, lower_is_better in METRICS:
            solver_value, ours_value = metric_values(
                metric, stem, solver, ours, solver_metrics, ours_metrics, dim_ratios
            )
            item = out[metric]
            if solver_value is None or ours_value is None:
                item.missing += 1
                continue
            picked = compare_model(solver_value, ours_value, lower_is_better, solver, ours)
            if picked is None:
                item.ties += 1
            elif picked == solver:
                item.solver += 1
            else:
                item.ours += 1
    return out


def metric_means(
    stems: list[str],
    solver: str,
    ours: str,
    solver_metrics: dict[str, dict[str, float]],
    ours_metrics: dict[str, dict[str, float]],
    dim_ratios: dict[str, dict[str, float]],
) -> dict[str, MetricMean]:
    out = {key: MetricMean() for key, _, _ in METRICS}
    for metric, _label, _lower_is_better in METRICS:
        solver_values = []
        ours_values = []
        missing = 0
        for stem in stems:
            solver_value, ours_value = metric_values(
                metric, stem, solver, ours, solver_metrics, ours_metrics, dim_ratios
            )
            if solver_value is None or ours_value is None:
                missing += 1
                continue
            if math.isnan(solver_value) or math.isnan(ours_value):
                missing += 1
                continue
            solver_values.append(solver_value)
            ours_values.append(ours_value)
        out[metric] = MetricMean(
            solver=sum(solver_values) / len(solver_values) if solver_values else float("nan"),
            ours=sum(ours_values) / len(ours_values) if ours_values else float("nan"),
            total=len(solver_values),
            missing=missing,
        )
    return out


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0.0 or y_var == 0.0:
        return float("nan")
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return cov / math.sqrt(x_var * y_var)


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for idx in range(start, end):
            ranks[order[idx]] = rank
        start = end
    return ranks


def preference_correlations(
    votes: list[Vote],
    stems: list[str],
    solver: str,
    ours: str,
    solver_metrics: dict[str, dict[str, float]],
    ours_metrics: dict[str, dict[str, float]],
    dim_ratios: dict[str, dict[str, float]],
) -> dict[str, PreferenceCorrelation]:
    by_stem: dict[str, Counter[str]] = defaultdict(Counter)
    for vote in votes:
        by_stem[vote.stem][vote.preferred] += 1

    out = {key: PreferenceCorrelation() for key, _, _ in METRICS}
    for metric, _label, lower_is_better in METRICS:
        metric_advantages = []
        preference_margins = []
        missing = 0
        for stem in stems:
            solver_value, ours_value = metric_values(
                metric, stem, solver, ours, solver_metrics, ours_metrics, dim_ratios
            )
            if solver_value is None or ours_value is None:
                missing += 1
                continue
            if math.isnan(solver_value) or math.isnan(ours_value):
                missing += 1
                continue
            metric_advantage = (
                solver_value - ours_value
                if lower_is_better
                else ours_value - solver_value
            )
            preference_margin = (by_stem[stem][ours] - by_stem[stem][solver]) / 4.0
            metric_advantages.append(metric_advantage)
            preference_margins.append(preference_margin)
        out[metric] = PreferenceCorrelation(
            pearson=_corr(metric_advantages, preference_margins),
            spearman=_corr(_rankdata(metric_advantages), _rankdata(preference_margins)),
            total=len(metric_advantages),
            missing=missing,
        )
    return out


def strict_majority_decisions(votes: list[Vote], solver: str, ours: str) -> list[tuple[str, str]]:
    by_stem: dict[str, Counter[str]] = defaultdict(Counter)
    for vote in votes:
        by_stem[vote.stem][vote.preferred] += 1

    decisions = []
    total_raters = len({vote.rater for vote in votes})
    for stem, counts in by_stem.items():
        for model in (solver, ours):
            if counts[model] > total_raters / 2:
                decisions.append((stem, model))
                break
    return decisions


def print_agreement(title: str, agreements: dict[str, Agreement]) -> None:
    print(title)
    for metric, label, _lower_is_better in METRICS:
        item = agreements[metric]
        pct = "nan" if math.isnan(item.pct) else f"{item.pct:.1f}"
        print(
            f"  {label:12s} {pct:>5s}% "
            f"({item.hit}/{item.total}; missing={item.missing}, metric_ties={item.metric_ties})"
        )


def main() -> None:
    args = parse_args()
    votes, solver, ours = load_votes(args.results_dir)
    valid_votes, invalid_by_stem = filter_valid_votes(votes, args.study_root, (solver, ours))
    if not args.include_invalid:
        votes = valid_votes
    stem_to_batch = {vote.stem: vote.batch for vote in votes}

    solver_metrics = load_per_sketch_metrics(args.solver_metrics)
    ours_metrics = load_per_sketch_metrics(args.ours_metrics)
    dim_ratios = load_dim_ratios(args.study_root, (solver, ours), stem_to_batch)

    raw_counts = Counter(vote.preferred for vote in votes)
    by_rater: dict[str, Counter[str]] = defaultdict(Counter)
    for vote in votes:
        by_rater[vote.rater][vote.preferred] += 1

    individual_decisions = [
        (vote.stem, vote.preferred)
        for vote in votes
        if vote.preferred in {solver, ours}
    ]
    majority_decisions = strict_majority_decisions(votes, solver, ours)
    pair_wins = metric_pair_wins(
        sorted(stem_to_batch), solver, ours, solver_metrics, ours_metrics, dim_ratios
    )
    means = metric_means(
        sorted(stem_to_batch), solver, ours, solver_metrics, ours_metrics, dim_ratios
    )
    correlations = preference_correlations(
        votes,
        sorted(stem_to_batch),
        solver,
        ours,
        solver_metrics,
        ours_metrics,
        dim_ratios,
    )

    print("User-study result files:", len(by_rater))
    if invalid_by_stem:
        verb = "Included" if args.include_invalid else "Excluded"
        print(f"{verb} invalid sketches:", len(invalid_by_stem))
        for stem in sorted(invalid_by_stem):
            labels = ", ".join(model_short(model) for model in invalid_by_stem[stem])
            print(f"  {stem} ({labels})")
    print("Sketches:", len(stem_to_batch))
    print("Judgments:", len(votes))
    print(f"Solver-RL: {solver}")
    print(f"Ours:      {ours}")
    print()

    print("Individual votes")
    print(f"  Solver-RL: {raw_counts[solver]}")
    print(f"  Ours:      {raw_counts[ours]}")
    print(f"  Tie:       {raw_counts['Tie']}")
    decisive = raw_counts[solver] + raw_counts[ours]
    if decisive:
        print(f"  Ours share among decisive votes: {100.0 * raw_counts[ours] / decisive:.1f}%")
    print()

    print("Per-rater votes")
    for rater in sorted(by_rater):
        counts = by_rater[rater]
        print(
            f"  {rater:8s} Solver-RL={counts[solver]:2d} "
            f"Ours={counts[ours]:2d} Tie={counts['Tie']:2d}"
        )
    print()

    print_agreement(
        "Pref. agree, individual non-tie preferences",
        agreement_for_decisions(
            individual_decisions, solver, ours, solver_metrics, ours_metrics, dim_ratios
        ),
    )
    print()
    print_agreement(
        "Audit: Pref. agree, strict per-sketch model majorities",
        agreement_for_decisions(
            majority_decisions, solver, ours, solver_metrics, ours_metrics, dim_ratios
        ),
    )
    print()

    print("Audit: metric pair wins on study sketches (Solver-RL/Ours/Tie)")
    for metric, label, _lower_is_better in METRICS:
        item = pair_wins[metric]
        print(
            f"  {label:12s} {item.solver}/{item.ours}/{item.ties} "
            f"(missing={item.missing})"
        )
    print()

    print("Preference-margin correlation on study sketches")
    print("  margin = (Ours votes - Solver-RL votes) / 4")
    print("  metric advantage is signed so positive means Ours is better")
    for metric, label, _lower_is_better in METRICS:
        item = correlations[metric]
        pearson = "nan" if math.isnan(item.pearson) else f"{item.pearson:.3f}"
        spearman = "nan" if math.isnan(item.spearman) else f"{item.spearman:.3f}"
        print(
            f"  {label:12s} Pearson={pearson:>6s} Spearman={spearman:>6s} "
            f"(n={item.total}, missing={item.missing})"
        )
    print()

    print("Metric means on study sketches")
    for metric, label, _lower_is_better in METRICS:
        item = means[metric]
        solver_mean = "nan" if math.isnan(item.solver) else f"{item.solver:.3f}"
        ours_mean = "nan" if math.isnan(item.ours) else f"{item.ours:.3f}"
        print(
            f"  {label:12s} Solver-RL={solver_mean:>6s} Ours={ours_mean:>6s} "
            f"(n={item.total}, missing={item.missing})"
        )
    print()

    individual = agreement_for_decisions(
        individual_decisions, solver, ours, solver_metrics, ours_metrics, dim_ratios
    )
    row = []
    for metric, _label, _lower_is_better in METRICS:
        pct = individual[metric].pct
        row.append("nan" if math.isnan(pct) else f"{pct:.1f}")
    print("Main table values:")
    print("  Votes:", raw_counts[solver], raw_counts[ours], "ties", raw_counts["Tie"])
    solver_means = []
    ours_means = []
    for metric, _label, _lower_is_better in METRICS:
        solver_means.append(f"{means[metric].solver:.3f}")
        ours_means.append(f"{means[metric].ours:.3f}")
    print("  Solver-RL metric means:", ", ".join(solver_means))
    print("  Ours metric means:", ", ".join(ours_means))
    corr_row = []
    for metric, _label, _lower_is_better in METRICS:
        value = correlations[metric].spearman
        corr_row.append("nan" if math.isnan(value) else f"{value:.3f}")
    print("  Pref. corr. Spearman:", ", ".join(corr_row))
    print()
    print("Agreement audit, not shown in main table:")
    print("  Pref. agree (%):", ", ".join(row))


if __name__ == "__main__":
    main()
