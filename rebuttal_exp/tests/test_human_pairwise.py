"""Pair construction is the whole experiment, so it is tested rather than trusted.

The synthetic rows cover the leave-one-out rid layout, the canonical-direction choice for the
symmetric metrics, the refusal to symmetrize theta_leak, and the float round trip that keeps a
1e-15 leak angle from being written as literal zero. The last test is a regression gate: the
min/max re-aggregation on the real motion_eval.csv must reproduce Appendix Fig. 8, otherwise the
loader disagrees with the published evaluation.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rebuttal_exp.common import RESOLVED_METRIC_FLOOR
from rebuttal_exp.human_pairwise import (
    DIRECTIONAL,
    _panel_note,
    PLACE,
    SYMMETRIC,
    _f,
    best_reference_values,
    build_pairs,
    check_fig8_reproduction,
    directed_scores,
    pair_rows,
    pair_values,
    per_sketch_rows,
    query_annotator,
    summary_table,
)
from rebuttal_exp.shared_reference import Sample, load_samples, quantile

MOTION_EVAL = (
    Path(__file__).resolve().parents[2] / "fusion360_annotation_" / "eda_code" / "out" / "motion_eval.csv"
)

# One synthetic sketch with three annotations A, B, C. Values are keyed by the ordered pair
# (pred, gt) so a test can assert exactly which direction ended up where.
LEAK = {
    ("A", "B"): 0.10, ("B", "A"): 0.80,
    ("A", "C"): 0.20, ("C", "A"): 0.90,
    ("B", "C"): 0.30, ("C", "B"): 0.70,
}
FULL = {frozenset("AB"): 0.4, frozenset("AC"): 0.5, frozenset("BC"): 0.0}
REL = {frozenset("AB"): 0.6, frozenset("AC"): 0.1, frozenset("BC"): 0.0}
PLACE_V = {frozenset("AB"): 0.2, frozenset("AC"): 0.9, frozenset("BC"): 1.0}


def make_samples(names=("A", "B", "C"), *, batch="0", filename="s.FCStd", variant="heldout"):
    """The leave-one-out rid layout of 06_run_motion_eval.py: C(3,2) refs, the third held out."""
    import itertools

    samples = []
    for refs in itertools.combinations(names, 2):
        heldout = next(n for n in names if n not in refs)
        values = {m: {} for m in ("theta_full", "theta_rel", "theta_leak", PLACE)}
        for ref in refs:
            key = frozenset((heldout, ref))
            values["theta_full"][ref] = FULL[key]
            values["theta_rel"][ref] = REL[key]
            values[PLACE][ref] = PLACE_V[key]
            values["theta_leak"][ref] = LEAK[(heldout, ref)]
        samples.append(
            Sample(
                rid=f"{batch}:{filename}:{'+'.join(refs)}__q={heldout}",
                batch=batch,
                filename=filename,
                variant=variant,
                refs=refs,
                values=values,
            )
        )
    return samples


def build(samples):
    directed, counts = directed_scores(samples)
    pairs, pair_counts = build_pairs(directed)
    return pairs, counts, pair_counts


def test_query_annotator_survives_ids_containing_colons_and_underscores():
    rid = "3:23132_1847c4ef_0002_sketch01.FCStd:J+R__q=S"
    assert query_annotator(rid) == "S"
    with pytest.raises(ValueError):
        query_annotator("3:s.FCStd:J+R")


def test_three_annotations_give_three_pairs_and_six_leak_values():
    pairs, counts, _ = build(make_samples())
    assert len(pairs) == 3
    assert {(p.annotator_i, p.annotator_j) for p in pairs} == {("A", "B"), ("A", "C"), ("B", "C")}
    assert counts["directed_observations"] == 6
    assert len(pair_values(pairs)[DIRECTIONAL]) == 6


def test_no_self_comparison_and_no_pair_counted_twice():
    pairs, counts, _ = build(make_samples())
    assert counts["self_comparisons"] == 0
    assert all(p.annotator_i != p.annotator_j for p in pairs)
    keys = [(p.batch, p.filename, p.annotator_i, p.annotator_j) for p in pairs]
    assert len(set(keys)) == len(keys)


def test_only_variant_heldout_is_used():
    samples = make_samples() + make_samples(variant="solver_rl")
    pairs, counts, _ = build(samples)
    assert counts["heldout_samples"] == 3
    assert len(pairs) == 3


def test_symmetric_metrics_take_the_canonical_direction_and_record_the_asymmetry():
    pairs, _, pair_counts = build(make_samples())
    by_pair = {(p.annotator_i, p.annotator_j): p for p in pairs}
    assert by_pair[("A", "B")].symmetric["theta_full"] == pytest.approx(0.4)
    assert by_pair[("A", "C")].symmetric[PLACE] == pytest.approx(0.9)
    # the synthetic values are exactly symmetric, so every diagnostic must be zero
    assert all(v == 0.0 for p in pairs for v in p.asymmetry.values())
    assert all(v == 0.0 for v in pair_counts["max_asymmetry"].values())
    assert set(pair_counts["max_asymmetry"]) == set(SYMMETRIC)


def test_annotator_order_does_not_change_the_result():
    forward, _, _ = build(make_samples(names=("A", "B", "C")))
    reverse, _, _ = build(make_samples(names=("C", "B", "A")))

    def key(pairs):
        return sorted(
            (p.annotator_i, p.annotator_j, p.symmetric["theta_full"], p.leak_i_to_j, p.leak_j_to_i)
            for p in pairs
        )

    assert key(forward) == key(reverse)


def test_asymmetry_above_the_floor_aborts():
    samples = make_samples()
    # break theta_full in one direction only, by more than the sqrt(eps) noise floor
    for sample in samples:
        if query_annotator(sample.rid) == "A" and "B" in sample.refs:
            sample.values["theta_full"]["B"] = 0.4 + 1e-3
    directed, _ = directed_scores(samples)
    with pytest.raises(SystemExit, match="not symmetric"):
        build_pairs(directed)


def test_asymmetry_inside_the_floor_is_tolerated():
    samples = make_samples()
    for sample in samples:
        if query_annotator(sample.rid) == "A" and "B" in sample.refs:
            sample.values["theta_full"]["B"] = 0.4 + 1e-8
    directed, _ = directed_scores(samples)
    pairs, counts = build_pairs(directed)
    assert len(pairs) == 3
    assert counts["max_asymmetry"]["theta_full"] == pytest.approx(1e-8)


def test_theta_leak_keeps_both_directions_and_is_never_averaged():
    pairs, _, _ = build(make_samples())
    by_pair = {(p.annotator_i, p.annotator_j): p for p in pairs}
    ab = by_pair[("A", "B")]
    assert ab.leak_i_to_j == pytest.approx(0.10)
    assert ab.leak_j_to_i == pytest.approx(0.80)
    assert sorted(pair_values(pairs)[DIRECTIONAL]) == sorted(LEAK.values())


def test_a_pair_missing_one_direction_is_dropped_not_half_counted():
    samples = make_samples()
    directed, _ = directed_scores(samples)
    del directed[("0", "s.FCStd")][("B", "A")]
    pairs, counts = build_pairs(directed)
    assert counts["pairs_missing_a_direction"] == 1
    assert {(p.annotator_i, p.annotator_j) for p in pairs} == {("A", "C"), ("B", "C")}


def test_all_pairs_protocol_applies_no_min_or_max():
    samples = make_samples()
    pairs, _, _ = build(samples)
    values = pair_values(pairs)
    assert sorted(values["theta_full"]) == sorted(FULL.values())
    # the best-reference protocol on the same rows does apply min/max, and differs
    best = best_reference_values(samples)
    assert sorted(best["theta_full"]) == [0.0, 0.0, 0.4]
    assert sorted(best[PLACE]) == [0.9, 1.0, 1.0]


def test_summary_quantiles_match_the_appendix_figure_convention():
    """12_plot_heldout_distribution.percentile is a linear interpolation on q in [0, 1]."""

    def reference_percentile(sorted_xs, q):
        pos = (len(sorted_xs) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(sorted_xs) - 1)
        frac = pos - lo
        return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac

    xs = [0.0, 0.1, 0.25, 0.4, 0.55, 0.8, 0.95]
    for q in (0.25, 0.5, 0.75, 0.9):
        assert quantile(xs, 100 * q) == pytest.approx(reference_percentile(sorted(xs), q))


def test_summary_reports_iqr_and_p90_per_protocol():
    samples = make_samples()
    pairs, _, _ = build(samples)
    rows = summary_table(pair_values(pairs), best_reference_values(samples), RESOLVED_METRIC_FLOOR)
    assert {r["protocol"] for r in rows} == {"all_pairs", "best_reference"}
    full = next(r for r in rows if r["protocol"] == "all_pairs" and r["metric"] == "theta_full")
    assert full["n"] == 3
    assert full["iqr"] == pytest.approx(full["q3"] - full["q1"])
    leak = next(r for r in rows if r["protocol"] == "all_pairs" and r["metric"] == DIRECTIONAL)
    assert leak["n"] == 6, "theta_leak is counted per direction, not per pair"


def test_per_sketch_aggregation_picks_max_for_angles_and_min_for_placement():
    pairs, _, _ = build(make_samples())
    rows = per_sketch_rows(pairs, RESOLVED_METRIC_FLOOR)
    assert len(rows) == 1
    row = rows[0]
    assert row["sketch_id"] == "s"
    assert row["n_pairs"] == 3
    assert row["annotators"] == "ABC"
    assert float(row["max_pairwise_theta_full"]) == pytest.approx(0.5)
    assert float(row["min_pairwise_S_place"]) == pytest.approx(0.2)
    assert float(row["max_pairwise_theta_leak"]) == pytest.approx(0.9)
    assert row["n_pairs_below_eps"] == 1
    assert row["agreement_label"] == "mixed"


def test_agreement_label_extremes():
    pairs, _, _ = build(make_samples())
    consensus = [
        type(p)(p.batch, p.filename, p.annotator_i, p.annotator_j,
                {**p.symmetric, "theta_full": 0.0}, p.asymmetry, p.leak_i_to_j, p.leak_j_to_i)
        for p in pairs
    ]
    assert per_sketch_rows(consensus, RESOLVED_METRIC_FLOOR)[0]["agreement_label"] == "consensus"
    divergent = [
        type(p)(p.batch, p.filename, p.annotator_i, p.annotator_j,
                {**p.symmetric, "theta_full": 0.5}, p.asymmetry, p.leak_i_to_j, p.leak_j_to_i)
        for p in pairs
    ]
    assert per_sketch_rows(divergent, RESOLVED_METRIC_FLOOR)[0]["agreement_label"] == "divergent"


def test_near_zero_values_are_not_flushed_to_literal_zero(tmp_path):
    assert _f(7.456170016064916e-15) == "7.456170016064916e-15"
    assert float(_f(1e-300)) == 1e-300
    samples = make_samples()
    for sample in samples:
        for ref in sample.refs:
            sample.values["theta_leak"][ref] = 2.2788422472883606e-15
    pairs, _, _ = build(samples)
    path = tmp_path / "pairs.csv"
    from rebuttal_exp.common import write_csv
    from rebuttal_exp.human_pairwise import PAIR_COLUMNS

    write_csv(path, pair_rows(pairs, {}), PAIR_COLUMNS)
    written = list(csv.DictReader(path.open()))
    assert all(float(r["theta_dim_leak_i_to_j"]) == 2.2788422472883606e-15 for r in written)


def test_annotation_ids_are_filled_from_the_manifest_and_tolerate_its_absence():
    pairs, _, _ = build(make_samples())
    ids = {("0", "s.FCStd"): {"A": "s_A.FCStd", "B": "s_B.FCStd", "C": "s_C.FCStd"}}
    row = next(r for r in pair_rows(pairs, ids) if r["annotator_i"] == "A")
    assert row["annotation_i_id"] == "s_A.FCStd"
    assert all(r["annotation_i_id"] == "" for r in pair_rows(pairs, {}))


def test_panel_note_lines_up_and_carries_both_protocols():
    """The figure's whole point is best-ref median next to all-pair median, so it is tested."""
    stats = {
        "best_reference": {"theta_full": {"mean": 0.16747, "median": 2.4333e-08}},
        "all_pairs": {"theta_full": {"mean": 0.24860, "median": 0.31623}},
    }
    lines = _panel_note(stats, "theta_full").split("\n")
    assert len(lines) == 3
    assert len({len(line) for line in lines}) == 1, "columns must be padded to a common width"
    assert lines[0].split() == ["mean", "median"]
    assert lines[1].startswith("best-ref") and "2.4e-08" in lines[1] and "0.17" in lines[1]
    assert lines[2].startswith("all pairs") and "0.32" in lines[2] and "0.25" in lines[2]


@pytest.mark.skipif(not MOTION_EVAL.exists(), reason="motion_eval.csv is not available")
def test_min_max_reaggregation_reproduces_appendix_figure_8():
    samples, _, _ = load_samples(MOTION_EVAL)
    directed, counts = directed_scores(samples)
    pairs, pair_counts = build_pairs(directed)
    rows = summary_table(pair_values(pairs), best_reference_values(samples), RESOLVED_METRIC_FLOOR)
    assert check_fig8_reproduction(rows) == []
    assert len({p.cluster for p in pairs}) == 286
    assert len(pairs) == 858
    assert len({(p.batch, p.filename, a) for p in pairs for a in (p.annotator_i, p.annotator_j)}) == 858
    assert len(pair_values(pairs)[DIRECTIONAL]) == 1716
    assert counts["self_comparisons"] == 0
    assert pair_counts["pairs_missing_a_direction"] == 0
    assert all(v <= RESOLVED_METRIC_FLOOR for v in pair_counts["max_asymmetry"].values())


@pytest.mark.skipif(not MOTION_EVAL.exists(), reason="motion_eval.csv is not available")
def test_the_gate_fires_when_the_statistics_move():
    samples, _, _ = load_samples(MOTION_EVAL)
    pairs, _, _ = build(samples)
    best = best_reference_values(samples)
    best["theta_full"] = [v + 0.05 for v in best["theta_full"]]
    rows = summary_table(pair_values(pairs), best, RESOLVED_METRIC_FLOOR)
    assert any("theta_full" in problem for problem in check_fig8_reproduction(rows))
