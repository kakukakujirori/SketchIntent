"""Fiverr recall/precision harness for the rule-based constraint detector.

For each human (fiverr) sketch we run ``rule_augmenter.detect_candidates`` on its
geometry and match the predicted constraints against the human annotations, then
report, per constraint (sub-)type and as a function of a tolerance scale:

  - **recall**    = TP / (TP + FN): of the relations a human drew, how many the
    detector recovers.  This calibrates the per-family tolerance and exposes the
    "trustworthy" types (high recall) vs the structurally-missed ones
    (e.g. Coincident.collinear, latent Mirror axes).
  - **precision** = TP / (TP + FP): of what the detector predicts, how much the
    human also drew.  Precision is EXPECTED to be < 1 because the detector is
    maximal (adds every true predicate) while humans annotate sparsely — so low
    precision here is mostly *density vs human*, not error.  Read it as a knob.

Matching signature: ``(effective_subtype_label, frozenset(base_entity_ids))``.
Base-entity sets (rather than exact ref strings) make the match robust to
subnode/ref-form differences — e.g. a human 3-ref ``Midpoint(L.start, L.end, p)``
and the detector's 2-ref ``Midpoint(p, L)`` both reduce to ``{L, p}`` — while
staying ~unique per constraint (at most one relation of a given label holds
between a given entity set).

Fiverr is EVAL-only: this script only *reads* the human data to calibrate the
detector; nothing here is trained on.  See plan §14.6.

Example:
    conda run -n freecad python scripts/eval_rule_augmenter_recall.py \\
        --fiverr-root /home/ryotaro/Downloads/fiverr \\
        --annotators Eko Johnny Ryotaro Sagar Tanaka --limit 200
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import Counter, defaultdict

from sketchgraphs.data import LocalReferenceParameter

from src.metrics.constraint_subtypes import classify_constraint_label
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.rule_augmenter import (
    DEFAULT_TOLERANCES,
    DETECTABLE_TYPES,
    detect_candidates,
    filter_candidates,
)
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter


def _base_ids(refs) -> frozenset[str]:
    return frozenset(str(r).split(".", 1)[0] for r in refs)


def _signature(label: str, refs) -> tuple[str, frozenset[str]]:
    return (label, _base_ids(refs))


def _human_signatures(sketch) -> tuple[set, Counter]:
    """Signatures of the human-drawn constraints + a per-label count."""
    sigs: set = set()
    by_label: Counter = Counter()
    for c in sketch.constraints.values():
        refs = [p.value for p in c.parameters if isinstance(p, LocalReferenceParameter)]
        if not refs:
            continue
        label = classify_constraint_label(c, sketch)
        sigs.add(_signature(label, refs))
        by_label[label] += 1
    return sigs, by_label


def _base_type(label: str) -> str:
    return label.split(".", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fiverr-root", default="/home/ryotaro/Downloads/fiverr")
    parser.add_argument("--annotators", nargs="+",
                        default=["Eko", "Johnny", "Ryotaro", "Sagar", "Tanaka"],
                        help="Human annotator folders (exclude Vitruvion* model outputs).")
    parser.add_argument("--limit", type=int, default=None, help="Max sketches (for a quick pass).")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0],
                        help="Tolerance multipliers applied to DEFAULT_TOLERANCES.")
    args = parser.parse_args()

    context = SketchContext()
    converter = SketchFreecadConverter(context, raise_on_error=False)

    files: list[str] = []
    for a in args.annotators:
        files += sorted(glob.glob(os.path.join(args.fiverr_root, a, "**", "*.py"), recursive=True))
    if args.limit is not None:
        files = files[: args.limit]

    # stats[scale][label] = [tp, fp, fn]
    stats: dict[float, dict[str, list[int]]] = {
        s: defaultdict(lambda: [0, 0, 0]) for s in args.scales
    }
    human_support: Counter = Counter()
    n_sketches = n_errors = 0

    for i, f in enumerate(files):
        try:
            sketch = converter.from_freecad_commands(f)
            human_sigs, by_label = _human_signatures(sketch)
            candidates = detect_candidates(sketch, context)
        except Exception:
            n_errors += 1
            continue
        n_sketches += 1
        human_support += by_label

        for scale in args.scales:
            tol = {k: v * scale for k, v in DEFAULT_TOLERANCES.items()}
            pred_sigs = {_signature(c.subtype, c.refs) for c in filter_candidates(candidates, tol)}
            labels = {s[0] for s in human_sigs} | {s[0] for s in pred_sigs}
            for label in labels:
                h = {s for s in human_sigs if s[0] == label}
                p = {s for s in pred_sigs if s[0] == label}
                st = stats[scale][label]
                st[0] += len(h & p)
                st[1] += len(p - h)
                st[2] += len(h - p)

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(files)} files", flush=True)

    print(f"\nannotators={args.annotators}")
    print(f"sketches={n_sketches} errors={n_errors} "
          f"human_constraints={sum(human_support.values())}")
    print(f"detectable types: {', '.join(DETECTABLE_TYPES)}\n")

    detectable = set(DETECTABLE_TYPES)
    for scale in args.scales:
        tol = {k: round(v * scale, 5) for k, v in DEFAULT_TOLERANCES.items()}
        print(f"=== tolerance scale {scale}  (orientation={tol['orientation']}, "
              f"position={tol['position']}, size={tol['size']}) ===")
        print(f"{'label':28s} {'support':>7s} {'TP':>5s} {'FP':>6s} {'FN':>5s} "
              f"{'recall':>7s} {'prec':>6s}")
        labels_sorted = sorted(stats[scale], key=lambda l: -(stats[scale][l][0] + stats[scale][l][2]))
        micro = [0, 0, 0]
        for label in labels_sorted:
            tp, fp, fn = stats[scale][label]
            support = tp + fn
            recall = tp / support if support else float("nan")
            prec = tp / (tp + fp) if (tp + fp) else float("nan")
            print(f"{label:28s} {support:7d} {tp:5d} {fp:6d} {fn:5d} {recall:7.3f} {prec:6.3f}")
            if _base_type(label) in detectable:
                micro[0] += tp
                micro[1] += fp
                micro[2] += fn
        m_tp, m_fp, m_fn = micro
        m_rec = m_tp / (m_tp + m_fn) if (m_tp + m_fn) else float("nan")
        m_prec = m_tp / (m_tp + m_fp) if (m_tp + m_fp) else float("nan")
        print(f"{'MICRO (detectable types)':28s} {m_tp + m_fn:7d} {m_tp:5d} {m_fp:6d} "
              f"{m_fn:5d} {m_rec:7.3f} {m_prec:6.3f}\n")


if __name__ == "__main__":
    main()
