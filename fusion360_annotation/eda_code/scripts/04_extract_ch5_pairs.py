from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from fusion360_annotation.eda_code.bundle_loader import build_bundle
from fusion360_annotation.eda_code.canonical import metric_base_eids
from fusion360_annotation.eda_code.config import BASE_POLICY
from fusion360_annotation.eda_code.dataset_features import display_type
from fusion360_annotation.eda_code.manifest import discover_groups
from fusion360_annotation.eda_code.progress import progress
from fusion360_annotation.eda_code.metrics_v4 import V4Hyperparams, compute_metrics_v4  # noqa: E402
from src.metrics.constraint_token_f1 import pair_f1  # noqa: E402
from src.metrics.edit_response_distance import _is_relational, collect_probes, constraint_signature  # noqa: E402
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def build_ch5_bundle(group, converter, geometry_tol: float):
    """Build a human bundle for the token-vs-motion figure from data_eval."""
    anns = tuple(sorted(group.files))
    bundle = build_bundle(
        group.batch,
        group.base_fcstd,
        group.files,
        converter,
        geometry_tol=geometry_tol,
    )
    bundle.remapped_sketches = bundle.sketches
    return bundle, anns


def per_type_agreement(sk_a, sk_b) -> dict[str, list[int]]:
    def grouped(sketch):
        out = defaultdict(set)
        for c in sketch.constraints.values():
            out[display_type(c, sketch.entities)].add(constraint_signature(c))
        return out

    ga, gb = grouped(sk_a), grouped(sk_b)
    return {
        t: [len(ga.get(t, set())), len(gb.get(t, set())), len(ga.get(t, set()) & gb.get(t, set()))]
        for t in set(ga) | set(gb)
    }


def compute_v4(pred, gt, probes, weights, ctx, base_eids, hp):
    rels = [c for c in gt.constraints.values() if _is_relational(c)]
    return compute_metrics_v4(
        pred_sketch=pred,
        gt_sketch=gt,
        edit_probes=probes,
        edit_weights=weights,
        rels=rels,
        context=ctx,
        base_eids=base_eids,
        hp=hp,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--geometry-tol", type=float, default=0.5)
    ap.add_argument("--limit-files", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = SketchContext()
    converter = SketchFreecadConverter(ctx, verbose=False, raise_on_error=False)
    hp = V4Hyperparams()
    complete, _incomplete, _stats = discover_groups()

    rows = []
    skipped = []
    n_ok = n_fail = processed = 0
    selected = complete[: args.limit_files] if args.limit_files is not None else complete
    for group in progress(selected, desc="ch5 sketches", total=len(selected)):
        try:
            bundle, anns = build_ch5_bundle(
                group,
                converter,
                geometry_tol=args.geometry_tol,
            )
            base_eids = metric_base_eids(BASE_POLICY, bundle.canonical_sketch)
            sketches = bundle.remapped_sketches
            probes, weights = collect_probes([sketches[a] for a in anns], kind="dim")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"folder": group.batch, "file": group.sketch_id, "reason": str(exc)})
            continue
        processed += 1

        for a, b in itertools.combinations(anns, 2):
            rec = {
                "folder": group.batch,
                "sketch_id": group.sketch_id,
                "ann_a": a,
                "ann_b": b,
            }
            try:
                rec["token_f1"] = float(pair_f1(sketches[a], sketches[b])["f1"])
                fwd = compute_v4(sketches[a], sketches[b], probes, weights, ctx, base_eids, hp)
                bwd = compute_v4(sketches[b], sketches[a], probes, weights, ctx, base_eids, hp)
                rec["d_sub_angle"] = float(fwd["d_sub_angle"])
                rec["d_H_angle"] = 0.5 * (float(fwd["d_H_angle"]) + float(bwd["d_H_angle"]))
                rec["leak_angle"] = 0.5 * (float(fwd["leak_angle"]) + float(bwd["leak_angle"]))
                rec["dim_a"] = int(fwd["dim_P"])
                rec["dim_b"] = int(fwd["dim_G"])
                rec["per_type"] = per_type_agreement(sketches[a], sketches[b])
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                n_fail += 1
            rows.append(rec)

    summary = {
        "n_complete_groups": len(complete),
        "n_sketches_processed": processed,
        "n_pairs": len(rows),
        "n_pairs_ok": n_ok,
        "n_pairs_fail": n_fail,
        "n_skipped_sketches": len(skipped),
        "skipped": skipped[:50],
    }
    (out_dir / "ch5_pairs.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "ch5_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
