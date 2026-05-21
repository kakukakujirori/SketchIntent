"""Pre-pass: token-level Constraint F1 per rotation × prediction.

Computes constraint-token F1 (combined + dimensional + relational) between
each candidate prediction and the rotation's reference annotators, taking
max F1 over references (multi-reference). Outputs a CSV that
`analyze_model_ranking.py` later merges with d_feas / OES result CSVs.

Predictions covered per rotation:
- the heldout human (`heldout`) — paired with the same refs as drop_rate=0
- every loaded Vitruvion model variant from `--vitruvion-spec`

The script intentionally does not run any solver / Newton step — it only
needs canonicalised sketches, so it is fast (CPU only, no FreeCAD probes).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.dirname(__file__)
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from exp_source_conditioned_dropout import (
    DEFAULT_BATCHES,
    build_file_bundle,
    get_fcstd_map,
    parse_human_specs,
    parse_vitruvion_specs,
)
from src.metrics.constraint_token_f1 import multi_ref_f1
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vitruvion-root", default=None,
                   help="Legacy single-model root. Prefer --vitruvion-spec.")
    p.add_argument("--vitruvion-spec", action="append", default=None,
                   help="label:path, repeatable.")
    p.add_argument("--batches", nargs="+", default=list(DEFAULT_BATCHES))
    p.add_argument("--human", action="append", default=None)
    p.add_argument("--ref-size", type=int, default=2)
    p.add_argument("--geometry-tol", type=float, default=5e-1)
    p.add_argument("--canonical-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-csv",
                   default="artifacts/constraint_token_f1.csv")
    args = p.parse_args()

    humans_by_batch = parse_human_specs(args.human)
    model_specs = parse_vitruvion_specs(args.vitruvion_spec, args.vitruvion_root)
    if model_specs:
        print("[Vitruvion specs] " + ", ".join(f"{l}={p_}" for l, p_ in model_specs))

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    rows: list[dict] = []
    n_files = 0
    rotation_counter = 0

    for batch in args.batches:
        humans = [(n, d) for n, d in humans_by_batch.get(batch, [])
                  if os.path.isdir(d)]
        if len(humans) < args.ref_size + 1:
            print(f"[skip] batch {batch}: need {args.ref_size + 1} humans")
            continue
        file_maps = {n: get_fcstd_map(d) for n, d in humans}
        common = sorted(set.intersection(*(set(m) for m in file_maps.values())))
        print(f"Batch {batch}: common={len(common)}")

        for filename in common:
            if args.limit is not None and n_files >= args.limit:
                break
            n_files += 1
            try:
                bundle = build_file_bundle(
                    batch=batch,
                    filename=filename,
                    humans=humans,
                    file_maps=file_maps,
                    converter=converter,
                    geometry_tol=args.geometry_tol,
                    canonical_dir=args.canonical_dir,
                    model_specs=model_specs,
                )
            except Exception as exc:
                print(f"  [skip] {batch}/{filename}: {exc}")
                continue

            names = tuple(name for name, _ in humans)
            for refs in combinations(names, args.ref_size):
                for heldout in names:
                    if heldout in refs:
                        continue
                    rotation_counter += 1
                    rotation_id = f"{batch}/{filename}#{'+'.join(refs)}->{heldout}"
                    ref_sketches = [bundle.sketches[r] for r in refs]
                    preds: list[tuple[str, float, object]] = []
                    if heldout in bundle.sketches:
                        preds.append((heldout, 0.0, bundle.sketches[heldout]))
                    for lbl in bundle.loaded_model_labels():
                        preds.append((lbl, -1.0, bundle.sketches[lbl]))

                    for pred_name, drop_rate, pred_sketch in preds:
                        f1_row = multi_ref_f1(pred_sketch, ref_sketches)
                        out_row = {
                            "batch": batch,
                            "filename": filename,
                            "rotation_id": rotation_id,
                            "refs": "|".join(refs),
                            "heldout": heldout,
                            "drop_rate": drop_rate,
                            "pred": pred_name,
                            **f1_row,
                        }
                        rows.append(out_row)

    full = args.output_csv if os.path.isabs(args.output_csv) \
        else os.path.join(REPO_ROOT, args.output_csv)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with open(full, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {full}  (rows={len(rows)}, rotations={rotation_counter})")


if __name__ == "__main__":
    main()
