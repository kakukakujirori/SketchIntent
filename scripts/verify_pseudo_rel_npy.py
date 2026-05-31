"""Verify a pseudo-rel npy is training-ready and report its relation mix.

Loads the npy through the REAL ConstraintDataset path (rloo context:
max_token_length=192, align=4, random_shuffle, primitive noise) and iterates
__getitem__ to confirm it never raises (incl. the >3-ref tokenizer guard), then
tallies the relation head-type / subtype distribution of the GT the reward sees.

Usage:
    conda run -n freecad python -m scripts.verify_pseudo_rel_npy \
        data_processed/sg_pseudo_rel_v2_full_all.npy [--limit 1000]
"""
import argparse
from collections import Counter

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
from sketchgraphs.data import flat_array

from src.data.components.constraint_dataset import ConstraintDataset
from src.data.components.primitive_noise import PrimitiveNoise
from src.metrics.constraint_subtypes import classify_constraint_label
from src.metrics.edit_response_distance import _is_relational
from src.sketch_handlers.context import SketchContext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("--limit", type=int, default=1000, help="Max __getitem__ calls (None-ish: pass a big number).")
    ap.add_argument("--progress_every", type=int, default=20000, help="Print progress every N items; 0 disables it.")
    args = ap.parse_args()

    raw = flat_array.load_dictionary_flat(args.npy)
    raw_lengths = {k: len(v) for k, v in raw.items() if hasattr(v, "__len__")}
    raw_n = raw_lengths.get("sequences")
    bad_raw_lengths = {k: v for k, v in raw_lengths.items() if v != raw_n}
    if bad_raw_lengths:
        raise ValueError(f"raw npy metadata length mismatch vs sequences={raw_n}: {bad_raw_lengths}")

    ctx = SketchContext(max_token_length=192, num_quantization_bins=64,
                        align_constraint_token_length_to=4, use_stop_padding=False,
                        constraint_order="random_shuffle")
    ds = ConstraintDataset(args.npy, context=ctx,
                           primitive_noise=PrimitiveNoise(std=0.15, max_diff=0.15),
                           relational_only=False)
    lengths = {"sequences": len(ds.sequences)}
    if ds.source_files is not None:
        lengths["source_files"] = len(ds.source_files)
    if ds.sketch_ids is not None:
        lengths["sketch_ids"] = len(ds.sketch_ids)
    bad_lengths = {k: v for k, v in lengths.items() if v != len(ds)}
    if bad_lengths:
        raise ValueError(f"metadata length mismatch vs sequences={len(ds)}: {bad_lengths}")
    n = min(args.limit, len(ds))
    print(f"{args.npy}: dataset size {len(ds)}; iterating {n} via __getitem__")
    print(f"raw metadata lengths ok: {raw_lengths}")
    print(f"dataset metadata lengths ok: {lengths}")

    errs = 0
    head: Counter = Counter()
    sub: Counter = Counter()
    n_rel = []
    cap = 0
    for i in range(n):
        try:
            it = ds[i]
            sk = it["sketch"]
            rels = [c for c in sk.constraints.values() if _is_relational(c)]
            n_rel.append(len(rels))
            for c in rels:
                head[c.constraint_type.name] += 1
                sub[classify_constraint_label(c, sk)] += 1
            if int(it["c_val"].numpy()[-1]) != 0:
                cap += 1
        except Exception as e:
            errs += 1
            if errs <= 8:
                print(f"  [err] item {i}: {type(e).__name__}: {e}")
        if args.progress_every and (i + 1) % args.progress_every == 0:
            print(f"  progress {i + 1}/{n}  ok={i + 1 - errs}  errors={errs}", flush=True)

    ok = n - errs
    print(f"\n__getitem__ ok={ok}  errors={errs}   <-- must be 0 for training-ready")
    if n_rel:
        print(f"relations/sketch: mean {np.mean(n_rel):.1f} max {max(n_rel)}")
    print(f"c_val hit 192 cap (truncated, benign for reward): {cap}/{ok}")
    print("relation HEAD-type:")
    for t, c in head.most_common():
        print(f"  {t:16s} {c:7d}")
    print("subtypes of interest:")
    for t in ["Parallel", "Perpendicular", "Equal.length", "Equal.radius",
              "Tangent.edge", "Tangent.point", "Mirror", "Coincident.point_point",
              "Coincident.point_on_object", "Midpoint"]:
        print(f"  {t:28s} {sub.get(t, 0):7d}")


if __name__ == "__main__":
    main()
