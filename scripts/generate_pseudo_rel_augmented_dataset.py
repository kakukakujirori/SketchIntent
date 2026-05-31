"""Generate pseudo-relational GT (v2) with the rule augmenter (no model).

Unlike ``scripts/generate_pseudo_rel_dataset.py`` (which runs a rel-only RLOO
*model* to predict relational constraints), this script populates the pseudo-GT
purely geometrically with ``src.sketch_handlers.rule_augmenter``: for every input
sketch it detects all relations whose handler residual is ~0 at the real
geometry and writes them back as the preservation/target relation set R_G.

Why a separate script (see notebooks/rloo_response_metric_reward_research_plan.md
§15 and the pipeline trace in notebooks/pseudo_rel_v2_augmenter_pipeline_trace.md):

  * The augmenter emits the *full relational closure* (e.g. Horizontal+Horizontal
    AND the implied Parallel).  That closure is intentionally redundant.  The
    reward consumes the GT as a ``datalib.Sketch`` object (``batch['sketch']``)
    and reduces its relations to a motion subspace via an SVD rank, so redundant
    rows are absorbed (rank-bounded) — verified empirically: round-trip
    motion_relational dim is preserved 59/59 and dim never grows with added
    redundant relations.
  * ``generate_pseudo_rel_dataset.py`` solver-gates on ``Status == 'Converged'``,
    but ``get_solver_status`` reclassifies Converged+redundant -> ``'Redundant'``,
    so that gate would reject essentially every augmented sketch.  Solver
    validation is therefore OFF by default here; ``--solver_validate`` enables a
    *relaxed* gate that tolerates Redundant and only rejects
    Conflicting/Malformed/Distorting.

Safety gate (always on): every produced sequence is re-loaded through the EXACT
``ConstraintDataset.__getitem__`` preprocessing (strict ``cleanse_sketch_constraints``
+ ``complete_constraint_values``, no skip/delete) and its ``motion_relational_subspace``
is computed.  Samples that would raise at train time, or that end up with no
relational constraints, are dropped — so the saved npy is safe-by-construction
for training and has a well-defined relational GT subspace.

Typical usage:
    conda run -n freecad python -m scripts.generate_pseudo_rel_augmented_dataset \\
        --input_npy  data_processed/sg_filtered_unique_valid.npy \\
        --output_npy data_processed/sg_pseudo_rel_v2_full_all.npy \\
        --scale 0.25

    # exclude Mirror (precision 0.006) for the full_no_mirror ablation:
    ... --output_npy data_processed/sg_pseudo_rel_v2_no_mirror.npy --no_mirror

    # shard 0 of 8 for parallel generation:
    ... --shard 0/8 --output_npy data_processed/sg_pseudo_rel_v2_full_all.shard0.npy
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array

from src.metrics.edit_response_distance import _is_relational, compute_native_response
from src.models.components.solver_feedback import SolverFeedback
from src.models.rewards.utils.geometry import motion_relational_subspace
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.rule_augmenter import (
    DEFAULT_TOLERANCES,
    DETECTABLE_TYPES,
    augment_sketch,
)
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_constraints,
    cleanse_sketch_primitives,
    complete_constraint_values,
    normalize_sketch,
    remove_dimensional_constraints,
)

EPS = 1e-9


def build_output_npy(original_data: dict, valid_indices: list[int], new_sequences: list) -> np.ndarray:
    """Pack new sequences + preserved metadata (subset by valid_indices)."""
    out: dict = {"sequences": flat_array.save_list_flat(new_sequences)}
    for key, value in original_data.items():
        if key == "sequences":
            continue
        if isinstance(value, np.ndarray):
            out[key] = value[valid_indices]
        elif hasattr(value, "__getitem__"):
            out[key] = np.array([value[i] for i in valid_indices])
    return flat_array.pack_dictionary_flat(out)


def merge_npys(input_paths: list[str], output_path: str) -> int:
    """Concatenate several shard npys (same schema) into one. Order is preserved
    so per-row sequence<->metadata alignment from build_output_npy is kept."""
    seqs_all: list = []
    meta: dict[str, list] = {}
    for path in input_paths:
        d = flat_array.load_dictionary_flat(path)
        seqs = d["sequences"]
        for k in range(len(seqs)):
            seqs_all.append(seqs[k])
        for key, val in d.items():
            if key == "sequences":
                continue
            if isinstance(val, np.ndarray):
                meta.setdefault(key, []).append(val)
    out: dict = {"sequences": flat_array.save_list_flat(seqs_all)}
    for key, parts in meta.items():
        out[key] = np.concatenate(parts)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, flat_array.pack_dictionary_flat(out), allow_pickle=False)
    return len(seqs_all)


def dataset_preprocess(sketch, *, relational_only: bool):
    """Replicate ConstraintDataset.__getitem__ preprocessing (strict, no noise)."""
    if relational_only:
        sketch = remove_dimensional_constraints(sketch)
    normalize_sketch(sketch)
    sketch = cleanse_sketch_primitives(sketch)
    sketch = cleanse_sketch_constraints(sketch)          # strict (no delete_invalid)
    sketch, _ = complete_constraint_values(sketch)        # strict (no skip)
    return sketch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_npy", default="data_processed/sg_filtered_unique_valid.npy")
    p.add_argument("--output_npy", required=True)
    p.add_argument("--scale", type=float, default=0.25, help="Multiplier on DEFAULT_TOLERANCES (operating point 0.25).")
    p.add_argument("--no_mirror", action="store_true", help="Exclude Mirror (precision 0.006) -> full_no_mirror ablation.")
    p.add_argument("--types", nargs="+", default=None, help="Restrict to these head-types (default: all DETECTABLE_TYPES).")
    p.add_argument("--keep_original", action="store_true",
                   help="Augment on top of existing relations instead of stripping to bare primitives first.")
    p.add_argument("--solver_validate", action="store_true",
                   help="Relaxed FreeCAD gate: tolerate Redundant, reject Conflicting/Malformed/Distorting.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, help="'i/n' to process shard i of n (for parallel runs).")
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--merge_inputs", default=None,
                   help="Comma-separated shard npys to concatenate into --output_npy (skips generation).")
    args = p.parse_args()

    if args.merge_inputs:
        parts = [s for s in args.merge_inputs.split(",") if s]
        print(f"merging {len(parts)} shards -> {args.output_npy}")
        n = merge_npys(parts, args.output_npy)
        print(f"merged {n} sequences. Done.")
        return

    import torch
    torch.set_num_threads(1)  # per-sketch CPU work is tiny; avoid intra-op oversubscription when run in parallel

    ctx = SketchContext()
    tol = {k: v * args.scale for k, v in DEFAULT_TOLERANCES.items()}
    types = set(args.types) if args.types else set(DETECTABLE_TYPES)
    if args.no_mirror:
        types.discard("Mirror")
    sf = SolverFeedback(ctx) if args.solver_validate else None

    data = flat_array.load_dictionary_flat(args.input_npy)
    seqs = data["sequences"]
    n_total = len(seqs)

    # limit bounds the GLOBAL index range first, then shards stride within it, so
    # the union of shards 0..n-1 equals exactly the first `limit` sketches.
    base = list(range(n_total if args.limit is None else min(args.limit, n_total)))
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        indices = base[si::sn]
    else:
        indices = base

    print(f"input={args.input_npy}  total={n_total}  processing={len(indices)}")
    print(f"scale={args.scale}  types={sorted(types)}  keep_original={args.keep_original}  "
          f"solver_validate={args.solver_validate}")

    valid_indices: list[int] = []
    new_sequences: list = []
    stats = {"total": 0, "valid": 0, "no_rel": 0, "solver_reject": 0, "load_reject": 0, "augment_error": 0}
    n_added_acc = 0
    t_start = time.perf_counter()

    for n_done, idx in enumerate(indices):
        stats["total"] += 1
        try:
            s = datalib.sketch_from_sequence(seqs[idx])
            normalize_sketch(s)
            s = cleanse_sketch_primitives(s)
            if not args.keep_original:
                s.constraints.clear()        # pure relational closure from bare primitives (v1-consistent)
            else:
                s = remove_dimensional_constraints(s)

            n_before = len(s.constraints)
            aug = augment_sketch(s, ctx, tolerances=tol, types=types)
            # lenient clean at generation time so a single bad augmented ref
            # cannot abort the whole sketch
            aug = cleanse_sketch_constraints(aug, delete_invalid=True)
            aug, _ = complete_constraint_values(aug, skip_invalid_constraints=True)

            rel = [c for c in aug.constraints.values() if _is_relational(c)]
            if not rel:
                stats["no_rel"] += 1
                continue

            if sf is not None:
                st = sf.get_solver_status(aug)
                if st.MalformedConstraints or st.ConflictingConstraints or st.DistortingConstraints:
                    stats["solver_reject"] += 1
                    continue
                # Redundant / Converged both accepted.

            new_seq = datalib.sketch_to_sequence(aug)

            # ---- safety gate: must survive the dataset's STRICT preprocessing ----
            check = dataset_preprocess(datalib.sketch_from_sequence(new_seq), relational_only=True)
            if not any(_is_relational(c) for c in check.constraints.values()):
                stats["load_reject"] += 1
                continue
            native = compute_native_response(check, ctx, eps=EPS)
            motion_relational_subspace(check, native.state, ctx, EPS)  # must not raise
            # ----------------------------------------------------------------------

            n_added_acc += len(aug.constraints) - n_before
            valid_indices.append(idx)
            new_sequences.append(new_seq)
            stats["valid"] += 1
        except Exception:
            stats["load_reject"] += 1

        if (n_done + 1) % args.log_every == 0:
            el = time.perf_counter() - t_start
            rate = (n_done + 1) / el
            eta = (len(indices) - n_done - 1) / rate / 60
            print(f"  {n_done+1}/{len(indices)}  valid={stats['valid']}  "
                  f"{rate:.1f}/s  ETA {eta:.1f} min", flush=True)

    el = time.perf_counter() - t_start
    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k:14s} {v:>8}")
    if stats["valid"]:
        print(f"  avg relations added/sketch: {n_added_acc / stats['valid']:.1f}")
    print(f"  wall: {el/60:.1f} min  ({stats['total']/max(el,1e-9):.1f} sketch/s)")

    if not valid_indices:
        print("ERROR: no valid samples; not saving.", file=sys.stderr)
        sys.exit(1)

    print(f"\nPacking {len(valid_indices)} sequences -> {args.output_npy}")
    Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
    out = build_output_npy(data, valid_indices, new_sequences)
    np.save(args.output_npy, out, allow_pickle=False)
    print("Done.")


if __name__ == "__main__":
    main()
