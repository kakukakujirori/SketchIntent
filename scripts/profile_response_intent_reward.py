"""Profile ResponseIntentReward.compute() on real sketches from the dataset.

The fixture loads B sketches from sg_filtered_unique_valid.npy, tokenizes each
to (p_val, c_val), and feeds the *GT* constraint tokens back in as the model's
"prediction".  This means pred_sketch decodes successfully every time and the
full Jacobian / SVD pipeline runs end-to-end, which is what we want for
profiling.

Run:
    conda run -n freecad python scripts/profile_response_intent_reward.py
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import torch

from sketchgraphs.data import flat_array, sketch_from_sequence

from src.data.components.primitive_noise import PrimitiveNoise
from src.models.rewards.base import RewardContext
from src.models.rewards.response_metric_reward import ResponseIntentReward
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_primitives,
    cleanse_sketch_constraints,
    complete_constraint_values,
    normalize_sketch,
)


def build_fixture(
    sequence_file: str,
    batch_size: int,
    context: SketchContext,
    seed: int = 0,
) -> RewardContext:
    sequence_array = flat_array.load_dictionary_flat(sequence_file)
    sequences = sequence_array["sequences"]
    tokenizer = SketchTokenizer(context)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(sequences), size=batch_size * 4, replace=False).tolist()

    prim_list: list[np.ndarray] = []
    cons_list: list[np.ndarray] = []
    sketches: list = []

    noise = PrimitiveNoise(std=0.0, max_diff=0.0)

    for idx in indices:
        if len(sketches) >= batch_size:
            break
        try:
            sketch = sketch_from_sequence(sequences[idx])
            normalize_sketch(sketch)
            sketch = cleanse_sketch_primitives(sketch)
            sketch = cleanse_sketch_constraints(sketch)
            sketch, _ = complete_constraint_values(sketch)
            noisy = noise.apply_primitive_noise(sketch)
            normalize_sketch(noisy)
            tok = tokenizer.tokenize(noisy)
        except Exception as e:  # noqa: BLE001
            print(f"skip {idx}: {e}")
            continue
        prim_list.append(tok["p_val"])
        cons_list.append(tok["c_val"])
        sketches.append(sketch)

    if len(sketches) < batch_size:
        raise RuntimeError(
            f"only collected {len(sketches)} valid sketches; bump the candidate pool"
        )

    L_prim = max(len(x) for x in prim_list)
    L_cons = max(len(x) for x in cons_list)
    prim_tokens = np.zeros((batch_size, L_prim), dtype=np.int64)
    cons_tokens = np.zeros((batch_size, L_cons), dtype=np.int64)
    for i, (p, c) in enumerate(zip(prim_list, cons_list)):
        prim_tokens[i, : len(p)] = p
        cons_tokens[i, : len(c)] = c

    ctx = RewardContext(
        prim_tokens=torch.from_numpy(prim_tokens),
        cons_tokens=torch.from_numpy(cons_tokens),
        sketch_original_list=sketches,
        per_sample_solver_status=["FC"] * batch_size,
    )
    return ctx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sequence-file",
        default="data_processed/sg_filtered_unique_valid.npy",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    torch.set_num_threads(1)  # match training (one worker usually)

    context = SketchContext(
        max_token_length=192,
        num_quantization_bins=64,
        align_constraint_token_length_to=4,
        use_stop_padding=False,
        constraint_order="random_shuffle",
    )
    reward = ResponseIntentReward(
        context=context,
        eps=1.0e-9,
        coeff_h_angle=3.0,
        cap_h_angle=0.70,
        h_angle_reward_metric="motion_relational_angle",
        gate="converged",
    )

    print(f"Building fixture (B={args.batch_size})...")
    ctx_template = build_fixture(args.sequence_file, args.batch_size, context)
    print("  done.")

    # Reset cache each iteration (training calls compute() once per batch with
    # a fresh RewardContext) so the cost we measure includes native_response
    # computation.
    def _fresh_ctx() -> RewardContext:
        return RewardContext(
            prim_tokens=ctx_template.prim_tokens,
            cons_tokens=ctx_template.cons_tokens,
            sketch_original_list=ctx_template.sketch_original_list,
            per_sample_solver_status=ctx_template.per_sample_solver_status,
        )

    # Warmup
    for _ in range(args.warmup):
        reward.compute(_fresh_ctx())

    # Wall-clock timing
    t0 = time.perf_counter()
    for _ in range(args.iters):
        reward.compute(_fresh_ctx())
    elapsed = time.perf_counter() - t0
    per_call = elapsed / args.iters
    per_sample = per_call / args.batch_size
    print(
        f"\nwall-clock: {per_call*1000:.1f} ms/batch  ({per_sample*1000:.1f} ms/sample"
        f", B={args.batch_size}, iters={args.iters})"
    )

    # cProfile a single batch
    profiler = cProfile.Profile()
    profiler.enable()
    reward.compute(_fresh_ctx())
    profiler.disable()

    print(f"\n=== top {args.top} by cumtime ===")
    s = StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(args.top)
    print(s.getvalue())

    print(f"\n=== top {args.top} by tottime ===")
    s2 = StringIO()
    pstats.Stats(profiler, stream=s2).sort_stats("tottime").print_stats(args.top)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
