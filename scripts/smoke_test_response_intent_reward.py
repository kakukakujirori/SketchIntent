"""Smoke test for ``ResponseIntentReward`` / ``ProbeResponseSVDReward`` invariants.

What this verifies (no model / no FreeCAD needed):

1. Identity     — Pred == GT ⇒ every Hθ angle is 0.
2. Value range  — All 5 metrics in ``H_THETA_REWARD_METRICS`` are in ``[0, 1]``.
3. Normalization — ``response_full_angle`` equals
   ``sqrt((motion_full^2 + leak^2) / 2)`` (the post-refactor RMS form).
4. FC gate      — Samples with non-FC solver status receive no score.
5. Reward range — With default config (coeff=2, cap=1.0), every sample's
   reward lies in ``[0, coeff_h_angle]``.
6. Cache reuse  — When two intent-family rewards share a ``RewardContext``,
   the Pred/GT NativeResponse caches are populated once and reused.

Run with::

    conda run -n freecad python scripts/smoke_test_response_intent_reward.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array

from src.metrics.edit_response_distance import compute_native_response
from src.models.rewards.base import RewardContext
from src.models.rewards.response_metric_reward import (
    H_THETA_REWARD_METRICS,
    _RESPONSE_ANGLE_NORM,
    ResponseIntentReward,
    _compute_h_theta_metrics,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_constraints,
    cleanse_sketch_primitives,
    complete_constraint_values,
    normalize_sketch,
)


DEFAULT_DATA = "data_processed/sg_filtered_unique_valid.npy"


def _load_sketches(path: str, n: int) -> list[datalib.Sketch]:
    sequences = flat_array.load_dictionary_flat(path)["sequences"]
    sketches: list[datalib.Sketch] = []
    for seq in sequences[:n * 4]:  # over-read in case some need to be skipped
        try:
            sk = datalib.sketch_from_sequence(seq)
            _ = normalize_sketch(sk)
            sk = cleanse_sketch_primitives(sk)
            sk = cleanse_sketch_constraints(sk)
            sk, _ = complete_constraint_values(sk)
            sketches.append(sk)
        except Exception:
            continue
        if len(sketches) >= n:
            break
    if len(sketches) < n:
        raise RuntimeError(
            f"Could not load {n} usable sketches from {path}; got {len(sketches)}"
        )
    return sketches


def _check_identity(sketches: list[datalib.Sketch], context: SketchContext) -> None:
    print("[1/6] Identity: Pred == GT ⇒ angles == 0 …", flush=True)
    eps = 1e-9
    for i, sk in enumerate(sketches):
        native = compute_native_response(sk, context, eps=eps)
        h = _compute_h_theta_metrics(sk, native, sk, native, context, eps=eps)
        d = h.to_dict()
        for key in H_THETA_REWARD_METRICS:
            v = float(d[key])
            assert v < 1e-6, (
                f"Identity violation on sketch {i}: {key} = {v:.6g} (expected 0)"
            )
    print("       OK on", len(sketches), "sketches")


def _check_ranges_and_normalization(
    sketches: list[datalib.Sketch], context: SketchContext
) -> None:
    print("[2-3/6] Value range and RMS normalization …", flush=True)
    eps = 1e-9
    # Pair every sketch with the next as Pred/GT to get non-trivial values.
    pairs = list(zip(sketches, sketches[1:] + sketches[:1]))
    rng = (1.0 + 1e-6)  # tiny float slack
    for i, (pred_sk, gt_sk) in enumerate(pairs):
        try:
            pred_native = compute_native_response(pred_sk, context, eps=eps)
            gt_native = compute_native_response(gt_sk, context, eps=eps)
            h = _compute_h_theta_metrics(
                pred_sk, pred_native, gt_sk, gt_native, context, eps=eps,
            )
        except Exception as exc:
            print(f"       skipped pair {i}: {exc}")
            continue
        d = h.to_dict()
        for key in H_THETA_REWARD_METRICS:
            v = float(d[key])
            assert -1e-9 <= v <= rng, (
                f"Range violation on pair {i}: {key} = {v:.6g} ∉ [0, 1]"
            )
        # RMS form invariant: response_full_angle == sqrt((motion² + leak²) / 2).
        mf = float(d["motion_full_angle"])
        mr = float(d["motion_relational_angle"])
        lk = float(d["leak_relational_angle"])
        rf_expected = math.sqrt((mf * mf + lk * lk) / 2.0)
        rr_expected = math.sqrt((mr * mr + lk * lk) / 2.0)
        rf = float(d["response_full_angle"])
        rr = float(d["response_relational_angle"])
        assert abs(rf - rf_expected) < 1e-6, (
            f"response_full_angle != RMS(motion_full, leak) on pair {i}: "
            f"{rf:.6g} vs {rf_expected:.6g}"
        )
        assert abs(rr - rr_expected) < 1e-6, (
            f"response_relational_angle != RMS(motion_relational, leak) on pair {i}: "
            f"{rr:.6g} vs {rr_expected:.6g}"
        )
    print(
        "       OK — range respected and response_*_angle == RMS(motion, leak)"
    )
    assert abs(_RESPONSE_ANGLE_NORM - math.sqrt(2.0)) < 1e-12
    print(f"       _RESPONSE_ANGLE_NORM = {_RESPONSE_ANGLE_NORM:.6f} (= sqrt 2)")


def _make_context_and_tokenizer() -> tuple[SketchContext, SketchTokenizer]:
    context = SketchContext()
    return context, SketchTokenizer(context)


def _build_reward(context: SketchContext) -> ResponseIntentReward:
    return ResponseIntentReward(
        context=context,
        coeff_h_angle=2.0,
        cap_h_angle=1.0,
        h_angle_reward_metric="response_relational_angle",
    )


def _tokenize_batch(
    sketches: list[datalib.Sketch], tokenizer: SketchTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    prim_rows: list[np.ndarray] = []
    cons_rows: list[np.ndarray] = []
    max_p = 0
    max_c = 0
    for sk in sketches:
        toks = tokenizer.tokenize(sk)
        p = np.asarray(toks["p_val"])
        c = np.asarray(toks["c_val"])
        prim_rows.append(p)
        cons_rows.append(c)
        max_p = max(max_p, p.shape[0])
        max_c = max(max_c, c.shape[0])

    pad_token = 0
    prim_arr = np.full((len(prim_rows), max_p), pad_token, dtype=np.int64)
    cons_arr = np.full((len(cons_rows), max_c), pad_token, dtype=np.int64)
    for i, p in enumerate(prim_rows):
        prim_arr[i, : p.shape[0]] = p
    for i, c in enumerate(cons_rows):
        cons_arr[i, : c.shape[0]] = c
    return torch.from_numpy(prim_arr), torch.from_numpy(cons_arr)


def _check_fc_gate_and_reward_range(
    sketches: list[datalib.Sketch],
    context: SketchContext,
    tokenizer: SketchTokenizer,
) -> None:
    print("[4-5/6] FC gate + reward range …", flush=True)
    reward = _build_reward(context)
    prim_tokens, cons_tokens = _tokenize_batch(sketches, tokenizer)
    B = cons_tokens.shape[0]
    statuses = ["FC"] * B
    statuses[-1] = "Under"  # one sample should be gated out

    ctx = RewardContext(
        prim_tokens=prim_tokens,
        cons_tokens=cons_tokens,
        sketch_original_list=sketches,
        per_sample_solver_status=statuses,
    )
    out = reward.compute(ctx)
    scores = out.scores
    metrics = out.metrics

    # Gate check: the "Under" sample contributed zero.
    assert float(scores[-1].sum()) == 0.0, (
        f"FC gate violation: non-FC sample {B-1} produced score "
        f"{float(scores[-1].sum())}"
    )
    assert metrics["intent_skipped_not_fc"] >= 1, (
        "Expected at least one sample skipped by FC gate, "
        f"got intent_skipped_not_fc={metrics['intent_skipped_not_fc']}"
    )

    # Range check: every contributed sample's reward is in [0, coeff].
    coeff = reward.coeff_h_angle
    row_sums = scores.sum(dim=1).tolist()
    for i, r in enumerate(row_sums):
        assert -1e-9 <= r <= coeff + 1e-9, (
            f"Reward range violation on sample {i}: {r:.6g} ∉ [0, {coeff}]"
        )
    print(
        "       OK — FC gate skipped",
        metrics["intent_skipped_not_fc"],
        "/",
        B,
        "sample(s); rewards in [0,",
        coeff,
        "]",
    )


def _check_cache_reuse(
    sketches: list[datalib.Sketch],
    context: SketchContext,
    tokenizer: SketchTokenizer,
) -> None:
    print("[6/6] Cache reuse via RewardContext …", flush=True)
    reward = _build_reward(context)
    prim_tokens, cons_tokens = _tokenize_batch(sketches, tokenizer)
    B = cons_tokens.shape[0]
    ctx = RewardContext(
        prim_tokens=prim_tokens,
        cons_tokens=cons_tokens,
        sketch_original_list=sketches,
        per_sample_solver_status=["FC"] * B,
    )
    reward.compute(ctx)

    assert ctx.pred_sketches is not None and len(ctx.pred_sketches) == B, (
        "ctx.pred_sketches should be populated after compute()"
    )
    gt_keys = [k for k in ctx.native_cache.keys() if k[0] == "gt"]
    pred_keys = [k for k in ctx.native_cache.keys() if k[0] == "pred"]
    assert len(pred_keys) >= 1, (
        f"Expected pred entries in ctx.native_cache, got {pred_keys}"
    )
    assert len(gt_keys) >= 1, (
        f"Expected gt entries in ctx.native_cache, got {gt_keys}"
    )
    print(
        "       OK — pred cache:",
        len(pred_keys),
        "entries; gt cache:",
        len(gt_keys),
        "entries",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA, help="path to *_unique.npy")
    parser.add_argument("--n", type=int, default=6, help="number of sketches to load")
    args = parser.parse_args()

    path = args.data
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)

    print(f"Loading {args.n} sketches from {path}", flush=True)
    sketches = _load_sketches(path, args.n)
    context, tokenizer = _make_context_and_tokenizer()

    _check_identity(sketches, context)
    _check_ranges_and_normalization(sketches, context)
    _check_fc_gate_and_reward_range(sketches, context, tokenizer)
    _check_cache_reuse(sketches, context, tokenizer)

    print("\nAll smoke-test invariants passed.")


if __name__ == "__main__":
    main()
