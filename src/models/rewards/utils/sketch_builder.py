"""Shared helpers for building a predicted sketch (and its native response) from token tensors."""
from __future__ import annotations

import numpy as np
import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    compute_native_response,
    pairwise_distance_observable,
)
from src.models.components.solver_feedback import SolverFeedback
from src.models.rewards.base import RewardContext
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import complete_constraint_values


def build_pred_sketch(
    tokenizer: SketchTokenizer,
    cons_tokens_np: np.ndarray,
    prim_tokens_np: np.ndarray,
    sketch_original: datalib.Sketch | None,
) -> datalib.Sketch:
    """Decode token arrays into a Sketch with original primitive geometry.

    Passing ``reference_sketch=sketch_original`` makes the decoded entities
    adopt GT's entity keys by position, so constraint references in Pred and
    GT resolve to the same physical entity (required for downstream
    response-metric computations).
    """
    pred_sketch, _, _ = tokenizer.detokenize(
        {"p_val": prim_tokens_np, "c_val": cons_tokens_np},
        allow_invalid_constraints=True,
        reference_sketch=sketch_original,
    )
    if sketch_original is not None:
        pred_sketch = SolverFeedback.overwrite_primitives_with_original(
            pred_sketch, sketch_original
        )
    pred_sketch, _ = complete_constraint_values(
        pred_sketch, skip_invalid_constraints=True
    )
    return pred_sketch


def get_or_build_pred_native(
    ctx: RewardContext,
    i: int,
    tokenizer: SketchTokenizer,
    cons_tokens_np: np.ndarray,
    prim_tokens_np: np.ndarray,
    gt_sketch: datalib.Sketch,
    sketch_context: SketchContext,
    eps: float,
) -> tuple[datalib.Sketch, NativeResponse]:
    """Return the cached ``(pred_sketch, pred_native)`` pair for sample ``i``.

    Rewards that need the first-order response map (ResponseIntentReward,
    ProbeResponseSVDReward, ...) share the pred sketch and its native
    response via ``ctx.pred_sketches`` and ``ctx.native_cache``.  The cache
    key for pred lives under ``("pred", i)`` because batch indices are stable
    within one composer pass.  Callers must use a consistent ``eps`` across
    rewards composed in the same context; mixing different eps values would
    silently return the cached NativeResponse built under the first caller's
    eps.
    """
    if ctx.pred_sketches is None:
        ctx.pred_sketches = [None] * int(ctx.cons_tokens.shape[0])

    pred_sketch = ctx.pred_sketches[i]
    if pred_sketch is None:
        pred_sketch = build_pred_sketch(
            tokenizer, cons_tokens_np, prim_tokens_np, gt_sketch
        )
        ctx.pred_sketches[i] = pred_sketch

    pred_key = ("pred", i)
    pred_native = ctx.native_cache.get(pred_key)
    if pred_native is None:
        pred_native = compute_native_response(
            pred_sketch, sketch_context, eps=eps
        )
        ctx.native_cache[pred_key] = pred_native
    return pred_sketch, pred_native


def get_or_compute_j_phi(
    ctx: RewardContext,
    sketch: datalib.Sketch,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the cached pairwise-distance Jacobian ``J_phi`` for ``sketch``.

    ``pairwise_distance_observable`` is invoked twice per (sample, sketch)
    pair via the v4 motion-subspace family — once inside ``motion_full_subspace``
    and once inside ``motion_relational_subspace`` — and additionally K times
    for the same ``gt_sketch`` across an RLOO rollout group.  Caching by
    ``(id(sketch), dtype)`` collapses all of that to a single computation per
    sketch object and dtype.

    The same eps caveat as :func:`get_or_build_pred_native` applies: callers in
    one composer pass must agree on dtype (and any future J_phi parameters)
    since the first caller's tensor is what subsequent ones receive.
    """
    key = (id(sketch), dtype)
    cached = ctx.j_phi_cache.get(key)
    if cached is None:
        cached = pairwise_distance_observable(sketch, dtype=dtype).to(dtype=dtype)
        ctx.j_phi_cache[key] = cached
    return cached


def get_or_compute_gt_native(
    ctx: RewardContext,
    gt_sketch: datalib.Sketch,
    sketch_context: SketchContext,
    eps: float,
) -> NativeResponse:
    """Return the cached GT ``NativeResponse`` keyed by ``id(gt_sketch)``.

    RLOO duplicates the same GT sketch instance across the ``K`` rollouts of
    each prompt, so ``id()`` is a stable in-batch identity that lets every
    rollout in the group share the GT-side native-response computation.
    The same eps caveat as :func:`get_or_build_pred_native` applies.
    """
    gt_key = ("gt", id(gt_sketch))
    cached = ctx.native_cache.get(gt_key)
    if cached is None:
        cached = compute_native_response(gt_sketch, sketch_context, eps=eps)
        ctx.native_cache[gt_key] = cached
    return cached
