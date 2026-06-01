"""Token-space flip augmentation for sketches (research plan §16.9).

This module owns reflections that act on *tokenized* primitive sequences — the
output of ``SketchTokenizer``, not the tokenization itself.  It is the token-space
counterpart of ``sketch_utils.reflect_sketch`` (which reflects a ``datalib.Sketch``
object): use ``reflect_sketch`` when you hold geometry, use the functions here when
you hold ``p_val`` / ``p_attr`` token tensors (e.g. inside the RLOO rollout loop).

Keeping this separate from ``sketch_tokenizer`` keeps the tokenizer's responsibility
to (de)tokenization and keeps its module free of a ``torch`` dependency.
"""
from __future__ import annotations

import torch

from .constants import PRIMITIVE_VAL_OFFSET, PrimitiveAttrToken


_X_COORD_ATTR_IDS = (
    int(PrimitiveAttrToken.StartX), int(PrimitiveAttrToken.MidX),
    int(PrimitiveAttrToken.EndX), int(PrimitiveAttrToken.CenterX),
)
_Y_COORD_ATTR_IDS = (
    int(PrimitiveAttrToken.StartY), int(PrimitiveAttrToken.MidY),
    int(PrimitiveAttrToken.EndY), int(PrimitiveAttrToken.CenterY),
)


def reflect_primitive_value_tokens(
    p_val: torch.Tensor, p_attr: torch.Tensor, axis: str, num_bins: int
) -> torch.Tensor:
    """Reflect a tokenized primitive sequence about a coordinate axis, in token space.

    Coordinate value tokens are ``PRIMITIVE_VAL_OFFSET + q`` with ``q`` in
    ``[0, num_bins)`` quantizing ``[-0.5, 0.5]`` at bin centers ``(q+0.5)/num_bins
    - 0.5``.  The reflection ``v -> -v`` maps bin ``q -> num_bins-1-q`` exactly
    (the grid is symmetric about 0), i.e. ``token -> 2*OFFSET + num_bins-1 -
    token``.  Only the requested axis's coordinate tokens are flipped — selected
    by ``p_attr`` (StartX/MidX/EndX/CenterX for ``axis="x"``); Radius / Type /
    Construction / Pad and the other axis are untouched.

    This is the token-space twin of ``sketch_utils.reflect_sketch`` and is exact
    for Arcs too, because the tokenizer point-encodes arcs (Start/Mid/End/Center),
    so reflecting those point coordinates reflects the arc with no frame/clockwise
    handling.  It operates on the (possibly noise-perturbed) tokens directly, so
    the reflected view preserves the same primitive noise as the original — which
    a re-tokenize-from-clean-sketch path would drop.

    Args:
        p_val: Primitive value tokens, any shape (``[..., L_p]``).
        p_attr: Matching attribute tokens, same shape as ``p_val``.
        axis: ``"x"`` or ``"y"`` — the coordinate axis to negate.
        num_bins: Quantization bin count (``context.num_quantization_bins``).
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    attr_ids = _X_COORD_ATTR_IDS if axis == "x" else _Y_COORD_ATTR_IDS
    mask = torch.isin(p_attr, torch.tensor(attr_ids, device=p_attr.device))
    flipped = (2 * PRIMITIVE_VAL_OFFSET + num_bins - 1) - p_val
    return torch.where(mask, flipped, p_val)


def flip_view_schedule(K: int, axes: list[str]) -> list[str]:
    """Assign each of a prompt's ``K`` rollout slots to a view.

    Splits ``K`` as evenly as possible across ``[identity, *axes]`` (the identity
    view and any earlier axes absorb the remainder), returning a length-``K`` list
    of view labels (``"identity"`` / ``"x"`` / ``"y"``).  Single source of truth
    for how the RLOO group's samples are budgeted across flip views.
    """
    views = ["identity"] + list(axes)
    base, rem = divmod(K, len(views))
    slot_view: list[str] = []
    for vi, v in enumerate(views):
        slot_view += [v] * (base + (1 if vi < rem else 0))
    return slot_view


def reflected_expanded_pval(
    p_val: torch.Tensor,
    p_attr: torch.Tensor,
    K: int,
    axes: list[str],
    num_bins: int,
) -> torch.Tensor:
    """Per-sample reflected primitive value tokens for flip-augmented rollouts.

    ``p_val`` / ``p_attr`` are the K-expanded ``[B*K, L_p]`` token tensors in
    prompt-major order (prompt ``p``'s ``K`` rollouts contiguous).  Returns a
    clone of ``p_val`` with each prompt's flip-view slots (per
    ``flip_view_schedule``) reflected in token space; identity slots are
    untouched.  ``p_attr`` / ``p_pos`` / ``p_cnt`` are view-invariant (reflection
    only negates coordinate VALUES) so callers reuse them as-is.  The reward path
    keeps the ORIGINAL (un-reflected) tokens — the coordinate-free constraints are
    scored on the un-reflected sketch.
    """
    slot_view = flip_view_schedule(K, axes)
    out = p_val.clone()
    B = p_val.shape[0] // K
    prompt_off = torch.arange(B, device=p_val.device).unsqueeze(1) * K  # [B, 1]
    for axis in axes:
        slots = [k for k, v in enumerate(slot_view) if v == axis]
        if not slots:
            continue
        rows = (prompt_off + torch.tensor(slots, device=p_val.device)).reshape(-1)
        out[rows] = reflect_primitive_value_tokens(
            out[rows], p_attr[rows], axis, num_bins
        )
    return out
