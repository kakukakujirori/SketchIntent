"""Token-level constraint-type statistics (relational vs dimensional).

Reward-independent counting shared by ``DimParsimonyReward`` (penalty) and the
intent evaluation metrics (logging).  Keeping the head-of-block counting in one
place means the penalty and the eval report can never disagree on what counts as
a relational / dimensional / valid constraint.

Layout assumption: one optional prefix token (START) followed by ``N`` blocks of
``block_size`` tokens, where each block's head token carries the constraint-type
id.  This mirrors the sequence layout produced by ``SketchTokenizer`` with
``align_constraint_token_length_to == block_size``.
"""
from __future__ import annotations

import torch

from src.sketch_handlers.constants import ConstraintToken


def constraint_type_counts(
    cons_tokens: torch.Tensor,
    block_size: int,
    dim_token_ids: torch.Tensor,
    prefix_len: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample ``(n_relational, n_dimensional, n_total)`` counts.

    A constraint block counts as *valid* when its head token precedes the first
    STOP and is not PAD.  ``n_dimensional`` is the number of valid heads whose id
    is in ``dim_token_ids``, ``n_relational`` is the remaining valid heads, and
    ``n_total`` is every valid head.  All three are ``(B,)`` ``long`` tensors on
    ``cons_tokens.device``.

    Empty sequences (``[START][STOP]`` / no valid block) yield all-zero counts,
    which lets callers surface the collapse failure mode via ``n_total == 0``.
    """
    if cons_tokens.ndim != 2:
        raise ValueError(
            f"expected (B, L) tokens, got shape {tuple(cons_tokens.shape)}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, L = cons_tokens.shape
    N = (L - prefix_len) // block_size
    if N <= 0:
        zeros = torch.zeros(B, dtype=torch.long, device=cons_tokens.device)
        return zeros, zeros.clone(), zeros.clone()

    head_idx = prefix_len + torch.arange(N, device=cons_tokens.device) * block_size
    heads = cons_tokens[:, head_idx]                                   # (B, N)

    dim_ids = dim_token_ids.to(heads.device)
    dim_mask = torch.isin(heads, dim_ids)
    # Positions at/after the first STOP are not real constraints.
    stop_seen = (heads == int(ConstraintToken.Stop)).cumsum(dim=-1) >= 1
    valid_mask = ~stop_seen & (heads != int(ConstraintToken.Pad))

    n_total = valid_mask.sum(dim=-1)
    n_dim = (dim_mask & valid_mask).sum(dim=-1)
    n_rel = n_total - n_dim
    return n_rel.long(), n_dim.long(), n_total.long()
