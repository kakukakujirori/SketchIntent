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


def constraint_head_counts(
    cons_tokens: torch.Tensor,
    block_size: int,
    type_token_ids: torch.Tensor,
    prefix_len: int = 1,
) -> torch.Tensor:
    """Return per-sample, per-type counts of valid constraint heads.

    ``type_token_ids`` is a 1-D tensor of the constraint-type head ids to count
    (e.g. ``CONSTRAINT_TYPE_TOKEN_IDS``).  The returned tensor has shape
    ``(B, len(type_token_ids))`` and column ``j`` holds the number of valid heads
    equal to ``type_token_ids[j]`` for each sample.

    "Valid" uses the same definition as :func:`constraint_type_counts`: a head
    precedes the first STOP and is not PAD.  Ids in ``type_token_ids`` that never
    appear (and any heads not in ``type_token_ids``) simply contribute zero, so
    the per-type counts always sum to at most ``n_total`` from
    ``constraint_type_counts``.  This shared head-of-block counting keeps the
    relational class-prior reward and the constraint-mix logging from ever
    disagreeing on what counts as a given constraint type.
    """
    if cons_tokens.ndim != 2:
        raise ValueError(
            f"expected (B, L) tokens, got shape {tuple(cons_tokens.shape)}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    B, L = cons_tokens.shape
    n_types = int(type_token_ids.numel())
    N = (L - prefix_len) // block_size
    if N <= 0 or n_types == 0:
        return torch.zeros(B, n_types, dtype=torch.long, device=cons_tokens.device)

    head_idx = prefix_len + torch.arange(N, device=cons_tokens.device) * block_size
    heads = cons_tokens[:, head_idx]                                   # (B, N)

    stop_seen = (heads == int(ConstraintToken.Stop)).cumsum(dim=-1) >= 1
    valid_mask = ~stop_seen & (heads != int(ConstraintToken.Pad))      # (B, N)

    type_ids = type_token_ids.to(device=heads.device, dtype=heads.dtype)
    # (B, N, n_types): head equals each requested type id, gated by validity.
    eq = heads.unsqueeze(-1) == type_ids.view(1, 1, -1)
    counts = (eq & valid_mask.unsqueeze(-1)).sum(dim=1)                # (B, n_types)
    return counts.long()
