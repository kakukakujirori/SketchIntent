"""Loss functions for constraint models.

This module provides loss computation including Hungarian matching for
order-agnostic training.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from scipy.optimize import linear_sum_assignment

from .tokens import ConstraintToken
from .tokenization import CONSTRAINT_TOKEN_LEN


class ConstraintIoU(torchmetrics.Metric):
    """Accumulates constraint set IoU statistics across batches.

    Tracks:
    - Total intersection count
    - Total prediction count
    - Total target count
    - Total union count
    - Number of samples (for averaging precision/recall per sample)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_union", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred_tokens: [B, L] predicted constraint tokens
            target_tokens: [B, L] ground truth constraint tokens
        """
        B = pred_tokens.shape[0]

        for b in range(B):
            pred_set = _tokens_to_constraint_set(pred_tokens[b])
            target_set = _tokens_to_constraint_set(target_tokens[b])

            intersection = pred_set & target_set
            union = pred_set | target_set

            self.total_intersection += len(intersection)
            self.total_pred += len(pred_set)
            self.total_target += len(target_set)
            self.total_union += len(union)
            self.num_samples += 1

    def compute(self) -> torch.Tensor:
        """Compute IoU.

        Returns:
            IoU as a tensor
        """
        # Avoid division by zero
        iou = self.total_intersection / self.total_union.clamp(min=1)
        return iou


def _tokens_to_constraint_set(tokens: torch.Tensor) -> set:
    """Convert token sequence to set of constraint tuples.

    Detects constraint boundaries by looking for constraint type tokens
    (Coincident through Vertical), rather than assuming fixed block sizes.

    Args:
        tokens: [L] token sequence (single example)

    Returns:
        Set of (constraint_type, (sorted_refs...)) tuples
    """
    if len(tokens.shape) != 1:
        raise ValueError(f"Expected 1D tensor, got {tokens.shape}")
    tokens = tokens.cpu().numpy()
    constraints = set()

    current_type = None
    current_refs = []

    for tok in tokens:
        tok = int(tok)

        if tok == ConstraintToken.Pad:
            continue
        if tok == ConstraintToken.Mask:
            raise ValueError(f"Mask token found in the evaluation token.")
        if tok == ConstraintToken.Stop:
            break
        if tok == ConstraintToken.InnerPad:
            continue

        if tok < len(ConstraintToken):
            assert tok != ConstraintToken.InnerPad
            # Save previous constraint if exists
            if current_type is not None:
                constraint_tuple = (current_type, tuple(sorted(current_refs)))
                constraints.add(constraint_tuple)
            # Start new constraint
            current_type = tok
            current_refs = []
        else:
            # This is a reference token
            ref_idx = tok - len(ConstraintToken)
            current_refs.append(ref_idx)

    # Don't forget the last constraint
    if current_type is not None:
        constraint_tuple = (current_type, tuple(sorted(current_refs)))
        constraints.add(constraint_tuple)

    return constraints


class ConstraintLoss:
    """Loss computation for constraint generation with optional Hungarian matching.

    When use_hungarian=False: Standard position-wise cross-entropy loss.
    When use_hungarian=True: Hungarian matching to align predicted constraint blocks
    with GT blocks before computing loss. This enables order-agnostic training.
    """

    def __init__(self, use_hungarian: bool = False):
        """
        Args:
            use_hungarian: If True, apply Hungarian matching before loss computation.
        """
        self.use_hungarian = use_hungarian
        self.block_size = CONSTRAINT_TOKEN_LEN  # 4 tokens per constraint block

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask_indices: torch.Tensor,
        reweighting: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute constraint loss.

        Args:
            logits: Model output logits [B, L, V]
            target: Ground truth tokens [B, L]
            mask_indices: Boolean mask indicating masked positions [B, L]
            reweighting: Per-position loss weights [B, L] (for diffusion)

        Returns:
            Scalar loss value
        """
        if self.use_hungarian:
            return self._hungarian_loss(logits, target, mask_indices, reweighting)
        else:
            return self._vanilla_loss(logits, target, mask_indices, reweighting)

    def _vanilla_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask_indices: torch.Tensor,
        reweighting: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard position-wise cross-entropy loss on masked positions."""
        loss = F.cross_entropy(
            logits[mask_indices],
            target[mask_indices],
            reduction='none'
        )
        if reweighting is not None:
            loss = loss * reweighting[mask_indices]
        return loss.sum() / (target.shape[0] * target.shape[1])

    def _reshape_to_blocks(
        self,
        tensor: torch.Tensor,
        is_logits: bool = False
    ) -> torch.Tensor:
        """Reshape [B, L, ...] to [B, N_blocks, block_size, ...].

        We skip the first token (START) and reshape the rest into blocks.
        """
        B = tensor.shape[0]
        # Skip START token (position 0) - this applies to all tensor types
        tensor = tensor[:, 1:]
        L = tensor.shape[1]
        n_blocks = L // self.block_size

        if is_logits:
            # [B, L, V] -> [B, N, 4, V]
            V = tensor.shape[2]
            return tensor[:, :n_blocks * self.block_size].reshape(B, n_blocks, self.block_size, V)
        else:
            # [B, L] -> [B, N, 4]
            return tensor[:, :n_blocks * self.block_size].reshape(B, n_blocks, self.block_size)

    def _compute_cost_matrix(
        self,
        pred_logits: torch.Tensor,
        gt_blocks: torch.Tensor,
        input_blocks: torch.Tensor,
        mask_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost matrix for Hungarian matching (vectorized).

        Args:
            pred_logits: [B, N, 4, V] predicted logits for each block
            gt_blocks: [B, N, 4] ground truth token values
            input_blocks: [B, N, 4] input token values (unmasked positions have GT)
            mask_blocks: [B, N, 4] boolean mask (True = masked position)

        Returns:
            cost: [B, N, N] where cost[b, i, j] = cost of matching pred_block_i to gt_block_j
        """
        B, N, block_size, V = pred_logits.shape
        device = pred_logits.device

        # Compute softmax probabilities for all positions
        # pred_probs: [B, N, 4, V]
        pred_probs = F.softmax(pred_logits, dim=-1)

        # For unmasked positions, replace with one-hot
        # Create one-hot for input tokens: [B, N, 4, V]
        one_hot_input = F.one_hot(input_blocks.long(), num_classes=V).float()

        # Expand mask to [B, N, 4, V] for broadcasting
        mask_expanded = mask_blocks.unsqueeze(-1).expand(-1, -1, -1, V)

        # Combine: use pred_probs where masked, one_hot_input where unmasked
        combined_probs = torch.where(mask_expanded, pred_probs, one_hot_input)
        # [B, N, 4, V]

        # For cost[b, i, j] = -sum_k log(combined_probs[b, i, k, gt_blocks[b, j, k]])
        # We need to gather probabilities at GT token positions

        # Expand gt_blocks for gathering: [B, 1, N, 4] -> [B, N_pred, N_gt, 4]
        gt_expanded = gt_blocks.unsqueeze(1).expand(B, N, N, block_size)

        # Expand combined_probs: [B, N, 4, V] -> [B, N, 1, 4, V]
        probs_expanded = combined_probs.unsqueeze(2)
        # -> [B, N_pred, N_gt, 4, V] using expand
        probs_expanded = probs_expanded.expand(B, N, N, block_size, V)

        # Gather probabilities at GT positions
        # gt_expanded: [B, N, N, 4] -> [B, N, N, 4, 1] for gather
        gt_for_gather = gt_expanded.unsqueeze(-1).long()
        # probs_at_gt: [B, N, N, 4, 1] -> [B, N, N, 4]
        probs_at_gt = torch.gather(probs_expanded, dim=-1, index=gt_for_gather).squeeze(-1)

        # Compute negative log probability and sum over block positions
        # cost: [B, N, N]
        log_probs = torch.log(probs_at_gt.clamp(min=1e-10))
        cost = -log_probs.sum(dim=-1)

        return cost

    def _hungarian_match(
        self,
        cost_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Hungarian matching to find optimal assignment.

        Args:
            cost_matrix: [B, N, N] cost matrix

        Returns:
            matched_indices: [B, N] where matched_indices[b, i] = j means
                            pred_block_i should match to gt_block_j
        """
        B, N, _ = cost_matrix.shape
        matched = []

        for b in range(B):
            cost_np = cost_matrix[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            # row_ind should be [0, 1, 2, ..., N-1], col_ind gives the matching
            matched.append(col_ind)

        # Convert to tensor (stack into numpy array first to avoid slow conversion)
        matched_np = np.stack(matched, axis=0)
        return torch.from_numpy(matched_np).to(device=cost_matrix.device, dtype=torch.long)

    def _hungarian_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        mask_indices: torch.Tensor,
        reweighting: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hungarian matching loss: match blocks optimally, then compute CE."""
        B, L, V = logits.shape
        device = logits.device

        # Reshape to blocks (skip START token)
        pred_logits = self._reshape_to_blocks(logits, is_logits=True)  # [B, N, 4, V]
        gt_blocks = self._reshape_to_blocks(target, is_logits=False)   # [B, N, 4]
        mask_blocks = self._reshape_to_blocks(mask_indices, is_logits=False)  # [B, N, 4]

        # Input blocks = where mask is False, the original GT is visible
        input_blocks = gt_blocks.clone()  # Input = GT at unmasked positions

        N = pred_logits.shape[1]
        block_tokens = N * self.block_size  # How many tokens are covered by blocks

        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(pred_logits, gt_blocks, input_blocks, mask_blocks)

        # Hungarian matching
        matched_indices = self._hungarian_match(cost_matrix)  # [B, N]

        # Reorder GT blocks according to matching
        # matched_gt[b, i] = gt_blocks[b, matched_indices[b, i]]
        matched_gt = torch.zeros_like(gt_blocks)
        for b in range(B):
            matched_gt[b] = gt_blocks[b, matched_indices[b]]

        # Flatten back: [B, N, 4] -> [B, N*4]
        matched_gt_flat = matched_gt.reshape(B, -1)
        mask_flat = mask_blocks.reshape(B, -1)

        # Reconstruct full sequence:
        # - Position 0: START token (from original target)
        # - Positions 1 to 1+block_tokens: matched GT blocks
        # - Remaining positions: original target (trailing tokens not in blocks)
        full_matched_gt = target.clone()
        full_mask = mask_indices.clone()

        # Replace block positions with matched GT
        end_pos = min(1 + block_tokens, L)
        full_matched_gt[:, 1:end_pos] = matched_gt_flat[:, :end_pos-1]
        full_mask[:, 1:end_pos] = mask_flat[:, :end_pos-1]

        # Compute loss on masked positions
        loss = F.cross_entropy(
            logits[full_mask],
            full_matched_gt[full_mask],
            reduction='none'
        )

        if reweighting is not None:
            loss = loss * reweighting[full_mask]

        return loss.sum() / (B * L)

