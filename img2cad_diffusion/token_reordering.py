import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment


class TokenReordering:
    def __init__(self, token_num_per_block: int = 1, token_prefix_len: int = 0):
        self.block_size = token_num_per_block
        self.prefix_len = token_prefix_len

    def __call__(self, query_tokens: torch.Tensor, ref_logits: torch.Tensor | None, ref_mask: torch.Tensor | None, strategy: str) -> torch.Tensor:
        if strategy == 'none':
            return query_tokens
        elif strategy == 'alphabetical':
            return self.alphabetical_sorting(query_tokens.detach())
        elif strategy == 'hungarian':
            if not isinstance(ref_logits, torch.Tensor) or not isinstance(ref_mask, torch.Tensor):
                raise ValueError("'hungarian' strategy requires ref_logits and ref_mask.")
            return self.hungarian_match(query_tokens.detach(), ref_logits.detach(), ref_mask.detach())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def alphabetical_sorting(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def hungarian_match(self, query_tokens: torch.Tensor, ref_logits: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        # Reshape to blocks (skip START token)
        query_blocks = self.cut_and_reshape_to_blocks(query_tokens)  # [B, N, block_size]
        ref_logits_blocks = self.cut_and_reshape_to_blocks(ref_logits)   # [B, N, block_size, V]
        mask_blocks = self.cut_and_reshape_to_blocks(ref_mask)  # [B, N, block_size]
        B, N, _, V = ref_logits_blocks.shape

        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(ref_logits_blocks, query_blocks, mask_blocks)

        # Hungarian matching
        matched_indices = self._hungarian_match_core(cost_matrix)  # [B, N]

        # Reorder query blocks according to matching
        # matched_query[b, i] = query_blocks[b, matched_indices[b, i]]
        matched_query = torch.zeros_like(query_blocks)
        for b in range(B):
            matched_query[b] = query_blocks[b, matched_indices[b]]

        # Flatten back: [B, N, block_size] -> [B, N * block_size]
        matched_query_flat = matched_query.reshape(B, -1)
        mask_flat = mask_blocks.reshape(B, -1)

        # Reconstruct full sequence
        end_pos = self.prefix_len + N * self.block_size
        full_matched_query = query_tokens.clone()
        full_matched_query[:, self.prefix_len : end_pos] = matched_query_flat

        return full_matched_query

    def cut_and_reshape_to_blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape [B, L, ...] to [B, N_blocks, block_size, ...].
        """
        B, L = tokens.shape[:2]
        end_idx = self.prefix_len + ((L - self.prefix_len) // self.block_size) * self.block_size
        return rearrange(tokens[:, self.prefix_len : end_idx], "b (n c) ... -> b n c ...", c=self.block_size)

    def _compute_cost_matrix(
        self,
        pred_logits: torch.Tensor,
        gt_blocks: torch.Tensor,
        mask_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cost matrix for Hungarian matching (vectorized).

        Args:
            pred_logits: [B, N, 4, V] predicted logits for each block
            gt_blocks: [B, N, 4] ground truth token values
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
        one_hot_input = F.one_hot(gt_blocks.long(), num_classes=V).float()

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

    def _hungarian_match_core(
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
