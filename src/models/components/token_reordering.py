import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext


class TokenReordering:
    def __init__(self, context: SketchContext, prefix_len: int = 1):
        self.pad_token = ConstraintToken.Pad
        self.stop_token = ConstraintToken.Stop
        self.block_size = context.align_constraint_token_length_to
        self.prefix_len = prefix_len

        if self.block_size is None:
            raise NotImplementedError("Variable-length constraint generation not yet handled.")

    @torch.no_grad()  # NOTE: query_tokens must not be model prediction results during training!!!
    def __call__(
        self,
        query_tokens: torch.Tensor,
        ref_logits: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        strategy: str,
        prefix_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if prefix_len is None:
            prefix_len = self.prefix_len
        if strategy == 'none':
            return query_tokens, ref_mask
        elif strategy == 'canonical_with_tail_strip':
            return self.canonical_sorting(query_tokens.detach(), replace_tokens_after_first_stop_by_pad=True, prefix_len=prefix_len), ref_mask
        elif strategy == 'canonical_full':
            return self.canonical_sorting(query_tokens.detach(), replace_tokens_after_first_stop_by_pad=False, prefix_len=prefix_len), ref_mask
        elif strategy == 'hungarian':
            if not isinstance(ref_logits, torch.Tensor) or not isinstance(ref_mask, torch.Tensor):
                raise ValueError("'hungarian' strategy requires ref_logits and ref_mask.")
            return self.hungarian_match(query_tokens.detach(), ref_logits.detach(), ref_mask.detach(), prefix_len=prefix_len), ref_mask
        elif strategy == 'hungarian_frontier':
            if not isinstance(ref_logits, torch.Tensor) or not isinstance(ref_mask, torch.Tensor):
                raise ValueError("'hungarian_frontier' strategy requires ref_logits and ref_mask.")
            return self.hungarian_frontier_match(query_tokens.detach(), ref_logits.detach(), ref_mask.detach(), prefix_len=prefix_len)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    ################################################################
    # Canonical Sorting
    ################################################################

    def get_first_stop_idx(self, tokens: torch.Tensor):
        assert len(tokens.shape) == 2, "Tokens shape must be [B, L]."
        is_stop_padded = F.pad(tokens == self.stop_token, (0, 1), value=True)
        first_stop_pos = is_stop_padded.long().argmax(dim=1).reshape(-1, 1)
        return first_stop_pos

    def canonical_sorting(
        self,
        tokens: torch.Tensor,
        replace_tokens_after_first_stop_by_pad: bool,
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        """
        tokens are block-wise sorted in the alpha-numeric order.
            - If replace_tokens_after_first_stop_by_pad=True, all the tokens after the first STOP are replaced by PAD.
            - If False, only the blocks containing STOP or PAD are appended at the tail in the same order.
        """
        if prefix_len is None:
            prefix_len = self.prefix_len
        if tokens.dtype != torch.long:
            raise ValueError(f"tokens.dtype must be long, but got {tokens.dtype}")

        B, L = tokens.shape
        end_idx = prefix_len + ((L - prefix_len) // self.block_size) * self.block_size
        tokens_to_sort = tokens[:, prefix_len:end_idx]

        # Prepare keys for lexicographic sort
        vocab_size = int(tokens.max().item()) + 1
        pack_multiplier = vocab_size ** torch.arange(self.block_size - 1, -1, -1, device=tokens.device)
        blocks_to_sort = tokens_to_sort.reshape(B, -1, self.block_size)
        blocks_to_sort_packed = (blocks_to_sort * pack_multiplier).sum(dim=2)

        # STOP + Invalid tokens want to come last
        if replace_tokens_after_first_stop_by_pad:
            # Fill tokens after the first stop token by torch.long.max
            first_stop_pos = self.get_first_stop_idx(tokens_to_sort)
            invalid_tokens = (first_stop_pos <= torch.arange(tokens_to_sort.shape[1], device=tokens.device).reshape(1, -1))
        else:
            invalid_tokens = (tokens_to_sort == self.stop_token) | (tokens_to_sort == self.pad_token)

        # lexicographic sort
        invalid_blocks = invalid_tokens.reshape(B, -1, self.block_size).any(dim=2)
        blocks_to_sort_packed[invalid_blocks] = torch.iinfo(blocks_to_sort_packed.dtype).max
        sort_indices = blocks_to_sort_packed.sort(dim=1, stable=True).indices
        sorted_blocks = blocks_to_sort.gather(dim=1, index=sort_indices.unsqueeze(-1).expand_as(blocks_to_sort))

        ret = tokens.detach().clone()
        ret[:, prefix_len:end_idx] = sorted_blocks.reshape(B, -1)

        # Replace tokens after first STOP with PAD
        if replace_tokens_after_first_stop_by_pad:
            first_stop_pos = self.get_first_stop_idx(ret)
            after_stop_mask = torch.arange(ret.shape[1], device=ret.device).unsqueeze(0) > first_stop_pos
            ret[after_stop_mask] = self.pad_token

        return ret

    ################################################################
    # Hungarian Matching
    ################################################################

    def cut_and_reshape_to_blocks(self, tokens: torch.Tensor, prefix_len: int | None = None) -> torch.Tensor:
        """Reshape [B, L, ...] to [B, N_blocks, block_size, ...].
        """
        if prefix_len is None:
            prefix_len = self.prefix_len
        B, L = tokens.shape[:2]
        end_idx = prefix_len + ((L - prefix_len) // self.block_size) * self.block_size
        return rearrange(tokens[:, prefix_len : end_idx], "b (n c) ... -> b n c ...", c=self.block_size)

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
            assert (row_ind == np.arange(N)).all(), "linear_sum_assignment output mismatch"
            matched.append(col_ind)

        # Convert to tensor (stack into numpy array first to avoid slow conversion)
        matched_np = np.stack(matched, axis=0)
        return torch.from_numpy(matched_np).to(device=cost_matrix.device, dtype=torch.long)

    def hungarian_match(
        self,
        query_tokens: torch.Tensor,
        ref_logits: torch.Tensor,
        ref_mask: torch.Tensor,
        prefix_len: int | None = None,
    ) -> torch.Tensor:
        if prefix_len is None:
            prefix_len = self.prefix_len
        # Reshape to blocks (skip START token)
        query_blocks = self.cut_and_reshape_to_blocks(query_tokens, prefix_len)  # [B, N, block_size]
        ref_logits_blocks = self.cut_and_reshape_to_blocks(ref_logits, prefix_len)   # [B, N, block_size, V]
        mask_blocks = self.cut_and_reshape_to_blocks(ref_mask, prefix_len)  # [B, N, block_size]
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
        end_pos = prefix_len + N * self.block_size
        full_matched_query = query_tokens.clone()
        full_matched_query[:, prefix_len : end_pos] = matched_query_flat

        return full_matched_query

    ################################################################
    # Frontier Matching
    ################################################################

    @staticmethod
    def _get_frontier_mask(mask_blocks: torch.Tensor) -> torch.Tensor:
        """
        Generates a mask that is True only at the first masked position (Frontier) for each block.
        mask_blocks: [B, N, block_size] Boolean Tensor, assuming the structure [False, ..., False, True, ..., True]
        """
        shifted_blocks = torch.zeros_like(mask_blocks)
        shifted_blocks[..., 1:] = mask_blocks[..., :-1]
        return mask_blocks & (~shifted_blocks)

    def _compute_frontier_cost_matrix(
        self,
        ref_logits_blocks: torch.Tensor,
        query_blocks: torch.Tensor,
        mask_blocks: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the cost matrix based on:
        1. History Consistency (Visible tokens must match)
        2. Frontier Likelihood (NLL of the first masked token)
        """
        B, N, block_size, V = ref_logits_blocks.shape
        device = ref_logits_blocks.device

        # --- 1. Identify Frontier ---
        # Find the index of the first True in the mask.
        # If a row has no mask, argmax returns 0, but we'll mask out the cost later.
        frontier_indices = mask_blocks.int().argmax(dim=2)  # [B, N]
        has_mask = mask_blocks.any(dim=2)  # [B, N]

        # --- 2. Calculate Frontier Soft Cost (NLL) ---
        # We need log_prob( GT_candidate[frontier_pos] | Pred_slot[history] )

        # a) Extract Logits at the frontier position for each prediction slot
        # frontier_indices_expanded: [B, N, 1, V]
        frontier_indices_expanded = frontier_indices.view(B, N, 1, 1).expand(-1, -1, -1, V)
        # Gather logits: [B, N, 1, V] -> [B, N, V]
        frontier_logits = torch.gather(ref_logits_blocks, 2, frontier_indices_expanded).squeeze(2)
        log_probs = F.log_softmax(frontier_logits, dim=-1)  # [B, N, V]

        # b) Extract Target Tokens from GT candidates at the frontier position
        # We need a matrix [B, Row(Pred), Col(GT)] containing the token value of GT[col] at Row's frontier index.

        # gt_candidates: [B, 1, N, 4] -> [B, N, N, 4] (Expand cols)
        gt_candidates = query_blocks.unsqueeze(1).expand(-1, N, -1, -1)

        # gather_indices: [B, N, 1, 1] -> [B, N, N, 1] (Expand cols)
        gather_indices = frontier_indices.view(B, N, 1, 1).expand(-1, -1, N, -1)

        # targets: [B, N, N]
        targets = torch.gather(gt_candidates, 3, gather_indices).squeeze(3)

        # c) Compute NLL Cost
        # cost[b, i, j] = -log_probs[b, i, targets[b, i, j]]
        # Expand log_probs to [B, N, N, V] to gather
        log_probs_expanded = log_probs.unsqueeze(2).expand(-1, -1, N, -1)
        targets_expanded = targets.unsqueeze(3)

        # selected_log_probs: [B, N, N]
        selected_log_probs = torch.gather(log_probs_expanded, 3, targets_expanded).squeeze(3)
        cost_soft = -selected_log_probs

        # --- 3. Calculate History Hard Cost ---
        # Visible tokens must match exactly.

        # query_expanded (Source/Pred): [B, N, 1, 4]
        query_expanded = query_blocks.unsqueeze(2)
        # gt_expanded (Candidate): [B, 1, N, 4]
        gt_expanded = query_blocks.unsqueeze(1)

        # visible_mask: [B, N, 1, 4] (False where masked)
        visible_mask = (~mask_blocks).unsqueeze(2)

        # Mismatch: (Query != GT) AND Visible
        mismatch = (query_expanded != gt_expanded) & visible_mask

        # Any mismatch in the block implies infinite cost
        # history_violation: [B, N, N]
        history_violation = mismatch.any(dim=3)

        # --- 4. Combine ---
        LARGE_COST = 1e6

        # Set the soft costs to zero if the block is fully visible
        # has_mask: [B, N] -> [B, N, 1] -> [B, N, N]
        has_mask_expanded = has_mask.unsqueeze(2).expand(-1, -1, N)
        cost_soft = torch.where(has_mask_expanded, cost_soft, torch.tensor(0.0, device=device))

        # Apply history violation penalty
        cost = cost_soft
        cost[history_violation] = LARGE_COST

        return cost

    def hungarian_frontier_match(
        self,
        query_tokens: torch.Tensor,
        ref_logits: torch.Tensor,
        ref_mask: torch.Tensor,
        prefix_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reorders query_tokens to match ref_logits based on:
        1. History Consistency (Hard Constraint): Visible tokens must match.
        2. Frontier Likelihood (Soft Constraint): Log-prob of the first masked token.

        Returns:
            reordered_tokens: [B, L]
            mask_for_loss: [B, L] (True only at Frontier positions)
        """
        if prefix_len is None:
            prefix_len = self.prefix_len
        # 1. Reshape to blocks
        query_blocks = self.cut_and_reshape_to_blocks(query_tokens, prefix_len)        # [B, N, 4]
        ref_logits_blocks = self.cut_and_reshape_to_blocks(ref_logits, prefix_len)     # [B, N, 4, V]
        mask_blocks = self.cut_and_reshape_to_blocks(ref_mask, prefix_len)             # [B, N, 4]
        B, N, block_size, _ = ref_logits_blocks.shape

        # 2. Compute Cost Matrix (Vectorized)
        cost_matrix = self._compute_frontier_cost_matrix(ref_logits_blocks, query_blocks, mask_blocks)

        # 3. Hungarian Matching
        matched_indices = self._hungarian_match_core(cost_matrix) # [B, N]

        # 4. Reorder Query Blocks
        # matched_query[b, i] = query_blocks[b, matched_indices[b, i]]
        matched_query = torch.zeros_like(query_blocks)
        for b in range(B):
            matched_query[b] = query_blocks[b, matched_indices[b]]

        # 5. Generate Mask for Loss (Frontier only)
        # NOTE: The mask itself corresponds to the Input/Prediction structure,
        # so it DOES NOT need to be reordered. The Input (and its mask) stays fixed.
        frontier_mask_blocks = self._get_frontier_mask(mask_blocks) # [B, N, 4]

        # 6. Flatten back to [B, L]
        matched_query_flat = matched_query.reshape(B, -1)
        frontier_mask_flat = frontier_mask_blocks.reshape(B, -1)

        end_pos = prefix_len + N * block_size

        # Construct full tokens
        full_matched_query = query_tokens.clone()
        full_matched_query[:, prefix_len : end_pos] = matched_query_flat

        # Construct full mask
        # Default to False (ignore loss) for prefix/padding
        full_frontier_mask = torch.zeros_like(ref_mask, dtype=torch.bool)
        full_frontier_mask[:, prefix_len : end_pos] = frontier_mask_flat

        return full_matched_query, full_frontier_mask