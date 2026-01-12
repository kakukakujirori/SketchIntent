from __future__ import annotations

import copy
import dataclasses
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from img2cad_diffusion.constraint_data import ConstraintTokenInput
from img2cad_diffusion.tokens import ConstraintToken, PrimitiveToken
from img2cad_diffusion.tokenization import CONSTRAINT_TOKEN_LEN

if TYPE_CHECKING:
    from img2cad_diffusion.constraint_models import ConstraintMDDModule, LogLinearNoise


def _sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    r"""Sample from a categorical distribution using the Gumbel-max trick.
    Args:
        categorical_probs: A tensor of shape `(..., num_classes)` representing
            categorical probabilities.
    Returns:
        A tensor of shape `(...)` representing sampled class indices.
    """
    categorical_probs = categorical_probs.to(torch.float64)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


@dataclasses.dataclass
class SamplingConfig:
    steps: int = 64
    method: str = 'tau_leaping' # 'topk_static', 'tau_leaping'
    N_tokenwise_unmasking: int = CONSTRAINT_TOKEN_LEN
    N_token_prefix: int = 1  # Start token
    noise_removal: bool = True
    sample_batch_size: int = 16 # Not strictly used if batch is provided
    # PRISM specific params
    num_initial_masks: int = 0
    loop_step_on: int = 1000
    loop_steps: int = 0
    num_remask_loop: int = 0
    nucleus_p: float = 1.0
    dfm: bool = False


class MaskProbPath:
    """Sampler for Masked Discrete Diffusion."""
    def __init__(self, model: ConstraintMDDModule, config: SamplingConfig, noise_scheduler: LogLinearNoise):
        """
        Args:
            model: The ConstraintMDDModel.
            config: Configuration object containing sampling parameters.
            mask_index: The index of the mask token.
            noise_scheduler: The noise scheduler (e.g., LogLinearNoise).
        """
        self.model = model
        # Sampling configuration
        # NOTE: config is already SamplingConfig (passed from constraint_models.py as config.sampling)
        # Use it directly, with getattr for robustness
        self.steps = getattr(config, 'steps', 64)
        self.method = getattr(config, 'method', 'tau_leaping')
        self.noise_removal = getattr(config, 'noise_removal', True)
        self.sample_batch_size = getattr(config, 'sample_batch_size', 16)
        # PRISM specific params
        self.num_initial_masks = getattr(config, 'num_initial_masks', 0)
        self.loop_step_on = getattr(config, 'loop_step_on', 1000)
        self.loop_steps = getattr(config, 'loop_steps', 0)
        self.num_remask_loop = getattr(config, 'num_remask_loop', 0)
        self.nucleus_p = getattr(config, 'nucleus_p', 1.0)
        self.dfm = getattr(config, 'dfm', False)

        self.pad_index = ConstraintToken.Pad  # NOTE: Even if use_stop_padding is True, we use Pad for padding (since we set logits[:, :, self.pad_index] = self.neg_infinity)
        self.mask_index = ConstraintToken.Mask
        self.start_index = ConstraintToken.Start
        self.stop_index = ConstraintToken.Stop
        self.num_fixed_tokens = len(ConstraintToken)
        self.noise = noise_scheduler
        self.neg_infinity = -1e6

    def _p_x0_xt(self, logits_ptr: torch.Tensor, batch_t: ConstraintTokenInput) -> torch.Tensor:
        logits_ptr = logits_ptr.clone()  # [B, L, V]

        # Mask out likelihood of predicting [Pad, Mask, Start, Stop] tokens
        logits_ptr[:, :, self.pad_index] = self.neg_infinity
        logits_ptr[:, :, self.mask_index] = self.neg_infinity
        logits_ptr[:, :, self.start_index] = self.neg_infinity
        logits_ptr[:, :, self.stop_index] = self.neg_infinity

        # Mask out likelihood of predicting indices pointing to PrimitiveToken.Pad
        prim_is_pad = rearrange(batch_t['val'] == PrimitiveToken.Pad, "b prim_token_len -> b () prim_token_len")
        num_constraint_tokens = len(ConstraintToken)
        logits_ptr[:, :, num_constraint_tokens:][prim_is_pad.expand(-1, logits_ptr.size(1), -1)] = self.neg_infinity

        # Convert to probabilities p(x0|xt)
        probs = F.softmax(logits_ptr, dim=-1) # [B, L, V]

        # For unmasked tokens in x, the probability is 1.0 for their current value
        # This preserves "clamped" or "observed" values.
        # x != mask_index
        xt_ptr = torch.cat([batch_t['c_val'][:, :1], batch_t['c_val'][:, 2:]], dim=1)  # [B, L]
        unmasked_indices = (xt_ptr != self.mask_index)
        probs[unmasked_indices] = 0
        probs[unmasked_indices, xt_ptr[unmasked_indices]] = 1.0
        return probs

    def _compute_model_outputs(
        self,
        batch: ConstraintTokenInput,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: The dictionary of original primitives and current masked constraint tokens [B, L]
            sigma: Noise level [B, 1]
        """
        # logits: [B, L, V]
        logits_ptr, logits_cnt = self.model.forward(batch, sigma)

        # Convert to probabilities p(x0|xt)
        probs_ptr = self._p_x0_xt(logits_ptr, batch)
        probs_cnt = F.softmax(logits_cnt, dim=-1)

        return probs_ptr, probs_cnt

    def _score_tokens(
        self,
        p_x0: torch.Tensor,
        masks: torch.Tensor,
        x0: torch.Tensor,
        fixed_tokens: torch.Tensor | None,
    ):
        # TODO: Simple random scoring for now
        # scores: [B, L]
        # masked tokens get random score [0, 1]
        # unmasked/fixed tokens get inf

        token_score = torch.rand(p_x0.shape[:-1], device=p_x0.device)
        token_score[~masks] = torch.inf

        if fixed_tokens is not None:
            token_score[fixed_tokens] = torch.inf

        return token_score

    def _select_k_tokens(
        self,
        p_x0: torch.Tensor,
        masks: torch.Tensor,
        k: int,
        x0: torch.Tensor,
        fixed_tokens: torch.Tensor | None,
        N_tokenwise_unmasking: int = 1,
    ) -> torch.Tensor:
        # Select k tokens to keep (or rather, select tokens such that we have k clean ones)
        # If we have N fixed/clean tokens already, and we want K total clean tokens,
        # we calculate scores for all masked tokens. Select top (K - N_current_clean) to unmask.

        # p_x0: [B, L, V]
        # masks: [B, L] (True if masked)
        if N_tokenwise_unmasking > 1:
            raise NotImplementedError("Blockwise masking is not supported yet for _select_k_tokens")
        else:
            token_score = self._score_tokens(p_x0, masks, x0, fixed_tokens)

        # k is target number of clean tokens
        # We want to select indices to be CLEAN.
        # Currently clean: ~masks
        # We want to select additional tokens from 'masks' to make total 'k'.

        # Using top-k on scores:
        # effectively we want to keep the ones with highest scores.
        # "clean tokens that will remain clean: torch.inf"

        # If k is int
        _, change_dim = torch.topk(token_score, k=k, dim=-1, largest=True)
        selected_tokens = torch.zeros_like(token_score, dtype=torch.bool)
        selected_tokens = selected_tokens.scatter_(1, change_dim, True)

        return selected_tokens

    def _topk_static_update(
        self,
        batch: ConstraintTokenInput,
        p_x0_ptr: torch.Tensor,
        p_x0_cnt: torch.Tensor,
        k: int,
        fixed_tokens: torch.Tensor | None,
        N_tokenwise_unmasking: int = 1,
    ) -> ConstraintTokenInput:
        x = batch['c_val']
        masks = (x == self.mask_index)

        # Nucleus sampling for x0 prediction
        if self.nucleus_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            top_p_mask = cumulative_probs <= self.nucleus_p
            top_p_mask[..., 0] = True
            nucleus_probs = sorted_probs * top_p_mask
            nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
            p_x0_nucleus = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
            x0_theta = _sample_categorical(p_x0_nucleus)
        else:
            x0_theta = _sample_categorical(p_x0)

        # Select k tokens to be clean
        clean_token_filter = self._select_k_tokens(p_x0, masks, k, x0_theta, fixed_tokens, N_tokenwise_unmasking)

        # Update x: where filter is True (keep clean), use predicted x0; else mask
        x_next = torch.where(clean_token_filter, x0_theta, self.mask_index)

        # Ensure fixed tokens remain fixed (though logic above should handle it via inf scores)
        if fixed_tokens is not None:
            x_next = torch.where(fixed_tokens, x, x_next)

        # NOTE: In-palace change
        batch['c_val'] = x_next

        return batch

    def _tau_leaping_update(
        self,
        batch: ConstraintTokenInput,
        p_x0_ptr: torch.Tensor,
        p_x0_cnt: torch.Tensor,
        delta_sigma: torch.Tensor,
        fixed_tokens: torch.Tensor | None,
        N_tokenwise_unmasking: int = 1,
    ) -> ConstraintTokenInput:
        x = batch['c_val']
        masks = (x == self.mask_index)

        new_token_indices = p_x0.argmax(dim=-1) # [B, L]

        if N_tokenwise_unmasking > 1:
            B, L = p_x0.shape[0], p_x0.shape[1]

            # Get max probabilities for each token position
            token_max_probs = p_x0.max(dim=-1).values # [B, L]

            # We assume the sequence structure is Start (1 token) + Blocks (N*BLOCK_SIZE tokens) + Remainder
            # The blockwise logic applies to the N*BLOCK_SIZE part.
            # The first token (Start) and any trailing tokens (padding/Stop) are handled individually.
            # For simplicity, we'll apply block logic to the largest possible block-aligned segment.

            # Determine the segment that can be treated as blocks
            # Exclude the first token (Start) from block processing
            body_start_idx = 1
            num_tokens_in_body = L - body_start_idx
            num_blocks = num_tokens_in_body // N_tokenwise_unmasking
            block_aligned_len = num_blocks * N_tokenwise_unmasking
            body_end_idx = body_start_idx + block_aligned_len

            # Extract probabilities for the block-aligned body
            body_token_max_probs = token_max_probs[:, body_start_idx : body_end_idx]
            body_token_max_probs = body_token_max_probs.reshape(B, num_blocks, N_tokenwise_unmasking)

            # Calculate block rates as product of token probabilities within each block
            # This represents the joint probability of all tokens in a block being unmasked to their most likely value
            block_rates_body = body_token_max_probs.prod(dim=-1) # [B, NumBlocks]

            # Apply delta_sigma as the Poisson rate (this is the expected unmask probability change)
            # delta_sigma is ~0.02 per step for 50 steps (uniform across all steps)
            block_poisson_rates = block_rates_body * delta_sigma # [B, NumBlocks]

            # Sample Poisson events at the block level
            block_counts = torch.poisson(block_poisson_rates).long() # [B, NumBlocks]

            # Expand block counts back to token level for the body segment
            # If a block has an event, all its tokens are considered to have an event.
            token_counts_body = block_counts.unsqueeze(-1).expand(-1, -1, N_tokenwise_unmasking).reshape(B, block_aligned_len)

            # Initialize full token counts tensor
            counts = torch.zeros_like(x, dtype=torch.long)
            # Place block-derived counts into the body segment
            counts[:, body_start_idx : body_end_idx] = token_counts_body

            # For tokens outside the block-aligned body (Start and remainder),
            # we can either assume they are fixed or handle them individually.
            # For now, they will have 0 counts unless explicitly handled, which is fine if they are fixed.
            # If they are masked and need to be unmasked, they won't be by this block logic.
            # The current implementation effectively treats them as having 0 unmasking rate.
            # If they are masked and not fixed, they will remain masked.
            # This is a simplification assuming Start/Stop/Padding are usually fixed or not part of dynamic unmasking.

        elif N_tokenwise_unmasking == 1:
            # Token-wise logic
            # Use delta_sigma as the Poisson rate (uniform ~0.02 per step)
            unmask_rate = p_x0 * delta_sigma.unsqueeze(-1) # [B, L, V]
            raw_counts = torch.poisson(unmask_rate).long() # [B, L, V]

            # Zero out counts for special tokens (Pad, Mask itself)
            raw_counts[..., self.pad_index] = 0
            raw_counts[..., self.mask_index] = 0

            # Sum over vocabulary dimension to get total events per position
            counts = raw_counts.sum(dim=-1) # [B, L]

            # new_token_indices is already p_x0.argmax(dim=-1) which is correct for token-wise.

        else:
            raise ValueError("N_tokenwise_unmasking must be 1 or greater.")

        # Common Update Logic for both blockwise and token-wise
        # `counts` here is [B, L] representing "number of unmask events for this position"
        # `new_token_indices` is [B, L] representing the most likely token for each position

        # Zero out counts for positions that are already unmasked (not masked)
        # This ensures we only attempt to unmask currently masked tokens.
        counts[~masks] = 0

        # Conflict resolution: only accept if exactly one event occurs at a position
        # For blockwise, if a block event occurs, all tokens in it get count=1.
        # For token-wise, this filters for single events.
        one_event = (counts == 1)

        # Update
        x_next = x.clone()
        # Only update where exactly one event happened AND it was a mask
        update_mask = one_event & masks
        x_next[update_mask] = new_token_indices[update_mask]

        if fixed_tokens is not None:
            x_next = torch.where(fixed_tokens, x, x_next)

        batch['c_val'] = x_next
        return batch

    def _frontier_tau_leaping_update(
        self,
        batch: ConstraintTokenInput,
        p_x0_ptr: torch.Tensor, # [B, L-1, V_ptr] (BOS + Constraints)
        p_x0_cnt: torch.Tensor, # [B, V_cnt] (Count)
        delta_sigma: torch.Tensor,
        fixed_tokens: torch.Tensor | None,
        block_size: int = CONSTRAINT_TOKEN_LEN,
        prefix_len: int = 2,
    ) -> ConstraintTokenInput:
        """
        Tau-leaping update restricted to the 'Frontier' of each constraint block.

        Theory:
        Unlike standard MDM where all MASKs are candidates, here only the first MASK
        in each block (the Frontier) is active. This reduces the 'active surface area'
        of the diffusion process by a factor of roughly 1/block_size.

        To preserve the global noise schedule (i.e., ensure the sequence finishes
        unmasking within the allotted T steps), the unmasking rate for these
        active frontier tokens must be scaled up by 'block_size'.
        """
        x = batch['c_val']
        masks = (x == self.mask_index)

        # -------------------------------------------------------------
        # 1. Frontier Mask
        # -------------------------------------------------------------
        B, L = x.shape

        # Determine number of full blocks
        num_blocks = (L - prefix_len) // block_size
        end_pos = prefix_len + num_blocks * block_size

        # Extract blocks: [B, N, 4]
        # x_body: [B, N*4]
        x_body = x[:, prefix_len : end_pos]
        x_blocks = x_body.reshape(B, num_blocks, block_size)

        mask_blocks = (x_blocks == self.mask_index)

        # Shift mask to find frontier (False -> True transition)
        # [False, False, True, True] -> Frontier is the first True
        # Pad left with False (assuming prefix/previous tokens are visible/handled)
        padded_mask = torch.cat(
            [torch.zeros_like(mask_blocks[:, :, :1]), mask_blocks[:, :, :-1]],
            dim=2
        )
        is_frontier_block = mask_blocks & (~padded_mask) # [B, N, 4]

        # Flatten back to [B, L]
        # Initialize with False (Prefix and remainder are handled as non-frontier)
        frontier_mask = torch.zeros_like(masks, dtype=torch.bool)
        frontier_mask[:, prefix_len : end_pos] = is_frontier_block.reshape(B, -1)
        frontier_mask[:, 1] = masks[:, 1]  # COUNT token

        # -------------------------------------------------------------
        # 2. Rate Calculation
        # -------------------------------------------------------------

        # A. Pointer Part (Constraints + BOS)
        # p_x0_ptr corresponds to indices [0, 2, 3, ...] of batch['c_val']
        # We need to construct a matching mask and scale map.

        # Mask for ptr: remove Count index (index 1)
        # frontier_mask: [BOS, COUNT, C1...] -> frontier_mask_ptr: [BOS, C1...]
        frontier_mask_ptr = torch.cat([frontier_mask[:, 0:1], frontier_mask[:, 2:]], dim=1)

        # Rate Scaling for Pointer
        # BOS (Index 0) -> Scale 1.0
        # Constraints (Index 1+) -> Scale block_size
        scale_map_ptr = torch.ones((B, L - 1, 1), device=x.device)
        scale_map_ptr[:, 1:, :] = float(block_size)

        # Calculate Rate
        # delta_sigma: [B, 1] -> [B, 1, 1]
        base_rate_ptr = p_x0_ptr * delta_sigma.unsqueeze(-1)
        unmask_rate_ptr = base_rate_ptr * scale_map_ptr * frontier_mask_ptr.unsqueeze(-1)

        # B. Count Part (Count Token)
        # p_x0_cnt is [B, V_cnt]
        # frontier_mask[:, 1] is [B]

        base_rate_cnt = p_x0_cnt * delta_sigma # [B, V_cnt]
        # Only valid if Count is currently a frontier (masked)
        unmask_rate_cnt = base_rate_cnt * frontier_mask[:, 1].unsqueeze(-1)

        # -------------------------------------------------------------
        # 3. Poisson Sampling & Merge
        # -------------------------------------------------------------

        # Sample separately
        raw_counts_ptr = torch.poisson(unmask_rate_ptr).long() # [B, L-1, V_ptr]
        raw_counts_cnt = torch.poisson(unmask_rate_cnt).long() # [B, V_cnt]

        # Prevent invalid transitions
        raw_counts_ptr[..., self.pad_index] = 0
        raw_counts_ptr[..., self.mask_index] = 0
        # (Count has no special tokens in its V_cnt space usually, so no masking needed)

        # Sum events per position
        counts_ptr = raw_counts_ptr.sum(dim=-1) # [B, L-1]
        counts_cnt = raw_counts_cnt.sum(dim=-1) # [B]

        # Merge counts into full sequence [B, L]
        # [BOS(ptr), COUNT(cnt), Constraints(ptr)...]
        counts = torch.cat([
            counts_ptr[:, 0:1],
            counts_cnt.unsqueeze(-1),
            counts_ptr[:, 1:]
        ], dim=1)

        # -------------------------------------------------------------
        # 4. Update State
        # -------------------------------------------------------------

        # Get Argmax Indices
        new_indices_ptr = p_x0_ptr.argmax(dim=-1) # [B, L-1]
        new_indices_cnt = p_x0_cnt.argmax(dim=-1) # [B]

        # Apply Offset to Count Indices
        new_indices_cnt = new_indices_cnt + self.num_fixed_tokens

        # Merge Indices into full sequence [B, L]
        new_token_indices = torch.cat([
            new_indices_ptr[:, 0:1],
            new_indices_cnt.unsqueeze(-1),
            new_indices_ptr[:, 1:]
        ], dim=1)

        # Update Logic (Standard Tau Leaping)
        one_event = (counts == 1)
        update_locs = one_event & frontier_mask

        x_next = x.clone()
        x_next[update_locs] = new_token_indices[update_locs]

        # Re-apply fixed tokens constraints
        if fixed_tokens is not None:
            x_next = torch.where(fixed_tokens, x, x_next)

        batch['c_val'] = x_next
        return batch

    def _apply_solver_feedback(
        self,
        batch: ConstraintTokenInput,
        p_x0: torch.Tensor,
        num_bins: int,
        max_solver_iters: int = 5,
        scoot_valid_constraints: bool = False,
        strip_first_stop_token: bool = False,
    ) -> ConstraintTokenInput:
        from img2cad_diffusion.convert_sketch_to_freecad import FreeCADConversionError
        from img2cad_diffusion.detokenization import reconstruct_constraints, reconstruct_sketch
        from img2cad_diffusion.solver_feedback import SolverStatus, convert_sketch_to_freecad, get_solver_status
        import FreeCAD

        if not self.model.hparams.data.align_constraint_token_length:
            raise ValueError("align_constraint_token_length must be True")

        x = batch['c_val'].clone()
        B, L = x.shape

        for b in range(B):
            prim_tokens = batch['val'][b]
            sample_x = x[b]

            # STEP 1: Mask syntax-error constraints
            valid_constraints, invalid_constraints = reconstruct_constraints(
                sample_x.cpu().numpy(),
                token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
            )
            for constraint_type, refs, (start, end) in invalid_constraints:
                sample_x[start:end] = self.mask_index

            # STEP2: MASK solver-alerting constraints
            sketch = reconstruct_sketch(
                prim_tokens.cpu().numpy(),
                sample_x.cpu().numpy(),
                num_bins,
                token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
            )

            status = SolverStatus()
            removed_constraint_ids: list[int] = []

            for _ in range(max_solver_iters):
                try:
                    doc, fc_sketch, _, fc_mapping = convert_sketch_to_freecad(sketch, doc_name="SolverEval")
                except FreeCADConversionError as e:
                    # If a constraint caused the error, remove it and retry
                    if e.phase == "constraint":
                        cid = e.constraint_id
                        if cid in sketch.constraints:
                            del sketch.constraints[cid]
                            removed_constraint_ids.append(int(cid[1:]))
                            continue  # Retry conversion
                    # If primitive error or unrecoverable, skip this sample
                    print(f"[_apply_solver_feedback] Error: {e}")
                    break
                except Exception as e:
                    print(f"[_apply_solver_feedback] Unexpected error: {e}")
                    break

                status = get_solver_status(fc_sketch, fc_mapping)
                FreeCAD.closeDocument(doc.Name)

                # Get problem constraints
                problem_indices = status.MappedConflictingIndices + status.MappedRedundantIndices + status.MappedMalformedIndices
                if not problem_indices:
                    break

                # Find least confident constraint to mask
                min_conf = float('inf')
                target_idx = -1

                for constraint_idx in problem_indices:
                    constraint_type, constraint_refs, (start_pos, end_pos) = valid_constraints[constraint_idx]

                    # Calculate confidence as product of p(x0|xt)
                    chunk_probs = p_x0[b, start_pos:end_pos, :]
                    chunk_tokens = sample_x[start_pos:end_pos]
                    token_probs = chunk_probs.gather(dim=-1, index=chunk_tokens.unsqueeze(1)).squeeze(1)
                    score = token_probs.prod().item()

                    if score < min_conf:
                        min_conf = score
                        target_idx = constraint_idx

                # Remove constraint from Sketch
                del sketch.constraints[f"c{target_idx}"]
                removed_constraint_ids.append(target_idx)

            for cid in removed_constraint_ids:
                _, _, (start_pos, end_pos) = valid_constraints[cid]
                sample_x[start_pos:end_pos] = self.mask_index

            x[b] = sample_x

        batch['c_val'] = x
        return batch

    @staticmethod
    def _schedule_k(
        total_mask_tokens: int,
        total_steps: int,
        remasking_s_on: int,
        remasking_steps: int,
        current_step: int,
    ) -> int:
        # Linearly decrease the number of masks (or increase number of clean tokens)
        model_length = total_mask_tokens

        if current_step < remasking_s_on:
            k = math.ceil((model_length / total_steps) * (current_step + 1))
        elif current_step >= remasking_s_on and current_step < (remasking_s_on + remasking_steps):
            k = math.ceil((model_length / total_steps) * remasking_s_on)
        else:
            k = math.ceil((model_length / total_steps) * (current_step + 1 - remasking_steps))

        return min(k, model_length)

    @torch.inference_mode()
    def generate_sample(
        self,
        batch: ConstraintTokenInput,
        num_steps: int = 64,
        start_tokens: list[int] | None = None,
        N_tokenwise_unmasking: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate samples using the Discrete Diffusion process.
        Args:
            batch: Initial batch containing primitives and 'c_val' (which serves as ground truth target usually, but here we mask it).
            num_steps: Number of diffusion steps.
            start_tokens: List of tokens to fix at the beginning of the sequence.
            N_tokenwise_unmasking: N-consecutive tokens are unmasked at one time
        """
        batch = copy.deepcopy(batch)

        # Initialize x with all masks
        # NOTE: The input batch['c_val'] DOESN'T contain the COUNT token
        device = batch['c_val'].device
        b, total_tokens = batch['c_val'].shape
        total_tokens += 1  # NOTE: ADD COUNT TOKEN
        x = torch.full((b, total_tokens), self.mask_index, device=device, dtype=torch.long)
        fixed_tokens = torch.zeros_like(x, dtype=torch.bool)

        # Initialize fixed_tokens
        if start_tokens is not None:
            n_start = len(start_tokens)
            start_tensor = torch.tensor(start_tokens, device=device, dtype=x.dtype)

            if n_start > total_tokens:
                raise ValueError(f"Length of start_tokens ({n_start}) exceeds sequence length ({total_tokens})")

            if not torch.all(batch['c_val'][:, :n_start] == start_tensor):
                 raise ValueError(f"`c_val` GT must start from `start_tokens`={start_tokens}")

            x[:, :n_start] = start_tensor.reshape(1, n_start)
            fixed_tokens[:, :n_start] = True

        timesteps = torch.linspace(1, 1e-5, num_steps + 1, device=device)

        # NOTE: From now on we don't have access to the GT, only the predicted tokens
        batch['c_val'] = x
        intermediates = [batch['c_val'].clone()]

        for i in tqdm(range(num_steps), desc="Sampling"):
            t = timesteps[i] * torch.ones(b, 1, device=device)
            t_next = timesteps[i + 1] * torch.ones(b, 1, device=device)

            # predict p(x0|xt)
            sigma_t, _ = self.noise(t)
            sigma_t_next, _ = self.noise(t_next)
            # delta_unmask_prob: the expected change in unmask probability
            # unmask_prob = 1 - exp(-sigma) ≈ (1-eps)*t for LogLinear noise
            # This gives uniform rate (~0.02 per step for 50 steps)
            unmask_prob_t = 1 - torch.exp(-sigma_t)
            unmask_prob_t_next = 1 - torch.exp(-sigma_t_next)
            delta_unmask_prob = unmask_prob_t - unmask_prob_t_next
            p_x0_ptr, p_x0_cnt = self._compute_model_outputs(batch, sigma_t)

            # update to x_{t-1} (NOTE: batch['c_val'] must be a partially masked for input in order to see the mask token positions)
            if self.method.startswith('tau_leaping'):
                raise NotImplementedError
                batch = self._tau_leaping_update(
                    batch,
                    p_x0_ptr,
                    p_x0_cnt,
                    delta_unmask_prob,
                    fixed_tokens=fixed_tokens,
                    N_tokenwise_unmasking=N_tokenwise_unmasking)
            elif self.method.startswith('frontier_tau_leaping'):
                batch = self._frontier_tau_leaping_update(
                    batch,
                    p_x0_ptr,
                    p_x0_cnt,
                    delta_unmask_prob,
                    fixed_tokens=fixed_tokens,
                    block_size=CONSTRAINT_TOKEN_LEN,
                    prefix_len=2)
            elif self.method.startswith('topk_static'):
                raise NotImplementedError
                k = self._schedule_k(total_tokens, num_steps, self.loop_step_on, self.loop_steps, i)
                batch = self._topk_static_update(
                    batch,
                    p_x0_ptr,
                    p_x0_cnt,
                    k,
                    fixed_tokens=fixed_tokens,
                    N_tokenwise_unmasking=N_tokenwise_unmasking)
            else:
                raise NotImplementedError(f"Sampling method {self.method} not implemented")

            # Solver feedback (Remasking)
            if self.method.endswith('solver_feedback'):
                raise NotImplementedError
                batch = self._apply_solver_feedback(
                    batch,
                    p_x0,
                    num_bins=self.model.hparams.data.num_position_bins,
                    max_solver_iters=5,
                    scoot_valid_constraints=False,
                    strip_first_stop_token=False,
                )

            intermediates.append(batch['c_val'])

        # Final cleanup / noise removal step if enabled
        if self.noise_removal:
            if self.method.startswith('frontier_tau_leaping'):
                # Block structure
                # prefix_len = 2 (BOS + COUNT)
                prefix_len = 2
                num_blocks = (total_tokens - prefix_len) // CONSTRAINT_TOKEN_LEN
                end_pos = prefix_len + num_blocks * CONSTRAINT_TOKEN_LEN

                # Prepare time t ~ 0
                t = timesteps[-1] * torch.ones(b, 1, device=device)
                sigma_t, _ = self.noise(t)

                # Iterate to fill dependency chains (Type -> Ref1 -> Ref2 -> Ref3)
                # Count token (depth 1) and Constraints (depth 4) will be resolved sequentially.
                for _ in range(CONSTRAINT_TOKEN_LEN):
                    x = batch['c_val']
                    masks = (x == self.mask_index)

                    # --- 1. Construct Frontier Mask ---

                    # A. Constraints Body (Left-to-Right dependency)
                    x_body = x[:, prefix_len : end_pos]
                    x_blocks = x_body.reshape(b, num_blocks, CONSTRAINT_TOKEN_LEN)
                    mask_blocks = (x_blocks == self.mask_index)

                    # Frontier = First MASK in each block
                    shifted = torch.zeros_like(mask_blocks)
                    shifted[..., 1:] = mask_blocks[..., :-1]
                    frontier_blocks = mask_blocks & (~shifted)

                    # B. Combine with Count Token
                    frontier_mask = torch.zeros_like(x, dtype=torch.bool)
                    frontier_mask[:, prefix_len : end_pos] = frontier_blocks.reshape(b, -1)

                    # Count (Index 1) is also a frontier if it is masked
                    frontier_mask[:, 1] = masks[:, 1]

                    # Skip if no more MASKs to update
                    if not frontier_mask.any():
                        break

                    # --- 2. Predict & Construct New Tokens ---

                    # p_x0_ptr: [B, L-1, V_ptr] (BOS, C1, C2...)
                    # p_x0_cnt: [B, V_cnt]      (Count Value 0~N)
                    p_x0_ptr, p_x0_cnt = self._compute_model_outputs(batch, sigma_t)

                    # Argmax
                    new_idx_ptr = p_x0_ptr.argmax(dim=-1) # [B, L-1]
                    new_idx_cnt = p_x0_cnt.argmax(dim=-1) # [B]

                    # Apply Offset to Count
                    new_idx_cnt = new_idx_cnt + self.num_fixed_tokens

                    # Merge into [B, L] structure: [BOS(ptr), COUNT(cnt), Constraints(ptr)...]
                    new_tokens = torch.cat([
                        new_idx_ptr[:, 0:1],          # BOS
                        new_idx_cnt.unsqueeze(-1),    # COUNT (with offset)
                        new_idx_ptr[:, 1:]            # Constraints
                    ], dim=1)

                    # --- 3. Update State ---

                    # Update only frontier positions
                    x_next = x.clone()
                    x_next[frontier_mask] = new_tokens[frontier_mask]

                    if fixed_tokens is not None:
                        x_next = torch.where(fixed_tokens, x, x_next)
                    batch['c_val'] = x_next

            else:
                raise NotImplementedError
                # Standard noise removal: fill all MASKs at once
                t = timesteps[-1] * torch.ones(b, 1, device=device)
                sigma_t, _ = self.noise(t)
                p_x0_ptr, p_x0_cnt = self._compute_model_outputs(batch, sigma_t)
                new_tokens = p_x0.argmax(dim=-1)

                # Only replace remaining MASK positions, preserve already-unmasked tokens
                still_masked = (batch['c_val'] == self.mask_index)
                x = torch.where(still_masked, new_tokens, batch['c_val'])

                if fixed_tokens is not None:
                    x = torch.where(fixed_tokens, batch['c_val'], x)
                batch['c_val'] = x

            intermediates.append(batch['c_val'])
        else:
            raise ValueError("To determine the COUNT token and the top-COUNT constraints, full unmasking is necessary.")

        return batch['c_val'], intermediates
