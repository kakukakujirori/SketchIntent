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
        sampling_config = getattr(config, 'sampling', None)

        if sampling_config is not None:
            self.steps = getattr(sampling_config, 'steps', 64)
            self.method = getattr(sampling_config, 'method', 'tau_leaping') # 'topk_static', 'tau_leaping'
            self.noise_removal = getattr(sampling_config, 'noise_removal', True)
            self.sample_batch_size = getattr(sampling_config, 'sample_batch_size', 16) # Not strictly used if batch is provided
            # PRISM specific params
            self.num_initial_masks = getattr(sampling_config, 'num_initial_masks', 0)
            self.loop_step_on = getattr(sampling_config, 'loop_step_on', 1000)
            self.loop_steps = getattr(sampling_config, 'loop_steps', 0)
            self.num_remask_loop = getattr(sampling_config, 'num_remask_loop', 0)
            self.nucleus_p = getattr(sampling_config, 'nucleus_p', 1.0)
            self.dfm = getattr(sampling_config, 'dfm', False)
        else:
            # Defaults if 'sampling' section is missing
            self.steps = 64
            self.method = 'tau_leaping' # 'topk_static' or 'tau_leaping'
            self.noise_removal = True
            self.sample_batch_size = 16
            self.num_initial_masks = 0
            self.loop_step_on = 1000
            self.loop_steps = 0
            self.num_remask_loop = 0
            self.nucleus_p = 1.0
            self.dfm = False

        self.pad_index = ConstraintToken.Stop if self.model.hparams.data.use_stop_padding else ConstraintToken.Pad
        self.mask_index = ConstraintToken.Mask
        self.noise = noise_scheduler
        self.neg_infinity = -1e6

    def _p_x0_xt(self, logits: torch.Tensor, batch_t: ConstraintTokenInput) -> torch.Tensor:
        logits = logits.clone()  # [B, L, V]

        # Mask out likelihood of predicting Pad & Mask tokens
        logits[:, :, self.pad_index] = self.neg_infinity
        logits[:, :, self.mask_index] = self.neg_infinity

        # Mask out likelihood of predicting indices pointing to PrimitiveToken.Pad
        prim_is_pad = rearrange(batch_t['val'] == PrimitiveToken.Pad, "b prim_token_len -> b () prim_token_len")
        num_constraint_tokens = len(ConstraintToken)
        logits[:, :, num_constraint_tokens:][prim_is_pad.expand(-1, logits.size(1), -1)] = self.neg_infinity

        # Convert to probabilities p(x0|xt)
        probs = F.softmax(logits, dim=-1) # [B, L, V]

        # For unmasked tokens in x, the probability is 1.0 for their current value
        # This preserves "clamped" or "observed" values.
        # x != mask_index
        xt = batch_t['c_val']  # [B, L]
        unmasked_indices = (xt != self.mask_index)
        probs[unmasked_indices] = 0
        probs[unmasked_indices, xt[unmasked_indices]] = 1.0
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
        logits = self.model.forward(batch, sigma)

        # Convert to probabilities p(x0|xt)
        probs = self._p_x0_xt(logits, batch)

        return probs, logits

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
        constraint_blockwise_masking: bool = False,
    ) -> torch.Tensor:
        # Select k tokens to keep (or rather, select tokens such that we have k clean ones)
        # If we have N fixed/clean tokens already, and we want K total clean tokens,
        # we calculate scores for all masked tokens. Select top (K - N_current_clean) to unmask.

        # p_x0: [B, L, V]
        # masks: [B, L] (True if masked)
        if constraint_blockwise_masking:
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
        p_x0: torch.Tensor,
        k: int,
        fixed_tokens: torch.Tensor | None,
        constraint_blockwise_masking: bool = False,
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
        clean_token_filter = self._select_k_tokens(p_x0, masks, k, x0_theta, fixed_tokens, constraint_blockwise_masking)

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
        p_x0: torch.Tensor,
        rate_t: torch.Tensor,
        dt: float,
        fixed_tokens: torch.Tensor | None,
        constraint_blockwise_masking: bool = False,
    ) -> ConstraintTokenInput:
        x = batch['c_val']
        masks = (x == self.mask_index)

        new_token_indices = p_x0.argmax(dim=-1) # [B, L]

        if constraint_blockwise_masking:
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
            num_blocks = num_tokens_in_body // CONSTRAINT_TOKEN_LEN
            block_aligned_len = num_blocks * CONSTRAINT_TOKEN_LEN
            body_end_idx = body_start_idx + block_aligned_len

            # Extract probabilities for the block-aligned body
            body_token_max_probs = token_max_probs[:, body_start_idx : body_end_idx]
            body_token_max_probs = body_token_max_probs.reshape(B, num_blocks, CONSTRAINT_TOKEN_LEN)

            # Calculate block rates as product of token probabilities within each block
            # This represents the joint probability of all tokens in a block being unmasked to their most likely value
            block_rates_body = body_token_max_probs.prod(dim=-1) # [B, NumBlocks]

            # Apply the global unmasking rate (rate_t) to the block rates
            block_poisson_rates = block_rates_body * rate_t # [B, NumBlocks]

            # Sample Poisson events at the block level
            block_counts = torch.poisson(block_poisson_rates * dt).long() # [B, NumBlocks]

            # Expand block counts back to token level for the body segment
            # If a block has an event, all its tokens are considered to have an event.
            token_counts_body = block_counts.unsqueeze(-1).expand(-1, -1, CONSTRAINT_TOKEN_LEN).reshape(B, block_aligned_len)

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

        else:
            # Token-wise logic
            unmask_rate = p_x0 * rate_t.unsqueeze(-1) # [B, L, V]
            raw_counts = torch.poisson(unmask_rate * dt).long() # [B, L, V]

            # Zero out counts for special tokens (Pad, Mask itself)
            raw_counts[..., self.pad_index] = 0
            raw_counts[..., self.mask_index] = 0

            # Sum over vocabulary dimension to get total events per position
            counts = raw_counts.sum(dim=-1) # [B, L]

            # new_token_indices is already p_x0.argmax(dim=-1) which is correct for token-wise.

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

        if not self.model.hparams.model.align_constraint_token_length:
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
        constraint_blockwise_masking: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Generate samples using the Discrete Diffusion process.
        Args:
            batch: Initial batch containing primitives and 'c_val' (which serves as ground truth target usually, but here we mask it).
            num_steps: Number of diffusion steps.
            start_tokens: List of tokens to fix at the beginning of the sequence.
        """
        batch = copy.deepcopy(batch)

        # Initialize x with all masks
        device = batch['c_val'].device
        b, total_tokens = batch['c_val'].shape
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

            # predict p(x0|xt)
            sigma_t, rate_t = self.noise(t)
            p_x0, _ = self._compute_model_outputs(batch, sigma_t)

            # update to x_{t-1} (NOTE: batch['c_val'] must be a partially masked for input in order to see the mask token positions)
            if self.method.startswith('tau_leaping'):
                batch = self._tau_leaping_update(
                    batch,
                    p_x0,
                    rate_t,
                    1.0/num_steps,
                    fixed_tokens=fixed_tokens,
                    constraint_blockwise_masking=constraint_blockwise_masking)
            elif self.method.startswith('topk_static'):
                k = self._schedule_k(total_tokens, num_steps, self.loop_step_on, self.loop_steps, i)
                batch = self._topk_static_update(
                    batch,
                    p_x0,
                    k,
                    fixed_tokens=fixed_tokens,
                    constraint_blockwise_masking=constraint_blockwise_masking)
            else:
                raise NotImplementedError(f"Sampling method {self.method} not implemented")

            # Solver feedback (Remasking)
            if self.method.endswith('solver_feedback'):
                batch = self._apply_solver_feedback(
                    batch,
                    p_x0,
                    num_bins=self.model.hparams.data.num_position_bins,
                    max_solver_iters=5,
                    scoot_valid_constraints=False,
                    strip_first_stop_token=False,
                )

            intermediates.append(batch['c_val'].clone())

        # Final cleanup / noise removal step if enabled
        if self.noise_removal:
            # Predict one last time with t ~ 0
            t = timesteps[-1] * torch.ones(b, 1, device=device)
            sigma_t, _ = self.noise(t)
            p_x0, _ = self._compute_model_outputs(batch, sigma_t)
            x = p_x0.argmax(dim=-1)
            if fixed_tokens is not None:
                x = torch.where(fixed_tokens, batch['c_val'], x) # Restore fixed tokens just in case
            batch['c_val'] = x
            intermediates.append(batch['c_val'].clone())

        return batch['c_val'], intermediates
