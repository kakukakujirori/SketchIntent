import torch
import math
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from typing import Optional, Tuple, List, Dict
from img2cad_diffusion import constraint_data

def _sample_categorical(categorical_probs: Tensor) -> Tensor:
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


class MaskProbPath:
    """Sampler for Masked Discrete Diffusion."""
    def __init__(self, model, config, mask_index: int, noise_scheduler):
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
            self.method = getattr(sampling_config, 'method', 'topk_static') # 'topk_static', 'ddpm_cache'
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
            self.method = 'topk_static'
            self.noise_removal = True
            self.sample_batch_size = 16
            self.num_initial_masks = 0
            self.loop_step_on = 1000
            self.loop_steps = 0
            self.num_remask_loop = 0
            self.nucleus_p = 1.0
            self.dfm = False

        self.mask_index = mask_index
        self.noise = noise_scheduler
        self.neg_infinity = -1e6

    def _p_x0_xt(self, logits: Tensor, x: Tensor) -> Tensor:
        # logits: [B, L, V]
        # x: [B, L]
        # Mask out likelihood of predicting Mask token itself
        logits = logits.clone()
        logits[:, :, self.mask_index] += self.neg_infinity
        probs = F.softmax(logits, dim=-1) # [B, L, V]

        # For unmasked tokens in x, the probability is 1.0 for their current value
        # This preserves "clamped" or "observed" values.
        # x != mask_index
        unmasked_indices = (x != self.mask_index)
        probs[unmasked_indices] = 0
        probs[unmasked_indices, x[unmasked_indices]] = 1.0
        return probs

    def _schedule_k(self, total_mask_tokens, total_steps, remasking_s_on, remasking_steps, current_step):
        # Linearly decrease the number of masks (or increase number of clean tokens)
        # We want k (clean tokens) to go from 0 to L
        # Or masked tokens to go from L to 0
        # This implementation seems to calculate "number of clean tokens"
        # Adjusted for this specific task where we start fully masked usually.

        # Simple linear schedule for standard topk:
        # k = ceil(L * (t+1)/T)

        model_length = total_mask_tokens # assuming full mask start

        if current_step < remasking_s_on:
             k = math.ceil((model_length / total_steps) * (current_step + 1))
        elif current_step >= remasking_s_on and current_step < (remasking_s_on + remasking_steps):
             k = math.ceil((model_length / total_steps) * remasking_s_on)
        else:
             k = math.ceil((model_length / total_steps) * (current_step + 1 - remasking_steps))

        return min(k, model_length)

    def _compute_model_outputs(self, x: Tensor, sigma: Tensor, batch_context: constraint_data.TokenInput) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Args:
            x: Current hypothesis for constraint tokens [B, L]
            sigma: Noise level [B, 1]
            batch_context: The original batch input containing primitives etc.
        """
        # Construct the input for the model
        src = batch_context.copy()
        src['c_val'] = x # Update with current hypothesis

        # Model forward
        # logits: [B, L, V]
        logits = self.model.forward(src, sigma)

        # Convert to probabilities p(x0|xt)
        probs = self._p_x0_xt(logits, x)

        # Confidence - simple random for now as we don't have a separate adapter/critic
        conf = torch.rand(logits.shape[:-1], device=logits.device)

        return probs, conf, logits

    def _score_tokens(self, p_x0, masks, conf=None, step=None, x0=None, fixed_tokens=None):
        # Simple random scoring for now (PRISM default/baseline)
        # scores: [B, L]
        # masked tokens get random score [0, 1]
        # unmasked/fixed tokens get inf

        token_score = torch.rand(p_x0.shape[:-1], device=p_x0.device)
        token_score[~masks] = torch.inf

        if fixed_tokens is not None:
            token_score[fixed_tokens] = torch.inf

        return token_score

    def _select_k_tokens(self, p_x0, masks, conf, k, step, x0, fixed_tokens):
        # Select k tokens to keep (or rather, select tokens such that we have k clean ones)
        # If we have N fixed/clean tokens already, and we want K total clean tokens.
        # logic: calculate scores for all masked tokens. Select top (K - N_current_clean) to unmask.

        # p_x0: [B, L, V]
        # masks: [B, L] (True if masked)

        token_score = self._score_tokens(p_x0, masks, conf, step, x0, fixed_tokens)

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


    def _topk_static_update(self, x, t, k, step, fixed_tokens, batch_context):
        sigma_t, _ = self.noise(t)
        masks = (x == self.mask_index)

        p_x0, conf, logits = self._compute_model_outputs(x, sigma_t, batch_context)

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
        clean_token_filter = self._select_k_tokens(p_x0, masks, conf, k, step, x0_theta, fixed_tokens)

        # Update x: where filter is True (keep clean), use predicted x0; else mask
        x_next = torch.where(clean_token_filter, x0_theta, self.mask_index)

        # Ensure fixed tokens remain fixed (though logic above should handle it via inf scores)
        if fixed_tokens is not None:
             x_next = torch.where(fixed_tokens, x, x_next)

        return x_next

    @torch.inference_mode()
    def generate_sample(self, batch: constraint_data.TokenInput, num_steps: int = 64, device='cuda') -> Tuple[Tensor, List[Tensor]]:
        """
        Generate samples using the Discrete Diffusion process.
        Args:
            batch: Initial batch containing primitives and 'c_val' (which serves as ground truth target usually, but here we mask it).
            num_steps: Number of diffusion steps.
        """
        # Initialize x with all masks (except maybe padding?)
        # batch['c_val'] has shape [B, L]
        # We should respect padding in c_val if possible, but simplest is to mask everything first.
        # Actually, if we mask padding, the model should ideally learn to predict Pad token for those positions
        # or we handled padding separately?
        # For validation, let's assume valid lengths are handled by the model or we mask everything up to max_len.

        b, l = batch['c_val'].shape
        x = torch.full((b, l), self.mask_index, device=device, dtype=torch.long)

        # Identify fixed tokens (e.g. Pad) if we wanted to keep them.
        # For now, let's assume we want to generate EVERYTHING.
        # If there are constraints we want to condition on (inpainting), we would set them here.
        fixed_tokens = torch.zeros_like(x, dtype=torch.bool)

        # If we strictly want to ignore padding during generation, we could pre-fill pads from batch['c_val']
        # and mark them fixed.
        # fixed_tokens = (batch['c_val'] == constraint_data.Token.Pad)
        # x = torch.where(fixed_tokens, batch['c_val'], x)

        intermediates = [x.clone()]

        timesteps = torch.linspace(1, 1e-5, num_steps + 1, device=device)

        # Schedule for k (number of clean tokens)
        # We start with 0 clean tokens (all masked), end with L clean tokens.
        # k_schedule[i] is how many clean tokens we want at step i.

        total_tokens = l # or count of non-fixed tokens per batch item? vectorized schedule is harder.
        # Let's use constant L for all batch items for simplicity.

        k_schedule = [self._schedule_k(total_tokens, num_steps, self.loop_step_on, self.loop_steps, i)
                      for i in range(num_steps)]

        for i in tqdm(range(num_steps), desc="Sampling"):
            t = timesteps[i] * torch.ones(b, 1, device=device) # [B, 1]
            k = k_schedule[i]

            x = self._topk_static_update(x, t, k, step=i, fixed_tokens=fixed_tokens, batch_context=batch)
            intermediates.append(x.clone())

        # Final cleanup / noise removal step if enabled
        if self.noise_removal:
             # Predict one last time with t ~ 0
             t = timesteps[-1] * torch.ones(b, 1, device=device)
             sigma_t, _ = self.noise(t)
             p_x0, _, _ = self._compute_model_outputs(x, sigma_t, batch)
             x = p_x0.argmax(dim=-1)
             if fixed_tokens is not None:
                x = torch.where(fixed_tokens, batch['c_val'], x) # Restore fixed tokens just in case
             intermediates.append(x.clone())

        return x, intermediates
