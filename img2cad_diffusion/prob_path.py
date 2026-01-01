import copy
import torch
import math
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from img2cad_diffusion.constraint_data import ConstraintTokenInput
from img2cad_diffusion.tokens import ConstraintToken, PrimitiveToken


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


class MaskProbPath:
    """Sampler for Masked Discrete Diffusion."""
    def __init__(self, model: torch.nn.Module, config: dict, noise_scheduler):
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

        self.pad_index = ConstraintToken.Pad
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
    ) -> torch.Tensor:
        # Select k tokens to keep (or rather, select tokens such that we have k clean ones)
        # If we have N fixed/clean tokens already, and we want K total clean tokens,
        # we calculate scores for all masked tokens. Select top (K - N_current_clean) to unmask.

        # p_x0: [B, L, V]
        # masks: [B, L] (True if masked)

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
        t: float | torch.Tensor,
        k: int,
        step: int,
        fixed_tokens: torch.Tensor | None,
    ) -> ConstraintTokenInput:
        x = batch['c_val']
        sigma_t, _ = self.noise(t)
        masks = (x == self.mask_index)

        p_x0, logits = self._compute_model_outputs(batch, sigma_t)

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
        clean_token_filter = self._select_k_tokens(p_x0, masks, k, x0_theta, fixed_tokens)

        # Update x: where filter is True (keep clean), use predicted x0; else mask
        x_next = torch.where(clean_token_filter, x0_theta, self.mask_index)

        # Ensure fixed tokens remain fixed (though logic above should handle it via inf scores)
        if fixed_tokens is not None:
            x_next = torch.where(fixed_tokens, x, x_next)

        # NOTE: In-palace change
        batch['c_val'] = x_next

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

    @torch.inference_mode()
    def generate_sample(self, batch: ConstraintTokenInput, num_steps: int = 64, start_tokens: list[int] | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        k_schedule = [self._schedule_k(total_tokens, num_steps, self.loop_step_on, self.loop_steps, i) for i in range(num_steps)]

        # NOTE: From now on we don't have access to the GT, only the predicted tokens
        batch['c_val'] = x
        intermediates = [batch['c_val'].clone()]

        for i in tqdm(range(num_steps), desc="Sampling"):
            t = timesteps[i] * torch.ones(b, 1, device=device)
            k = k_schedule[i]
            batch = self._topk_static_update(batch, t, k, step=i, fixed_tokens=fixed_tokens)
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
