"""Main pytorch-lightning models for training constraint problem.
"""
import dataclasses
from typing import List, Optional, Union

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from einops import rearrange

from img2cad_diffusion import primitive_data
from sketchgraphs_models.nn.summary import CohenKappa
from . import dit
from .constraint_data import ConstraintDataConfig, ConstraintTokenInput
from .iou import ConstraintIoU
from .prob_path_with_cnt import MaskProbPath, SamplingConfig
from .tokens import SpecialToken, PrimitiveToken, PrimitiveParamToken, ConstraintToken, ConstraintParamToken
from .tokenization import CONSTRAINT_TOKEN_LEN
from .token_masking import TokenMasking
from .token_reordering import TokenReordering


@dataclasses.dataclass
class ConstraintNetConfig:
    """Configuration for the main constraint model architecture.

    Attributes
    ----------
    embedding_dimension
        Dimension of embeddings used
    hidden_size
        Dimension of hidden layers
    num_heads
        Number of heads used in multi-head attention
    num_layers
        Number of transformer layers
    """
    model_type: str = "mdd"  # "ar" or "mdd"
    embedding_dimension: int = 128
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 8


@dataclasses.dataclass
class ConstraintModelConfig:
    """Configuration for the main constraint model.

    Attributes
    ----------
    net : torch.nn.Module
        Configuration for the main constraint model architecture
    token_masking_strategies : list[str]
        Token masking strategies (e.g., ["tokenwise", "4-tokenwise"])
    token_masking_probs : Union[str, float, list[str | float]]
        Token masking probabilities (e.g., [0.25, "antithetic"])
    gt_reordering_strategies : list[str]
        GT reordering strategies (e.g., ["hungarian", "none"])
    """
    net: ConstraintNetConfig = dataclasses.field(default_factory=ConstraintNetConfig)
    token_masking_strategies : list[str] = dataclasses.field(default_factory=lambda: ["tokenwise"])
    token_masking_probs : list = dataclasses.field(default_factory=lambda: ["antithetic"])
    gt_reordering_strategies : list[str] = dataclasses.field(default_factory=lambda: ["none"])


@dataclasses.dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0


@dataclasses.dataclass
class ConstraintModelTrainingConfig:
    """Training configuration for constraint models."""
    data: ConstraintDataConfig = dataclasses.field(default_factory=ConstraintDataConfig)
    model: ConstraintModelConfig = dataclasses.field(default_factory=ConstraintModelConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
    max_epochs: int = 30
    num_gpus: int = 1
    mixed_precision: bool = False
    training_seed: Optional[int] = 42  # Seed for reproducibility
    # Test-only mode
    test_only: bool = False  # If True, only run test with checkpoint
    checkpoint_path: Optional[str] = None  # Path to checkpoint for test_only mode


class PrimitiveModel(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        max_token_length: int,
        embed_dim: int,
        fc_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0,
    ):
        super().__init__()

        # Embeddings
        # Value embeddings
        num_val_embeddings = len(PrimitiveToken) + num_bins + 1  # +1 is for the void coord of arc.center
        if primitive_data.INCLUDE_CONSTRUCTION:
            num_val_embeddings += 2
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim, padding_idx=PrimitiveToken.Pad)

        # Coordinate embeddings (use PrimitiveParamToken size for semantic param types)
        num_coord_embeddings = len(PrimitiveParamToken)
        self.coord_embed = torch.nn.Embedding(num_coord_embeddings, embed_dim, padding_idx=PrimitiveParamToken.Pad)

        # Position embeddings
        # max(pos) = number_of_entities + 4 (Pad=0, Mask=1, Start=2, Stop)
        # number_of_entities ≈ max_token_length / 5 (avg ~5 tokens per entity)
        num_pos_embeddings = max_token_length // 5 + 4
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim, padding_idx=PrimitiveToken.Pad)

        # Rotary
        self.rotary_emb = dit.Rotary(embed_dim // num_heads)

        # Transformer Blocks
        self.blocks = torch.nn.ModuleList([
            dit.DDiTPlainBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # Final Norm
        self.norm = dit.LayerNorm(embed_dim)

    def forward(self, src: ConstraintTokenInput) -> torch.Tensor:
        # Embed
        val_embeddings = self.val_embed(src['val'])
        coord_embeddings = self.coord_embed(src['coord'])
        pos_embeddings = self.pos_embed(src['pos'])
        x = val_embeddings + coord_embeddings + pos_embeddings

        # Rotary
        rotary_cos_sin = self.rotary_emb(x)

        # Blocks
        # Using bfloat16 for blocks as per DiT design
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin)
            x = self.norm(x)

        return x


class ConstraintMDDWithCntModel(torch.nn.Module):
    """Masked discrete diffusion model of constraints conditioned on primitives.

    Includes explicit Count Token prediction using a hybrid head architecture.
    """
    def __init__(self,
                 num_bins: int,
                 max_token_length: int,
                 embed_dim: int,
                 fc_size: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0,
                 cond_dim: int | None = None,
                 max_constraint_count: int = 32,
                 num_fixed_tokens: int = len(ConstraintToken),
                ):
        super().__init__()
        # 1. Primitive Model
        self.prim_model = PrimitiveModel(num_bins, max_token_length, embed_dim,
            fc_size, num_heads, num_layers, dropout)

        # 2. Fixed Embeddings (Pad, Mask, Start, Stop, Coincident, ...)
        self.num_fixed_tokens = num_fixed_tokens
        self.val_embed = torch.nn.Embedding(self.num_fixed_tokens, embed_dim,
            padding_idx=ConstraintToken.Pad)

        # 3. Count Embedder and Prediction Head
        self.max_constraint_count = max_constraint_count
        self.count_embedder = dit.TimestepEmbedder(embed_dim)
        self.count_head = torch.nn.Linear(embed_dim, max_constraint_count + 1)

        # 4. DiT Backbone
        if cond_dim is None:
            cond_dim = embed_dim

        dit_config = {
            'model': {
                'hidden_size': embed_dim,
                'n_heads': num_heads,
                'n_blocks': num_layers,
                'dropout': dropout,
                'cond_dim': cond_dim,
                'scale_by_sigma': False
            }
        }
        # Cross-Attention: Target(Constraints) attends to Memory(Primitives + Fixed)
        self.dit_backbone = dit.CrossAttentionFeatureDIT(dit_config, output_size=embed_dim)
        self.sigma_map = dit.TimestepEmbedder(cond_dim)
        self.embed_dim = embed_dim

        # Special Token IDs
        self.mask_token_id = ConstraintToken.Mask
        self.pad_token_id = ConstraintToken.Pad

    def _embed_tokens(self, src: ConstraintTokenInput):
        """Embed primitive and constraint tokens.

        Returns:
            c_embeddings: Constraint embeddings [B, L_c, D], where [B, 1:2, D] is the count embedding
            p_embeddings: Primitive embeddings with prepended constraint type tokens [B, L_p, D]
                          Used as cross-attention memory and pointer network vocabulary
        """
        # A. Embed Primitives [B, L_p, D]
        prim_embeddings = self.prim_model(src)

        # B. Embed Fixed Tokens [B, F, D]
        batch_size = src['c_val'].shape[0]
        fixed_indices = torch.arange(self.num_fixed_tokens, device=src['c_val'].device)
        fixed_indices = fixed_indices.unsqueeze(0).expand(batch_size, -1)
        fixed_val_embeddings = self.val_embed(fixed_indices)

        # C. Construct Pointer Memory [Fixed Tokens | Primitive Tokens]
        # This serves as both cross-attention memory and pointer vocabulary
        # NOTE: This ordering is important so that it aligns with the constraint tokenization
        # NOTE: Don't include the Count value in the Pointer Memory
        p_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], dim=1)

        # D. Embed Input Sequence (Query)
        # c_val: [B, L_c] (Index 0=Start, Index 1=COUNT, Index 2+=Constraints)
        c_val_ptr = src['c_val'].clone()

        # D-1. Gather Pointer-oriented embeddings
        # Index 1 (COUNT position) contains the MASK index or a raw integer value + self.num_fixed_tokens.
        # The latter is irrelevant to the Memory position, so we temporarily fill it by 1 (Mask).
        c_val_ptr[:, 1] = self.mask_token_id

        # Gather [B, L_c, D]
        c_embeddings = torch.gather(p_embeddings, 1, c_val_ptr.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # D-2. Inject Count Embedding (Index 1)
        # If the input is a valid number, inject the Sinusoidal Embedding.
        # If the input is a MASK, use the Mask Embedding (Fixed) directly.
        count_with_offset_val = src['c_val'][:, 1].clone() # [B]
        is_numeric = (count_with_offset_val != self.mask_token_id) & (count_with_offset_val != self.pad_token_id)

        if is_numeric.any():
            valid_count_values = (count_with_offset_val[is_numeric].float() - self.num_fixed_tokens) / self.max_constraint_count
            c_embeddings[is_numeric, 1, :] = self.count_embedder(valid_count_values).to(c_embeddings.dtype)

        return c_embeddings, p_embeddings

    @torch.amp.autocast('cuda', torch.bfloat16)
    def forward(self, src: ConstraintTokenInput, sigma: torch.Tensor):
        # c_embeddings: [B, L_c, D] (Count is Sinusoidal-embedded)
        # p_embeddings: [B, F+P, D] (Pure Pointer Memory)
        # L_c = 1(Start) + 1(COUNT) + N*4(Slots) + 1(End)
        c_embeddings, p_embeddings = self._embed_tokens(src)

        # Conditioning on sigma (timestep)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        c = F.silu(self.sigma_map(sigma))

        # Cross-Attention DiT: c_embeddings attend to p_embeddings
        # c_embeddings: target (queries), p_embeddings: memory (keys/values)
        c_out = self.dit_backbone(c_embeddings, p_embeddings, c)

        # --- Hybrid Output Head ---

        # NOTE: INDEX 1 IS COUNT, SO IT SHOULDN'T POINT ANY CONSTRAINT TYPES OR PRIMITIVES.
        c_out_count = c_out[:, 1]
        c_out_without_count = torch.cat([c_out[:, 0:1], c_out[:, 2:]], dim=1)

        # 1. Pointer Head (for Constraints)
        # # [B, L_c, D] @ [B, F+P, D]^T -> [B, L_c, F+P]
        logits_ptr = torch.matmul(c_out_without_count, p_embeddings.to(c_out.dtype).transpose(1, 2))

        # 2. Classification Head (for Count)
        # [B, D] -> [B, max_count+1]
        logits_cnt = self.count_head(c_out_count)

        return logits_ptr, logits_cnt


################################################################


class LogLinearNoise(torch.nn.Module):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

    def forward(self, t):
        # Assume time goes from 0 to 1
        return self.total_noise(t), self.rate_noise(t)

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

    def importance_sampling_transformation(self, t):
        f_T = torch.log1p(- torch.exp(- self.sigma_max))
        f_0 = torch.log1p(- torch.exp(- self.sigma_min))
        sigma_t = - torch.log1p(- torch.exp(t * f_T + (1 - t) * f_0))
        t = - torch.expm1(- sigma_t) / (1 - self.eps)
        return t


class ConstraintMDDWithCntModule(L.LightningModule):
    """Main constraint masked discrete diffusion model implementation.
    """
    hparams: ConstraintModelTrainingConfig

    def __init__(self, config: ConstraintModelTrainingConfig):
        super().__init__()

        self.save_hyperparameters(config)

        self.num_fixed_tokens = len(ConstraintToken)
        self.model = ConstraintMDDWithCntModel(
            num_bins=self.hparams.data.num_position_bins,
            max_token_length=self.hparams.data.max_token_length,
            embed_dim=self.hparams.model.net.embedding_dimension,
            fc_size=self.hparams.model.net.hidden_size,
            num_heads=self.hparams.model.net.num_heads,
            num_layers=self.hparams.model.net.num_layers,
            cond_dim=self.hparams.model.net.embedding_dimension,
            max_constraint_count=self.hparams.data.max_token_length//CONSTRAINT_TOKEN_LEN,
            num_fixed_tokens=self.num_fixed_tokens,
        )

        if isinstance(self.hparams.model.token_masking_strategies, str):
            self.hparams.model.token_masking_strategies = [self.hparams.model.token_masking_strategies]
        if isinstance(self.hparams.model.token_masking_probs, (str, float)):
            self.hparams.model.token_masking_probs = [self.hparams.model.token_masking_probs]
        if isinstance(self.hparams.model.gt_reordering_strategies, str):
            self.hparams.model.gt_reordering_strategies = [self.hparams.model.gt_reordering_strategies]
        assert len(self.hparams.model.token_masking_strategies) == \
            len(self.hparams.model.token_masking_probs) == \
            len(self.hparams.model.gt_reordering_strategies)
        self.token_masking = TokenMasking(mask_id=ConstraintToken.Mask)
        self.token_reordering = TokenReordering(
            token_num_per_block=CONSTRAINT_TOKEN_LEN,
            prefix_len=2)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad, reduction='none')

        # num_pointer_targets = p_embeddings vocabulary size = len(ConstraintToken) + max_token_length
        num_pointer_targets = len(ConstraintToken) + self.hparams.data.max_token_length

        # Metrics (0 => normal data, 1 => fully constrained data)
        # Used for both step-by-step and full-generation evaluation
        # Call _reset_metrics() before each use context (train epoch, val epoch, val_gen, test, etc.)
        self.kappa_metric = torch.nn.ModuleList([CohenKappa(num_pointer_targets) for _ in range(2)])
        self.accuracy_metric = torch.nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets) for _ in range(2)])
        self.set_metrics = torch.nn.ModuleList(
            [ConstraintIoU(token_num_per_block=CONSTRAINT_TOKEN_LEN, prefix_len=2) for _ in range(2)]
        )

        # >>> PRISM >>>
        self.noise = LogLinearNoise()
        self.sampling_eps = 0.001
        self.inference_sampling_steps = config.sampling.steps

        self.ProbPath = MaskProbPath(self, config.sampling, self.noise)
        # <<< PRISM <<<

    def _format_constraint_sequence_with_cnt(self, tokens, have_count_offset: bool, max_show: int | None = None) -> list[str]:
        """Format constraint tokens for display."""
        result = []
        for i, tok in enumerate(tokens):
            if isinstance(tok, str):
                result.append(tok)
                continue

            tok_val = int(tok)

            if i == 1:  # COUNT token
                if have_count_offset:
                    result.append(f"[COUNT {tok_val - self.num_fixed_tokens}]" if tok_val != ConstraintToken.Mask else "[MASK]")
                else:
                    result.append(f"[COUNT {tok_val}]")
                continue

            if tok_val == ConstraintToken.Mask:
                result.append("[MASK]")
            elif tok_val == ConstraintToken.Pad:
                break
            elif tok_val == ConstraintToken.Start:
                result.append("[START]")
            elif tok_val == ConstraintToken.Stop:
                result.append("[STOP]")
                break
            elif tok_val == ConstraintToken.InnerPad:
                result.append("[PAD]")
            elif tok_val < len(ConstraintToken):
                result.append(ConstraintToken(tok_val).name)
            else:
                result.append(f"R{tok_val - len(ConstraintToken)}")
        if max_show and len(result) > max_show:
            return result[:max_show] + [f"... (+{len(result) - max_show} more)"]
        return result

    def configure_optimizers(self):
        base_lr = self.hparams.optim.learning_rate * self.hparams.data.batch_size / 128

        optim = torch.optim.AdamW(self.model.parameters(), lr=base_lr)

        if self.hparams.data.dataset_size is None:
            raise ValueError('config.data.dataset_size must be specified to create optimizers!')

        steps_per_epoch = (self.hparams.data.dataset_size + self.hparams.data.batch_size - 1) // self.hparams.data.batch_size

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=base_lr,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=steps_per_epoch)

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optim], [scheduler_config]

    def forward(self, batch: ConstraintTokenInput, sigma: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(batch, sigma)

    def _compute_loss(
        self,
        batch: ConstraintTokenInput,
        masking_strategy: str,
        masking_prob: float | str,
        gt_reordering_strategy: str,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Model expects the following input format: [Start, [Count], Coincident, Ref1, Ref2, IPad, ...]

        2-Pass Training:
        - Pass 1: Fill inactive blocks with GT-masked
        - Pass 2: Fill inactive blocks with Pass 1's detached predictions (with same mask pattern)
        This addresses train/inference domain gap where inactive blocks accumulate coherent predictions.
        """
        # Prepare input for model
        target = batch["c_val"]
        target_cnt = batch["c_cnt"]
        target_with_offset_cnt = torch.cat([
            batch["c_val"][:, 0:1], self.num_fixed_tokens + target_cnt.reshape(-1, 1), batch['c_val'][:, 1:]
        ], dim=1)

        # Sample t
        batch_size = target.shape[0]
        _eps_0 = 1e-3  # TODO: MAGIC NUMBER
        if isinstance(masking_prob, str):
            _eps_t = torch.rand((batch_size,), device=target.device)
            if 'antithetic' in masking_prob:
                offset = torch.arange(batch_size, device=target.device) / batch_size
                _eps_t = (_eps_t / batch_size + offset) % 1
            t = (1 - _eps_0) * _eps_t + _eps_0
            if 'importance' in masking_prob:
                t = self.noise.importance_sampling_transformation(t)
            sigma, dsigma = self.noise(t)
            move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise
        elif isinstance(masking_prob, float):
            move_chance = torch.tensor([masking_prob] * batch_size, device=target.device)
            t = move_chance / (1 - _eps_0)  # move_chance = (1-eps) * t for LogLinearNoise
            sigma, dsigma = self.noise(t)
        else:
            raise ValueError(f"Invalid masking_prob: {masking_prob}")

        # Masking
        c_val_with_offset_cnt_masked, mask_with_cnt = self.token_masking(
            tokens=target_with_offset_cnt.clone(),
            mask_prob=move_chance,
            mask_start_pos=1,  # [START]
            prefix_len=2,      # [START, COUNT]
            strategy=masking_strategy)

        # ---------------------------------------------------------------------
        # Dummy Filling Logic
        # ---------------------------------------------------------------------
        B, L_total = c_val_with_offset_cnt_masked.shape
        block_num = (L_total - 2) // CONSTRAINT_TOKEN_LEN

        c_val_active_block_num = batch["c_cnt"].reshape(B, 1)  # [B, 1]
        # assert (c_val_active_block_num > 0).all(), "No active blocks found."  # FIXME

        block_indices = torch.arange(block_num, device=target.device).unsqueeze(0)  # [1, N]
        inactive_mask = (block_indices >= c_val_active_block_num) * (c_val_active_block_num > 0).expand(-1, block_num)  # [B, N]

        # ---------------------------------------------------------------------
        # PASS 1: Fill inactive with GT-masked
        # ---------------------------------------------------------------------
        c_val_masked_blocks_p1 = c_val_with_offset_cnt_masked[:, 2 : 2 + block_num * CONSTRAINT_TOKEN_LEN].reshape(B, block_num, CONSTRAINT_TOKEN_LEN)

        # Randomly sample from Active slots to fill Inactive slots
        rand_indices = (torch.rand(B, block_num, device=target.device) * c_val_active_block_num).long()
        gather_indices = rand_indices.unsqueeze(-1).expand(-1, -1, CONSTRAINT_TOKEN_LEN)
        sampled_active_blocks = torch.gather(c_val_masked_blocks_p1, 1, gather_indices)

        # Apply to Inactive slots only
        c_val_masked_blocks_p1 = c_val_masked_blocks_p1.clone()
        c_val_masked_blocks_p1[inactive_mask] = sampled_active_blocks[inactive_mask]

        # Write back for Pass 1
        c_val_with_offset_cnt_masked_p1 = c_val_with_offset_cnt_masked.clone()
        c_val_with_offset_cnt_masked_p1[:, 2 : 2 + block_num * CONSTRAINT_TOKEN_LEN] = c_val_masked_blocks_p1.reshape(B, -1)

        # Batch creation for Pass 1
        batch_masked_p1 = ConstraintTokenInput(
            val=batch['val'], coord=batch['coord'], pos=batch['pos'],
            c_val=c_val_with_offset_cnt_masked_p1,
        )

        # Model Forward (Pass 1)
        output_logits_ptr_p1, output_logits_cnt_p1 = self.model.forward(batch_masked_p1, sigma)

        # Postprocess: Forbid invalid tokens globally
        invalid_token_ids = [ConstraintToken.Pad, ConstraintToken.Mask, ConstraintToken.Start, ConstraintToken.Stop]
        output_logits_ptr_p1[:, :, invalid_token_ids] = float('-inf')

        # ---------------------------------------------------------------------
        # PASS 2: Fill inactive with Pass 1's detached predictions
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # Get Pass 1 predictions (sample from distribution to match inference behavior)
            # NOTE: We only need constraint predictions, not COUNT (inactive_mask only covers constraint blocks)
            preds_ptr_p1 = torch.distributions.Categorical(logits=output_logits_ptr_p1).sample()  # [B, L_ptr]

            # Re-apply mask pattern to constraint blocks only
            mask_constraint_blocks = mask_with_cnt[:, 2 : 2 + block_num * CONSTRAINT_TOKEN_LEN]
            preds_constraint_blocks_p1_masked = torch.where(
                mask_constraint_blocks,
                ConstraintToken.Mask,
                preds_ptr_p1[:, 1 : 1 + block_num * CONSTRAINT_TOKEN_LEN]
            )

        # Extract blocks and fill inactive with Pass 1's predictions
        c_val_masked_blocks_p2 = c_val_with_offset_cnt_masked[:, 2 : 2 + block_num * CONSTRAINT_TOKEN_LEN].reshape(B, block_num, CONSTRAINT_TOKEN_LEN)
        preds_blocks_p1 = preds_constraint_blocks_p1_masked.reshape(B, block_num, CONSTRAINT_TOKEN_LEN)

        c_val_masked_blocks_p2 = c_val_masked_blocks_p2.clone()
        c_val_masked_blocks_p2[inactive_mask] = preds_blocks_p1[inactive_mask]

        # Write back for Pass 2
        c_val_with_offset_cnt_masked_p2 = c_val_with_offset_cnt_masked.clone()
        c_val_with_offset_cnt_masked_p2[:, 2 : 2 + block_num * CONSTRAINT_TOKEN_LEN] = c_val_masked_blocks_p2.reshape(B, -1)

        # Batch creation for Pass 2
        batch_masked_p2 = ConstraintTokenInput(
            val=batch['val'], coord=batch['coord'], pos=batch['pos'],
            c_val=c_val_with_offset_cnt_masked_p2,
        )

        # Model Forward (Pass 2)
        output_logits_ptr_p2, output_logits_cnt_p2 = self.model.forward(batch_masked_p2, sigma)
        output_logits_ptr_p2[:, :, invalid_token_ids] = float('-inf')

        # ---------------------------------------------------------------------
        # Loss Calculation (Combined from both passes)
        # ---------------------------------------------------------------------

        # NOTE: output_logits_ptr and target don't contain the COUNT token, so 1+ rather than 2+
        B, L = target.shape
        invalid_tokens = (1 + target_cnt * CONSTRAINT_TOKEN_LEN).reshape(B, 1) <= torch.arange(L, device=target.device).reshape(1, L)

        # Prepare PAD one-hot vector
        pad_onehot = torch.full_like(output_logits_ptr_p1[0:1, 0:1, :], float('-inf'))
        pad_onehot[:, :, ConstraintToken.Stop] = 0.0

        # Overwrite inactive constraint blocks (for both passes)
        output_logits_ptr_forced_p1 = torch.where(invalid_tokens.unsqueeze(-1), pad_onehot, output_logits_ptr_p1)
        output_logits_ptr_forced_p2 = torch.where(invalid_tokens.unsqueeze(-1), pad_onehot, output_logits_ptr_p2)
        target_forced = torch.where(invalid_tokens, ConstraintToken.Stop, target)

        # Matching (same for both passes since GT target is the same)
        mask_ptr = torch.cat([mask_with_cnt[:, 0:1], mask_with_cnt[:, 2:]], dim=1)
        target_reordered, mask_for_loss_ptr = self.token_reordering(
            target_forced,
            output_logits_ptr_forced_p1,  # Use Pass 1 logits for matching (output_logits_ptr_forced_p2 should also be used)
            mask_ptr,
            gt_reordering_strategy,
            prefix_len=1,  # NOTE: target doesn't contain COUNT token, only START as prefix
        )

        # Loss reweighting
        if gt_reordering_strategy == "hungarian_frontier":
            reweighting_ptr = torch.ones_like(t).unsqueeze(-1).expand(-1, target.size(1))
            reweighting_cnt = torch.ones_like(t)
        else:
            reweighting_ptr = 1 / t.unsqueeze(-1).expand(-1, target.size(1))
            reweighting_cnt = 1 / t

        # Pointer loss (average of both passes)
        loss_ptr_p1 = self.criterion(
            output_logits_ptr_forced_p1[mask_for_loss_ptr],
            target_reordered[mask_for_loss_ptr],
        ) * reweighting_ptr[mask_for_loss_ptr]
        loss_ptr_p1 = loss_ptr_p1.sum() / mask_for_loss_ptr.sum().clamp(min=1)

        loss_ptr_p2 = self.criterion(
            output_logits_ptr_forced_p2[mask_for_loss_ptr],
            target_reordered[mask_for_loss_ptr],
        ) * reweighting_ptr[mask_for_loss_ptr]
        loss_ptr_p2 = loss_ptr_p2.sum() / mask_for_loss_ptr.sum().clamp(min=1)

        loss_ptr = (loss_ptr_p1 + loss_ptr_p2) / 2

        # Count loss (average of both passes)
        mask_for_loss_cnt = c_val_with_offset_cnt_masked[:, 1] == ConstraintToken.Mask
        loss_cnt_p1 = self.criterion(
            output_logits_cnt_p1[mask_for_loss_cnt],
            target_cnt[mask_for_loss_cnt],
        ) * reweighting_cnt[mask_for_loss_cnt]
        loss_cnt_p1 = loss_cnt_p1.sum() / mask_for_loss_cnt.sum().clamp(min=1)

        loss_cnt_p2 = self.criterion(
            output_logits_cnt_p2[mask_for_loss_cnt],
            target_cnt[mask_for_loss_cnt],
        ) * reweighting_cnt[mask_for_loss_cnt]
        loss_cnt_p2 = loss_cnt_p2.sum() / mask_for_loss_cnt.sum().clamp(min=1)

        loss_cnt = (loss_cnt_p1 + loss_cnt_p2) / 2

        # Pred (use Pass 2 predictions as they are more representative of inference)
        with torch.no_grad():
            preds_ptr = torch.distributions.Categorical(logits=output_logits_ptr_p2).sample()
            preds_cnt = torch.distributions.Categorical(logits=output_logits_cnt_p2).sample()

            B, L = preds_ptr.shape
            invalid_tokens = (1 + preds_cnt * CONSTRAINT_TOKEN_LEN).reshape(B, 1) <= torch.arange(L, device=target.device).reshape(1, L)
            preds_ptr = torch.where(
                invalid_tokens,
                ConstraintToken.Stop,
                preds_ptr
            )

            preds_with_cnt = torch.cat([preds_ptr[:, 0:1], preds_cnt.reshape(-1, 1), preds_ptr[:, 1:]], dim=1)
            target_reordered_with_cnt = torch.cat([target_reordered[:, 0:1], target_cnt.reshape(-1, 1), target_reordered[:, 1:]], dim=1)

        return (loss_ptr, loss_cnt), preds_with_cnt, target_reordered_with_cnt, mask_with_cnt

    def _compute_metrics(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        ignore_pred_tokens_after_stop: bool,
        dataloader_idx: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute metrics and return sorted sequences.

        Args:
            preds: Predicted tokens [B, L]
            target: Ground truth tokens [B, L]
            mask: Boolean mask [B, L] indicating which positions were masked in preds, or None for full generation
            ignore_pred_tokens_after_stop: Whether to ignore predictions after STOP token
            dataloader_idx: 0 for normal, 1 for FC validation

        Returns:
            Tuple of:
                - Dict with 'kappa', 'accuracy', 'iou', 'precision', 'recall' (and 'masked_acc', 'unmasked_acc' if mask provided)
                - preds_sorted: Canonically sorted predictions
                - target_sorted: Canonically sorted targets
        """
        canonicalize_strategy = "canonical_with_tail_strip" if ignore_pred_tokens_after_stop else "canonical_full"
        preds_sorted, _ = self.token_reordering(preds, None, None, canonicalize_strategy)
        target_sorted, _ = self.token_reordering(target, None, None, "canonical_with_tail_strip")

        # identify valid tokens
        first_stop_idx = self.token_reordering.get_first_stop_idx(target_sorted)
        valid_token_mask = torch.arange(preds_sorted.shape[1], device=preds_sorted.device).reshape(1, -1) <= first_stop_idx.reshape(-1, 1)

        kappa = self.kappa_metric[dataloader_idx]
        accuracy = self.accuracy_metric[dataloader_idx]
        set_metric = self.set_metrics[dataloader_idx]

        # forward() updates internal state AND returns per-batch value
        kappa_val = kappa(preds_sorted[valid_token_mask], target_sorted[valid_token_mask])
        accuracy_val = accuracy(preds_sorted[valid_token_mask], target_sorted[valid_token_mask])
        set_metrics_val = set_metric(preds_sorted, target_sorted)  # Returns {"iou": ..., "precision": ..., "recall": ...}

        metrics = {
            "kappa": kappa_val,
            "accuracy": accuracy_val,
            **set_metrics_val,  # iou, precision, recall (per-batch)
        }

        # Count Metrics
        metrics["count_acc"] = (preds[:, 1] == target[:, 1]).float().mean()
        metrics["count_mae"] = (preds[:, 1].float() - target[:, 1].float()).abs().mean()

        # Compute masked/unmasked accuracy if mask is provided
        if mask is not None:
            masked_relevant = mask * valid_token_mask
            unmasked_relevant = (~mask) * valid_token_mask
            metrics["masked_acc"] = (preds_sorted[masked_relevant] == target_sorted[masked_relevant]).float().mean()
            metrics["unmasked_acc"] = (preds_sorted[unmasked_relevant] == target_sorted[unmasked_relevant]).float().mean()

        return metrics, preds_sorted, target_sorted

    def _compute_all_strategies(self, batch: ConstraintTokenInput, dataloader_idx: int = 0) -> tuple[dict[str, list[torch.Tensor]], dict[str, list]]:
        loss_lists = {"ptr": [], "cnt": []}
        metrics: dict[str, list] = {}

        for masking_strategy, masking_prob, reordering_strategy in zip(
            self.hparams.model.token_masking_strategies,
            self.hparams.model.token_masking_probs,
            self.hparams.model.gt_reordering_strategies):

            (loss_ptr, loss_cnt), preds, target_reordered, mask = self._compute_loss(
                batch, masking_strategy, masking_prob, reordering_strategy)

            step_metrics, _, _ = self._compute_metrics(
                preds, target_reordered, mask,
                ignore_pred_tokens_after_stop=("hungarian" not in reordering_strategy),
                dataloader_idx=dataloader_idx)

            loss_lists["ptr"].append(loss_ptr)
            loss_lists["cnt"].append(loss_cnt)
            for key, value in step_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

        return loss_lists, metrics

    def _reset_metrics(self):
        """Reset all metrics to prevent accumulation across epochs."""
        for kappa in self.kappa_metric:
            kappa.reset()
        for acc in self.accuracy_metric:
            acc.reset()
        for sm in self.set_metrics:
            sm.reset()

    def on_train_epoch_start(self):
        self._reset_metrics()

    def on_validation_epoch_start(self):
        self._reset_metrics()

    def training_step(self, batch: ConstraintTokenInput, batch_idx: int):
        loss_lists, metrics = self._compute_all_strategies(batch)
        total_loss = sum(sum(l) for l in loss_lists.values())

        self.log('train/loss', total_loss)
        for i, (loss_ptr_per_strategy, loss_cnt_per_strategy) in enumerate(zip(loss_lists["ptr"], loss_lists["cnt"])):
            self.log(f'train/loss/{i}', loss_ptr_per_strategy, prog_bar=True)
            self.log(f'train/loss/{i}_cnt', loss_cnt_per_strategy, prog_bar=True)

        for metric_key, metric_values in metrics.items():
            for i, metric_per_strategy in enumerate(metric_values):
                self.log(f'train/{metric_key}/{i}', metric_per_strategy, prog_bar=True)

        return total_loss

    def validation_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss_lists, metrics = self._compute_all_strategies(batch, dataloader_idx=dataloader_idx)
        total_loss = sum(sum(l) for l in loss_lists.values())

        suffix = '_fc' if dataloader_idx == 1 else ''

        self.log(f'validation/loss{suffix}', total_loss, add_dataloader_idx=False, sync_dist=True)
        for i, (loss_ptr_per_strategy, loss_cnt_per_strategy) in enumerate(zip(loss_lists["ptr"], loss_lists["cnt"])):
            self.log(f'validation/loss{suffix}/{i}', loss_ptr_per_strategy, add_dataloader_idx=False, sync_dist=True)
            self.log(f'validation/loss{suffix}/{i}_cnt', loss_cnt_per_strategy, add_dataloader_idx=False, sync_dist=True)

        for metric_key, metric_values in metrics.items():
            for i, metric_per_strategy in enumerate(metric_values):
                self.log(f'validation/{metric_key}{suffix}/{i}', metric_per_strategy, add_dataloader_idx=False, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        - Load a batch from each validation dataloader
        - Generate samples using the full denoising process
        - Compute generation metrics for each dataloader
        """
        val_loaders = self.trainer.val_dataloaders
        if val_loaders is None:
            return

        # Handle single dataloader vs list
        if not isinstance(val_loaders, list):
            val_loaders = [val_loaders]

        # Reset metrics before generation evaluation
        self._reset_metrics()

        for loader_idx, loader in enumerate(val_loaders):
            try:
                val_iter = next(iter(loader))
            except StopIteration:
                continue

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in val_iter.items()}

            # Generate samples
            start_tokens = [ConstraintToken.Start]
            samples_with_offset_cnt, intermediates_with_offset_cnt = self.ProbPath.generate_sample(
                batch=batch,
                num_steps=self.inference_sampling_steps,
                start_tokens=start_tokens,
                N_tokenwise_unmasking=CONSTRAINT_TOKEN_LEN if f"{CONSTRAINT_TOKEN_LEN}-tokenwise" in self.hparams.model.token_masking_strategies else 1,
            )

            # remove offset
            samples_with_cnt = samples_with_offset_cnt.clone()
            samples_with_cnt[:, 1] -= self.num_fixed_tokens
            assert torch.all(samples_with_cnt[:, 1]) >= 0, "COUNT token must be unmasked in the final result."

            target = batch['c_val']
            target_cnt = batch['c_cnt'].reshape(-1, 1)
            target_with_cnt = torch.cat([target[:, :1], target_cnt, target[:, 1:]], dim=1)
            del target, target_cnt

            # overwrite inactive constraint blocks
            B, L = samples_with_cnt.shape
            invalid_tokens = (2 + samples_with_cnt[:, 1] * CONSTRAINT_TOKEN_LEN).reshape(B, 1) <= torch.arange(L, device=samples_with_cnt.device).reshape(1, L)
            samples_with_cnt[invalid_tokens] = ConstraintToken.Stop

            B, L = target_with_cnt.shape
            invalid_tokens = (2 + target_with_cnt[:, 1] * CONSTRAINT_TOKEN_LEN).reshape(B, 1) <= torch.arange(L, device=target_with_cnt.device).reshape(1, L)
            target_with_cnt[invalid_tokens] = ConstraintToken.Stop

            # Compute metrics
            ignore_pred_tokens_after_stop = all(
                "hungarian" not in masking_strategy  # subword search
                for masking_strategy in self.hparams.model.token_masking_strategies
            )
            metrics, samples_sorted, target_sorted = self._compute_metrics(
                samples_with_cnt, target_with_cnt, mask=None,
                ignore_pred_tokens_after_stop=ignore_pred_tokens_after_stop,
                dataloader_idx=loader_idx,
            )

            # Log metrics
            suffix = '_fc' if loader_idx == 1 else ''
            for metric_name, metric_value in metrics.items():
                self.log(f'validation_gen/{metric_name}{suffix}', metric_value, add_dataloader_idx=False, sync_dist=True)

            # Print samples
            if self.global_rank == 0:
                display_label_gt = "GT"
                display_label_gen = "Gen"

                print("\n" + "="*80)
                print(f"VALIDATION{suffix.upper()} SAMPLE COMPARISON")
                print("="*80)

                for i in range(min(5, samples_with_cnt.size(0))):
                    print(f"\n[Sample {i}] Full constraint sequence:")

                    # Show intermediate denoising steps
                    if intermediates_with_offset_cnt:
                        print(f"\n  Denoising progression ({len(intermediates_with_offset_cnt)} steps):")
                        steps_to_show = list(range(0, len(intermediates_with_offset_cnt), max(1, len(intermediates_with_offset_cnt) // 5)))
                        for step_idx in steps_to_show:
                            step_tokens = intermediates_with_offset_cnt[step_idx][i].cpu().numpy()
                            formatted = self._format_constraint_sequence_with_cnt(step_tokens, have_count_offset=True, max_show=None)
                            print(f"    Step {step_idx:3d}: {formatted}")

                    # Show the final result and GT
                    print("  " + "-"*40)
                    print(f"  {display_label_gen:15}: {self._format_constraint_sequence_with_cnt(samples_sorted[i].cpu().numpy(), have_count_offset=False)}")
                    print(f"  {display_label_gt:15}: {self._format_constraint_sequence_with_cnt(target_sorted[i].cpu().numpy(), have_count_offset=False)}")

                print("="*80 + "\n")

    def test_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss_lists, metrics = self._compute_all_strategies(batch, dataloader_idx=dataloader_idx)
        total_loss = sum(sum(l) for l in loss_lists.values())

        suffix = '_fc' if dataloader_idx == 1 else ''

        self.log(f'test/loss{suffix}', total_loss)
        for i, (loss_ptr_per_strategy, loss_cnt_per_strategy) in enumerate(zip(loss_lists["ptr"], loss_lists["cnt"])):
            self.log(f'test/loss{suffix}/{i}', loss_ptr_per_strategy, add_dataloader_idx=False, sync_dist=True)
            self.log(f'test/loss{suffix}/{i}_cnt', loss_cnt_per_strategy, add_dataloader_idx=False, sync_dist=True)

        for metric_key, metric_values in metrics.items():
            for i, metric_per_strategy in enumerate(metric_values):
                self.log(f'test/{metric_key}{suffix}/{i}', metric_per_strategy, add_dataloader_idx=False, sync_dist=True)

    def on_test_epoch_end(self):
        """
        - Load a batch from each test dataloader
        - Generate samples using the full denoising process
        - Compute generation metrics for each dataloader
        """
        test_loaders = self.trainer.test_dataloaders
        if test_loaders is None:
            return

        # Handle single dataloader vs list
        if not isinstance(test_loaders, list):
            test_loaders = [test_loaders]

        # Reset metrics before generation evaluation
        self._reset_metrics()

        for loader_idx, loader in enumerate(test_loaders):
            try:
                val_iter = next(iter(loader))
            except StopIteration:
                continue

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in val_iter.items()}

            # Generate samples
            start_tokens = [ConstraintToken.Start]
            samples_with_offset_cnt, intermediates_with_offset_cnt = self.ProbPath.generate_sample(
                batch=batch,
                num_steps=self.inference_sampling_steps,
                start_tokens=start_tokens,
                N_tokenwise_unmasking=CONSTRAINT_TOKEN_LEN if f"{CONSTRAINT_TOKEN_LEN}-tokenwise" in self.hparams.model.token_masking_strategies else 1,
            )

            target = batch['c_val']
            target_cnt = batch['c_cnt'].reshape(-1, 1)
            target_with_cnt = torch.cat([target[:, :1], target_cnt, target[:, 1:]], dim=1)
            del target, target_cnt

            # Compute metrics
            ignore_pred_tokens_after_stop = all(
                "hungarian" not in masking_strategy  # subword search
                for masking_strategy in self.hparams.model.token_masking_strategies
            )
            metrics, samples_sorted, target_sorted = self._compute_metrics(
                samples_with_cnt, target_with_cnt, mask=None,
                ignore_pred_tokens_after_stop=ignore_pred_tokens_after_stop,
                dataloader_idx=loader_idx,
            )

            # Log metrics
            suffix = '_fc' if loader_idx == 1 else ''
            for metric_name, metric_value in metrics.items():
                self.log(f'test_gen/{metric_name}{suffix}', metric_value, add_dataloader_idx=False, sync_dist=True)

            # Print samples
            if self.global_rank == 0:
                display_label_gt = "GT"
                display_label_gen = "Gen"

                print("\n" + "="*80)
                print(f"TEST{suffix.upper()} SAMPLE COMPARISON")
                print("="*80)

                for i in range(min(5, samples_with_cnt.size(0))):
                    print(f"\n[Sample {i}] Full constraint sequence:")

                    # Show intermediate denoising steps
                    if intermediates:
                        print(f"\n  Denoising progression ({len(intermediates)} steps):")
                        steps_to_show = list(range(0, len(intermediates), max(1, len(intermediates) // 5)))
                        for step_idx in steps_to_show:
                            # Apply canonical sorting for clearer display (but keep MASK tokens visible)
                            ignore_pred_tokens_after_stop = all(
                                "hungarian" not in reordering_strategy  # subword search
                                for reordering_strategy in self.hparams.model.gt_reordering_strategies
                            )
                            step_sorted, _ = self.token_reordering(
                                intermediates[step_idx], None, None,
                                "canonical_with_tail_strip" if ignore_pred_tokens_after_stop else "canonical_full",
                            )
                            step_tokens = step_sorted[i].cpu().numpy()
                            formatted = self._format_constraint_sequence_with_cnt(step_tokens, have_count_offset=True, max_show=None)
                            print(f"    Step {step_idx:3d}: {formatted}")

                    # Show the final result and GT
                    print("  " + "-"*40)
                    print(f"  {display_label_gen:15}: {self._format_constraint_sequence_with_cnt(samples_sorted[i].cpu().numpy(), have_count_offset=False)}")
                    print(f"  {display_label_gt:15}: {self._format_constraint_sequence_with_cnt(target_sorted[i].cpu().numpy(), have_count_offset=False)}")

                print("="*80 + "\n")

            # # Solver-based evaluation (rank 0 only, expensive)
            # if self.global_rank == 0:
            #     from .solver_feedback import evaluate_constraints_batch, HAS_FREECAD
            #     if HAS_FREECAD:
            #         solver_results = evaluate_constraints_batch(
            #             batch['val'],
            #             samples_with_cnt,
            #             num_bins=self.hparams.data.num_position_bins,
            #             fully_constrained_dof=2,
            #         )
            #         if solver_results:
            #             print(f"Solver Report (test{suffix}): {solver_results}")
