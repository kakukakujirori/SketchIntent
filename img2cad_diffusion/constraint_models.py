"""Main pytorch-lightning models for training constraint problem.
"""
import dataclasses

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from img2cad_diffusion import primitive_data
from sketchgraphs_models.nn.summary import CohenKappa
from . import dit
from .constraint_data import ConstraintDataConfig, ConstraintTokenInput
from .detokenization import reconstruct_constraints
from .losses import ConstraintLoss, ConstraintIoU
from .prob_path import MaskProbPath
from .tokens import PrimitiveToken, PrimitiveParamToken, ConstraintToken, ConstraintParamToken


# Function overload to not need 'modules'
def generate_square_subsequent_mask(source: torch.Tensor) -> torch.Tensor:
    return source.new_full((source.shape[0], source.shape[0]), fill_value=float('-inf')).triu_(diagonal=1)


@dataclasses.dataclass
class ConstraintModelConfig:
    """Configuration for the main constraint model.

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
class OptimConfig:
    learning_rate: float = 1e-4
    gradient_clip_norm: float | None = 1.0


@dataclasses.dataclass
class ConstraintModelTrainingConfig:
    """Training configuration for constraint models."""
    data: ConstraintDataConfig = dataclasses.field(default_factory=ConstraintDataConfig)
    model: ConstraintModelConfig = dataclasses.field(default_factory=ConstraintModelConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    batch_size: int = 256
    num_data_workers: int = 8
    max_epochs: int = 30
    num_gpus: int = 1
    mixed_precision: bool = False
    training_seed: int | None = 42  # Seed for reproducibility
    use_hungarian_loss: bool = False  # Use Hungarian matching for order-agnostic training


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
        num_val_embeddings = len(PrimitiveToken) + num_bins
        if primitive_data.INCLUDE_CONSTRUCTION:
            num_val_embeddings += 2
        if primitive_data.INCLUDE_CLOCKWISE:
            num_val_embeddings += 2  # clockwise True/False tokens for Arc
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


class ConstraintMDDModel(torch.nn.Module):
    """Masked discrete diffusion model of constraints conditioned on primitives.

    Uses cross-attention architecture where constraints attend to primitives,
    providing a stronger inductive bias for learning constraint-primitive relationships.
    """
    def __init__(self,
                 num_bins: int,
                 max_token_length: int,
                 embed_dim: int,
                 fc_size: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0,
                 cond_dim: int | None = None):
        super().__init__()
        # Primitive model (for dynamic embeddings)
        self.prim_model = PrimitiveModel(num_bins, max_token_length, embed_dim,
            fc_size, num_heads, num_layers, dropout)
        # Value embeddings: constraint tokens + reference tokens (prim positions up to max_length)
        # Reference tokens are: len(ConstraintToken) + prim_token_position
        num_val_embeddings = len(ConstraintToken)
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim,
            padding_idx=ConstraintToken.Pad)

        # DiT Backbone with Cross-Attention
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
        # Use CrossAttentionFeatureDIT: c_embeddings attend to p_embeddings
        self.dit_backbone = dit.CrossAttentionFeatureDIT(dit_config, output_size=embed_dim)
        self.sigma_map = dit.TimestepEmbedder(cond_dim)
        self.embed_dim = embed_dim

    def _embed_tokens(self, src: dict[str, torch.Tensor]):
        """Embed primitive and constraint tokens.

        Returns:
            c_embeddings: Constraint embeddings [B, L_c, D]
            p_embeddings: Primitive embeddings with prepended constraint type tokens [B, L_p, D]
                          Used as cross-attention memory and pointer network vocabulary
        """
        # Embed primitives
        prim_embeddings = self.prim_model(src)

        # Prepend fixed val embeddings (constraint types) - these form the pointer vocabulary
        batch_size = src['c_val'].shape[0]
        fixed_tokens = np.tile(range(len(ConstraintToken)), (batch_size, 1))
        fixed_tokens = torch.tensor(fixed_tokens).to(src['c_val'].device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)

        # p_embeddings: [constraint type tokens | primitive embeddings]
        # This serves as both cross-attention memory and pointer vocabulary
        # NOTE: This ordering is important so that it aligns with the constraint tokenization
        p_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], 1)

        # Embed constraint tokens (the noisy input)
        # c_val contains indices into p_embeddings:
        #   - 0 to len(ConstraintToken)-1: constraint type tokens
        #   - len(ConstraintToken) to len(ConstraintToken)+max_prim_len-1: primitive references
        # Tokenization ensures c_val never exceeds p_embeddings.shape[1]
        c_val_indices = src['c_val']

        # Gather embeddings from p_embeddings using c_val as indices
        # NOTE: The most crucial part - this implements the pointer mechanism
        c_embeddings = torch.gather(p_embeddings, 1, c_val_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        return c_embeddings, p_embeddings

    def forward(self, src: dict[str, torch.Tensor], sigma: torch.Tensor):
        # Embed tokens
        c_embeddings, p_embeddings = self._embed_tokens(src)

        # Conditioning on sigma (timestep)
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        c = F.silu(self.sigma_map(sigma))

        # Cross-attention DiT: c_embeddings attend to p_embeddings
        # c_embeddings: target (queries), p_embeddings: memory (keys/values)
        c_out = self.dit_backbone(c_embeddings, p_embeddings, c)

        # Pointer dot-product: Match against p_embeddings (the vocabulary)
        # p_embeddings shape: [B, L_p, H] -> transpose to [B, H, L_p]
        logits = torch.matmul(c_out, p_embeddings.to(c_out.dtype).transpose(1, 2))

        return logits


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


class ConstraintMDDModule(L.LightningModule):
    """Main constraint masked discrete diffusion model implementation.
    """
    hparams: ConstraintModelTrainingConfig

    def __init__(self, config: ConstraintModelTrainingConfig):
        super().__init__()

        self.save_hyperparameters(config)


        # Ignoring redundant comments about embedding dimension mapping
        self.model = ConstraintMDDModel(
            num_bins=self.hparams.data.num_position_bins,
            max_token_length=self.hparams.data.max_token_length,
            embed_dim=self.hparams.model.embedding_dimension,
            fc_size=self.hparams.model.hidden_size,
            num_heads=self.hparams.model.num_heads,
            num_layers=self.hparams.model.num_layers,
            cond_dim=self.hparams.model.embedding_dimension
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)

        # num_pointer_targets = p_embeddings vocabulary size = len(ConstraintToken) + max_token_length
        num_pointer_targets = len(ConstraintToken) + self.hparams.data.max_token_length

        # Metrics for training (dataloader_idx=0)
        self._kappa_metric = CohenKappa(num_pointer_targets)
        self._accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets)
        self._iou_metric = ConstraintIoU()

        # Metrics for FC validation (dataloader_idx=1)
        self._kappa_metric_fc = CohenKappa(num_pointer_targets)
        self._accuracy_metric_fc = torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets)
        self._iou_metric_fc = ConstraintIoU()

        # Metrics for full generation (on_validation_epoch_end) - normal validation
        self._kappa_metric_gen = CohenKappa(num_pointer_targets)
        self._accuracy_metric_gen = torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets)
        self._iou_metric_gen = ConstraintIoU()

        # Metrics for full generation (on_validation_epoch_end) - FC validation
        self._kappa_metric_gen_fc = CohenKappa(num_pointer_targets)
        self._accuracy_metric_gen_fc = torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets)
        self._iou_metric_gen_fc = ConstraintIoU()

        # >>> PRISM >>>
        self.noise = LogLinearNoise()
        self.antithetic_sampling = True
        self.importance_sampling = False
        self.sampling_eps = 0.001
        self.inference_sampling_steps = 50

        self.ProbPath = MaskProbPath(self, config, self.noise)
        # <<< PRISM <<<

        # Loss function (supports Hungarian matching)
        self.loss_fn = ConstraintLoss(use_hungarian=getattr(config, 'use_hungarian_loss', False))

    def _canonicalize_for_comparison(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canonicalize sequences for comparison and compute relevant indices.

        For Hungarian mode: sorts constraint blocks by min reference.
        Computes relevant_idxs based on padding type (STOP or PAD).

        Returns:
            (preds_canonical, target_canonical, relevant_idxs)
        """
        from .tokenization import CONSTRAINT_TOKEN_LEN

        B, L = target.shape
        block_size = CONSTRAINT_TOKEN_LEN  # 4
        n_blocks = (L - 1) // block_size  # Exclude START token


        def sort_by_min_ref(tokens: torch.Tensor) -> torch.Tensor:
            """Sort constraint blocks by minimum reference index.

            Only sorts valid constraint blocks (before first STOP token).
            Everything at or after the first STOP is left unchanged.
            """
            # Find first STOP position in entire sequence (not just block boundaries)
            is_stop = (tokens == ConstraintToken.Stop)
            is_stop_padded = F.pad(is_stop, (0, 1), value=True)
            first_stop_pos = is_stop_padded.long().argmax(dim=1)  # [B]

            # Calculate which block contains the first STOP (0-indexed)
            # Block i covers positions [1 + i*block_size, 1 + (i+1)*block_size)
            first_stop_block = (first_stop_pos - 1) // block_size  # [B]
            first_stop_block = first_stop_block.clamp(min=0, max=n_blocks)

            # Extract blocks: [B, L] -> [B, N, 4] (skip START at position 0)
            blocks = tokens[:, 1:1 + n_blocks * block_size].reshape(B, n_blocks, block_size)

            # Get min ref for each block (refs are tokens >= num_constraint_tokens)
            refs = blocks[:, :, 1:]  # [B, N, 3] - skip constraint type
            num_c_tokens = len(ConstraintToken)
            is_ref = refs >= num_c_tokens
            ref_values = torch.where(is_ref, refs - num_c_tokens, torch.tensor(9999, device=tokens.device))
            min_ref = ref_values.min(dim=-1).values  # [B, N]

            # Mask out blocks at or after first STOP so they stay in place
            block_range = torch.arange(n_blocks, device=tokens.device).unsqueeze(0)  # [1, N]
            is_valid_block = block_range < first_stop_block.unsqueeze(1)  # [B, N]
            # Give invalid blocks their original position as sort key (so they stay put)
            min_ref = torch.where(is_valid_block, min_ref, block_range.float() + 10000)

            # Sort blocks by min_ref
            sorted_indices = min_ref.argsort(dim=-1)  # [B, N]
            sorted_blocks = torch.gather(blocks, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, block_size))

            # Reconstruct: [START] + sorted_blocks + remaining
            sorted_tokens = tokens.clone()
            sorted_tokens[:, 1:1 + n_blocks * block_size] = sorted_blocks.reshape(B, -1)
            return sorted_tokens

        # Sort if Hungarian mode (order-agnostic comparison)
        if self.loss_fn.use_hungarian:
            preds_canonical = sort_by_min_ref(preds)
            target_canonical = sort_by_min_ref(target)
        else:
            preds_canonical = preds
            target_canonical = target

        # Compute relevant_idxs based on padding type
        if self.hparams.data.use_stop_padding:
            # With STOP padding: include positions up to and including first STOP
            # Add sentinel at end so argmax always finds a STOP
            is_stop = (target_canonical == ConstraintToken.Stop)
            is_stop_padded = F.pad(is_stop, (0, 1), value=True)  # [B, L+1]
            first_stop_pos = is_stop_padded.long().argmax(dim=1)  # [B]
            pos_range = torch.arange(L, device=target_canonical.device).unsqueeze(0)
            relevant_idxs = pos_range <= first_stop_pos.unsqueeze(1)
        else:
            # With PAD padding: exclude all Pad tokens
            relevant_idxs = (target_canonical != ConstraintToken.Pad)

        return preds_canonical, target_canonical, relevant_idxs

    @staticmethod
    def _format_constraint_sequence(tokens, max_show: int | None = None) -> list[str]:
        """Format constraint tokens for display."""
        result = []
        for tok in tokens:
            tok_val = int(tok)
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
        base_lr = self.hparams.optim.learning_rate * self.hparams.batch_size / 128

        optim = torch.optim.AdamW(self.model.parameters(), lr=base_lr)

        if self.hparams.data.dataset_size is None:
            raise ValueError('config.data.dataset_size must be specified to create optimizers!')

        steps_per_epoch = (self.hparams.data.dataset_size + self.hparams.batch_size - 1) // self.hparams.batch_size

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

    def _sample_t(self, b: int, device: torch.device, _eps_0: float = 0.001):
        _eps_t = torch.rand(b, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(b, device=device) / b
            _eps_t = (_eps_t / b + offset) % 1
        t = (1 - _eps_0) * _eps_t + _eps_0
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _compute_loss(self, batch: ConstraintTokenInput):

        t = self._sample_t(batch["c_val"].shape[0], batch["c_val"].device, _eps_0=self.sampling_eps)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise

        # Prepare input for model
        target = batch["c_val"]

        # Apply noise (masking) to constraint tokens only
        # Generate mask indices once and apply to all constraint tensors
        mask_indices = torch.rand_like(batch["c_val"], dtype=torch.float32) < move_chance
        c_val_masked = torch.where(mask_indices, ConstraintToken.Mask, batch["c_val"])

        # Create masked batch (primitives unchanged, constraints masked)
        masked_batch = ConstraintTokenInput(
            val = batch['val'],
            coord = batch['coord'],
            pos = batch['pos'],
            c_val = c_val_masked,
        )

        # model forward pass
        output = self.model.forward(masked_batch, sigma)  # [B, L, C] -> logits

        # compute the loss--the reweighting term = 1/t for LogLinearNoise
        reweighting = 1 / t.unsqueeze(-1).expand(-1, target.size(1))
        loss = self.loss_fn.forward(output, target, mask_indices, reweighting)

        with torch.no_grad():
            preds = torch.distributions.Categorical(logits=output).sample()

        return loss, preds, target, mask_indices

    def _compute_metrics(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask_indices: torch.Tensor,
        dataloader_idx: int = 0,
    ) -> tuple[CohenKappa, ConstraintIoU, torch.Tensor, torch.Tensor]:
        """Update metrics and compute accuracies for the given dataloader.

        Args:
            preds: Predicted tokens [B, L]
            target: Ground truth tokens [B, L]
            mask_indices: Boolean mask indicating which positions were masked [B, L]
            dataloader_idx: 0 for normal, 1 for FC validation

        Returns:
            (kappa_metric, iou_metric, masked_acc, unmasked_acc)
        """
        preds_for_metric, target_for_metric, relevant_idxs = self._canonicalize_for_comparison(preds, target)

        if dataloader_idx == 0:
            kappa = self._kappa_metric
            accuracy = self._accuracy_metric
            iou = self._iou_metric
        else:
            kappa = self._kappa_metric_fc
            accuracy = self._accuracy_metric_fc
            iou = self._iou_metric_fc

        kappa(preds_for_metric[relevant_idxs], target_for_metric[relevant_idxs])
        accuracy(preds_for_metric[relevant_idxs], target_for_metric[relevant_idxs])
        iou(preds, target)  # IoU works on full sequences (constraint-level)

        # Compute masked/unmasked accuracy
        masked_relevant = mask_indices & relevant_idxs
        unmasked_relevant = (~mask_indices) & relevant_idxs
        masked_acc = (preds_for_metric[masked_relevant] == target_for_metric[masked_relevant]).float().mean()
        unmasked_acc = (preds_for_metric[unmasked_relevant] == target_for_metric[unmasked_relevant]).float().mean()

        return kappa, iou, masked_acc, unmasked_acc

    def training_step(self, batch: ConstraintTokenInput, batch_idx: int):
        loss, preds, target, mask_indices = self._compute_loss(batch)
        kappa, iou, masked_acc, unmasked_acc = self._compute_metrics(preds, target, mask_indices, dataloader_idx=0)

        self.log('loss', loss)
        self.log('kappa', kappa, prog_bar=True)
        self.log('acc_masked', masked_acc, prog_bar=True)
        self.log('acc_unmasked', unmasked_acc, prog_bar=True)

        return loss

    def validation_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target, mask_indices = self._compute_loss(batch)
        kappa, iou, masked_acc, unmasked_acc = self._compute_metrics(preds, target, mask_indices, dataloader_idx=dataloader_idx)

        # Prefix based on dataloader_idx
        # idx=0: normal validation, idx=1: FC validation
        prefix = 'validation' if dataloader_idx == 0 else 'validation_fc'

        self.log(f'{prefix}/loss', loss, add_dataloader_idx=False)
        self.log(f'{prefix}/kappa', kappa, add_dataloader_idx=False)
        self.log(f'{prefix}/iou', iou, add_dataloader_idx=False)
        self.log(f'{prefix}/acc_masked', masked_acc, add_dataloader_idx=False)
        self.log(f'{prefix}/acc_unmasked', unmasked_acc, add_dataloader_idx=False)

    def on_validation_epoch_end(self):
        """
        - Load a batch from each validation dataloader
        - Generate samples using the full denoising process
        - Compute kappa_gen, accuracy_gen, iou_gen for each dataloader
        """
        val_loaders = self.trainer.val_dataloaders
        if val_loaders is None:
            return

        # Handle single dataloader vs list
        if not isinstance(val_loaders, list):
            val_loaders = [val_loaders]

        # Define metrics and prefixes for each dataloader
        dataloader_configs = [
            {
                'prefix': 'validation',
                'kappa': self._kappa_metric_gen,
                'accuracy': self._accuracy_metric_gen,
                'iou': self._iou_metric_gen,
                'print_samples': True,
            },
            {
                'prefix': 'validation_fc',
                'kappa': self._kappa_metric_gen_fc,
                'accuracy': self._accuracy_metric_gen_fc,
                'iou': self._iou_metric_gen_fc,
                'print_samples': True,
            },
        ]

        for loader_idx, (loader, config) in enumerate(zip(val_loaders, dataloader_configs)):
            try:
                val_iter = next(iter(loader))
            except StopIteration:
                continue

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in val_iter.items()}

            # Generate samples
            start_tokens = [ConstraintToken.Start]
            samples, intermediates = self.ProbPath.generate_sample(
                batch=batch, num_steps=self.inference_sampling_steps, start_tokens=start_tokens
            )

            # Evaluate
            target = batch['c_val']
            samples_for_metric, target_for_metric, relevant_idxs = self._canonicalize_for_comparison(samples, target)

            config['kappa'](samples_for_metric[relevant_idxs], target_for_metric[relevant_idxs])
            config['accuracy'](samples_for_metric[relevant_idxs], target_for_metric[relevant_idxs])
            config['iou'](samples, target)

            prefix = config['prefix']
            self.log(f'{prefix}/kappa_gen', config['kappa'])
            self.log(f'{prefix}/accuracy_gen', config['accuracy'])
            self.log(f'{prefix}/iou_gen', config['iou'], prog_bar=(loader_idx == 0))

            # Print samples only for first dataloader on rank 0
            if config['print_samples'] and self.global_rank == 0:
                display_label_gt = "GT (sorted)" if self.loss_fn.use_hungarian else "GT"
                display_label_gen = "Gen (sorted)" if self.loss_fn.use_hungarian else "Gen"

                print("\n" + "="*80)
                print(f"VALIDATION SAMPLE COMPARISON ({prefix}, rank 0)")
                print("="*80)

                for i in range(min(5, samples.size(0))):
                    print(f"\n[Sample {i}] Full constraint sequence:")

                    # Show intermediate denoising steps
                    if intermediates:
                        print(f"\n  Denoising progression ({len(intermediates)} steps):")
                        steps_to_show = list(range(0, len(intermediates), max(1, len(intermediates) // 5)))
                        for step_idx in steps_to_show:
                            step_tokens = intermediates[step_idx][i].cpu().numpy()
                            formatted = self._format_constraint_sequence(step_tokens, max_show=None)
                            print(f"    Step {step_idx:3d}: {formatted}")

                    # Show the final result and GT
                    print("  " + "-"*40)
                    print(f"  {display_label_gen:15}: {self._format_constraint_sequence(samples_for_metric[i].cpu().numpy())}")
                    print(f"  {display_label_gt:15}: {self._format_constraint_sequence(target_for_metric[i].cpu().numpy())}")

                print("="*80 + "\n")

    def test_step(self, batch: ConstraintTokenInput, batch_idx: int):
        raise NotImplementedError
        # loss = self._compute_loss(batch)

        # self.log('test/loss', loss)
        # self.log('test/kappa', self._kappa_metric)
        # self.log('test/accuracy', self._accuracy_metric)