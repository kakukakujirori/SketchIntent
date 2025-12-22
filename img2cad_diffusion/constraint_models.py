"""Main pytorch-lightning models for training constraint problem.
"""
from werkzeug.exceptions import NotImplemented

import dataclasses
from typing import Optional

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from img2cad_diffusion import constraint_data, lightning, dataset
from sketchgraphs_models.nn.summary import CohenKappa
from . import dit
from .prob_path import MaskProbPath

# Function overload to not need 'modules'
def generate_square_subsequent_mask(source: torch.Tensor) -> torch.Tensor:
    return source.new_full((source.shape[0], source.shape[0]), fill_value=float('-inf')).triu_(diagonal=1)


@dataclasses.dataclass
class ConstraintModelConfig:
    """Configuration for the main constraint model.

    Attributes
    ----------
    num_position_embeddings
        Maximum number of entities in the sketch
    embedding_dimension
        Dimension of embeddings used
    hidden_size
        Dimension of hidden layers
    num_heads
        Number of heads used in multi-head attention
    num_layers
        Number of transformer layers
    """
    num_positional_embeddings: int = 16
    embedding_dimension: int = 128
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 8

@dataclasses.dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    gradient_clip_norm: Optional[float] = 1.0


@dataclasses.dataclass
class ConstraintModelTrainingConfig(lightning.TrainingConfiguration):
    data: constraint_data.ConstraintDataConfig = dataclasses.field(default_factory=constraint_data.ConstraintDataConfig)
    model: ConstraintModelConfig = dataclasses.field(default_factory=ConstraintModelConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    batch_size: int = 256
    num_data_workers: int = 8


class PrimitiveBlock(torch.nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = dit.LayerNorm(dim)
        self.attn_qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = torch.nn.Linear(dim, dim, bias=False)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.norm2 = dit.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_ratio * dim, bias=True),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x, rotary_cos_sin):
        # x: [B, S, D]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Attention
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = dit.rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = dit.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        qkv = dit.rearrange(qkv, 'b s ... -> (b s) ...')

        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, step=seq_len,
            dtype=torch.int32, device=qkv.device
        )

        x = dit.flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False
        )

        x = dit.rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        x = self.attn_out(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = x + x_skip

        # MLP
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x_mlp = F.dropout(x_mlp, p=self.dropout_p, training=self.training)
        x = x + x_mlp

        return x


class PrimitiveModel(torch.nn.Module):
    def __init__(self, num_bins, max_entities, embed_dim, fc_size, num_heads, num_layers, dropout=0):
        super().__init__()

        # Embeddings
        # Value embeddings
        num_val_embeddings = len(dataset.Token) + num_bins
        if dataset.INCLUDE_CONSTRUCTION:
            num_val_embeddings += 2
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim, padding_idx=dataset.Token.Pad)

        # Coordinate embeddings
        num_coord_embeddings = 2 + sum([len(cm) for cm in dataset.COORD_TOKEN_MAP.values()])
        self.coord_embed = torch.nn.Embedding(num_coord_embeddings, embed_dim, padding_idx=dataset.Token.Pad)

        # Position embeddings
        num_pos_embeddings = 3 + max_entities
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim, padding_idx=dataset.Token.Pad)

        # Rotary
        self.rotary_emb = dit.Rotary(embed_dim // num_heads)

        # Transformer Blocks
        self.blocks = torch.nn.ModuleList([
            PrimitiveBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # Final Norm
        self.norm = dit.LayerNorm(embed_dim)

    def forward(self, src: constraint_data.TokenInput) -> torch.Tensor:
        # Embed
        val_embeddings = self.val_embed(src['val'])
        coord_embeddings = self.coord_embed(src['coord'])
        pos_embeddings = self.pos_embed(src['pos'])
        x = val_embeddings + coord_embeddings + pos_embeddings

        # Rotary
        rotary_cos_sin = self.rotary_emb(x)

        # Blocks
        # Using bfloat16 for blocks as per DiT design
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin)
            x = self.norm(x)

        return x


class ConstraintMDDModel(torch.nn.Module):
    """Masked discrete diffusion model of constraints conditioned on primitives.
    """
    def __init__(self,
                 num_bins,
                 max_entities,
                 embed_dim,
                 fc_size,
                 num_heads,
                 num_layers,
                 dropout=0,
                 cond_dim=None):
        super().__init__()
        # Primitive model (for dynamic embeddings)
        self.prim_model = PrimitiveModel(num_bins, max_entities, embed_dim,
            fc_size, num_heads, num_layers, dropout)
        pad_tok = constraint_data.Token.Pad
        # Value embeddings (only fixed ones)
        num_val_embeddings = len(constraint_data.Token)
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim,
            padding_idx=pad_tok)
        # Coordinate embeddings
        num_coord_embeddings = 2 + len(
            constraint_data.CONSTRAINT_COORD_TOKENS)
        self.coord_embed = torch.nn.Embedding(num_coord_embeddings, embed_dim,
            padding_idx=pad_tok)
        # Position embeddings
        num_pos_embeddings = 3 + (4 * max_entities)  # see make_sequence_dataset
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim,
            padding_idx=pad_tok)  # TODO: dry-ify overlapping logic w/ PrimModel

        # DiT Backbone
        # Use simple config dict for DIT
        if cond_dim is None:
            cond_dim = embed_dim

        dit_config = {
            'model': {
                'hidden_size': embed_dim,
                'n_heads': num_heads,
                'n_blocks': num_layers,
                'dropout': dropout,
                'cond_dim': cond_dim,
                'scale_by_sigma': False # Default
            }
        }
        # Output size is embed_dim because we will do dot product with embeddings later
        self.dit_backbone = dit.FeatureDIT(dit_config, output_size=embed_dim)
        self.sigma_map = dit.TimestepEmbedder(cond_dim)
        self.embed_dim = embed_dim

    def _embed_tokens(self, src):
        # Embed primitives
        prim_embeddings = self.prim_model(src)
        # Prepend fixed val embeddings (constraint types)
        batch_size = src['c_val'].shape[0]
        fixed_tokens = np.tile(range(len(constraint_data.Token)), (batch_size, 1))  # [[0, 1, ..., N-1], [0, 1, ..., N-1], ...]
        fixed_tokens = torch.tensor(fixed_tokens).to(src['c_val'].device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)

        # p_embeddings contains [Fixed Tokens (including Mask), Primitives]
        p_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], 1)

        # Embed constraint tokens (the noisy input)
        # c_val contains INDICES into p_embeddings
        c_val_indices = src['c_val']

        val_embeddings = torch.gather(p_embeddings, 1, c_val_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        coord_embeddings = self.coord_embed(src['c_coord'])
        pos_embeddings = self.pos_embed(src['c_pos'])
        c_embeddings = val_embeddings + coord_embeddings + pos_embeddings

        return c_embeddings, p_embeddings

    def forward(self, src, sigma):
        # Embed
        c_embeddings, p_embeddings = self._embed_tokens(src)

        # Concatenate for DiT: [Primitives, Constraints]
        # We want the model to attend to primitives.
        x = torch.cat([p_embeddings, c_embeddings], dim=1)

        # Conditioning on sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        c = F.silu(self.sigma_map(sigma))

        # Pass to DiT
        output = self.dit_backbone(x, c)

        # Separate output: we only care about the constraint part
        # output shape: [B, L_p + L_c, H]
        c_out = output[:, p_embeddings.shape[1]:, :]

        # Pointer dot-product: Match against p_embeddings (the dictionary)
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
            max_entities=self.hparams.model.num_positional_embeddings,
            embed_dim=self.hparams.model.embedding_dimension,
            fc_size=self.hparams.model.hidden_size,
            num_heads=self.hparams.model.num_heads,
            num_layers=self.hparams.model.num_layers,
            cond_dim=self.hparams.model.embedding_dimension
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=constraint_data.Token.Pad)

        # Increment num_classes for Mask token
        num_classes = len(constraint_data.Token) + config.data.max_token_length
        num_classes = len(constraint_data.Token) + config.data.max_token_length
        self._kappa_metric = CohenKappa(num_classes)
        self._accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        # Metrics for full generation
        self._kappa_metric_total = CohenKappa(num_classes)
        self._accuracy_metric_total = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        # >>> PRISM >>>
        self.mask_index = constraint_data.Token.Mask # Mask token index is now Token.Mask
        self.noise = LogLinearNoise()
        self.antithetic_sampling = True
        self.importance_sampling = False
        self.sampling_eps = 0.001
        self.inference_sampling_steps = 50

        self.ProbPath = MaskProbPath(self, config, self.mask_index, self.noise)
        # <<< PRISM <<<

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

    def forward(self, batch: constraint_data.TokenInput, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def _sample_xt_x0(self, x: torch.Tensor, move_chance: float | torch.Tensor):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size, diffusion_model_input_length), input.
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        mask_indices = (xt == self.mask_index)

        return xt, mask_indices

    def _compute_loss(self, batch: constraint_data.TokenInput) -> torch.Tensor:

        t = self._sample_t(batch["c_val"].shape[0], batch["c_val"].device, _eps_0=self.sampling_eps)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise

        # Prepare input for model
        target = batch["c_val"]

        # Apply noise (masking)
        xt, mask_indices = self._sample_xt_x0(target, move_chance)

        src_noisy = batch.copy()
        src_noisy['c_val'] = xt

        # model forward pass
        output = self.model.forward(src_noisy, sigma)  # [B, L, C] -> logits

        # compute the loss--the reweighting term = 1/t for LogLinearNoise
        reweighting = 1 / t.unsqueeze(-1).expand(-1, target.size(1))
        loss_n = F.cross_entropy(output[mask_indices], target[mask_indices], reduction='none')
        loss_n = loss_n * reweighting[mask_indices]
        loss = loss_n.sum() / (target.shape[0] * target.shape[1])

        with torch.no_grad():
            preds = torch.distributions.Categorical(logits=output).sample()
            relevant_idxs = (target != constraint_data.Token.Pad)
            self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])
            self._accuracy_metric(preds[relevant_idxs], target[relevant_idxs])

        return loss

    def training_step(self, batch: constraint_data.TokenInput, batch_idx: int):
        loss = self._compute_loss(batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch: constraint_data.TokenInput, batch_idx: int):
        loss = self._compute_loss(batch)

        self.log('validation/loss', loss)
        self.log('validation/kappa', self._kappa_metric)

    def on_validation_epoch_end(self):
        """
        - Load the first element of the validation dataloader
        - Mask all the constraint tokens
        - Run the masked discrete diffusion multiple times to recover all the masks
        - Compute the kappa metric, and log it
        """
        val_loader = self.trainer.val_dataloaders
        if val_loader is None:
             return

        # Handle list of dataloaders or single dataloader
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        val_iter = next(iter(val_loader))
        # Move batch to device
        batch: constraint_data.TokenInput = {k: v.to(self.device) for k, v in val_iter.items()} # type: ignore

        # Generate samples
        # generate_sample takes the batch (which contains primitives) and masks c_val internally
        samples, _ = self.ProbPath.generate_sample(batch=batch, num_steps=self.inference_sampling_steps, device=self.device)

        # Evaluate
        target = batch['c_val']
        relevant_idxs = (target != constraint_data.Token.Pad)

        # samples is the generated index tensor [B, L]
        # target is the ground truth [B, L]

        # Note: samples might contain Pad tokens if we allowed them to be predicted,
        # but target has Pad tokens.

        self._kappa_metric_total(samples[relevant_idxs], target[relevant_idxs])
        self._accuracy_metric_total(samples[relevant_idxs], target[relevant_idxs])

        self.log('validation/kappa_gen', self._kappa_metric_total)
        self.log('validation/accuracy_gen', self._accuracy_metric_total)

    def test_step(self, batch: constraint_data.TokenInput, batch_idx: int):
        raise NotImplementedError
        # loss = self._compute_loss(batch)

        # self.log('test/loss', loss)
        # self.log('test/kappa', self._kappa_metric)
        # self.log('test/accuracy', self._accuracy_metric)