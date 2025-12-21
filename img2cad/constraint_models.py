"""Main pytorch-lightning models for training constraint problem.
"""

import dataclasses
from typing import Optional

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from img2cad import constraint_data, modules, lightning
from sketchgraphs_models.nn.summary import CohenKappa


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


class ConstraintModel(torch.nn.Module):
    """Autoregressive generative model of constraints conditioned on primitives.

    Primitives are first embedded via a non-masked PrimitiveModel.
    Constraints are sampled via a masked decoder, using the learned primitive
    embeddings to form dynamic constraint embeddings.
    """
    def __init__(self,
                 num_bins,
                 max_entities,
                 embed_dim,
                 fc_size,
                 num_heads,
                 num_layers,
                 dropout=0):
        super(ConstraintModel, self).__init__()
        # Primitive model (for dynamic embeddings)
        self.prim_model = modules.PrimitiveModel(num_bins, max_entities, embed_dim,
            fc_size, num_heads, num_layers, dropout, use_mask=False,
            linear_decode=False)
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
        # Transformer decoder
        decoder_layers = torch.nn.TransformerDecoderLayer(embed_dim, num_heads, fc_size,
            dropout)
        self.trans_decoder = torch.nn.TransformerDecoder(decoder_layers, num_layers)

    def _embed_tokens(self, src):
        # Embed primitives
        prim_embeddings = self.prim_model(src)
        # Prepend fixed val embeddings (constraint types)
        batch_size = src['c_val'].shape[0]
        fixed_tokens = np.tile(range(len(constraint_data.Token)), (batch_size, 1))  # [[0, 1, ..., N-1], [0, 1, ..., N-1], ...]
        fixed_tokens = torch.tensor(fixed_tokens).to(src['c_val'].device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)
        prim_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], 1)
        val_tokens = torch.unsqueeze(src['c_val'], 2).expand(
            -1, -1, prim_embeddings.shape[2])
        # Embed constraint tokens
        val_embeddings = torch.gather(prim_embeddings, 1, val_tokens)
        coord_embeddings = self.coord_embed(src['c_coord'])
        pos_embeddings = self.pos_embed(src['c_pos'])
        embeddings = val_embeddings + coord_embeddings + pos_embeddings
        return embeddings, prim_embeddings

    def _feed_transformer(self, c_embeddings, p_embeddings):
        # Transpose to match transformer dimensions
        c_embeddings = torch.transpose(c_embeddings, 0, 1)
        p_embeddings = torch.transpose(p_embeddings, 0, 1)
        # Pass to transformer
        tgt_mask = modules.generate_square_subsequent_mask(c_embeddings)
        output = self.trans_decoder(c_embeddings, p_embeddings, tgt_mask)
        return torch.transpose(output, 0, 1)

    def forward(self, src):
        # Embed
        c_embeddings, p_embeddings = self._embed_tokens(src)
        # Pass to transformer
        output = self._feed_transformer(c_embeddings, p_embeddings)
        # Pointer dot-product
        p_embeddings = torch.transpose(p_embeddings, 1, 2)
        output = torch.matmul(output, p_embeddings)
        return output



class ConstraintModule(L.LightningModule):
    """Main constraint model implementation.
    """
    hparams: ConstraintModelTrainingConfig

    def __init__(self, config: ConstraintModelTrainingConfig):
        super().__init__()

        self.save_hyperparameters(config)
        self.model = ConstraintModel(
            self.hparams.data.num_position_bins,
            self.hparams.model.num_positional_embeddings,
            self.hparams.model.embedding_dimension,
            self.hparams.model.hidden_size,
            self.hparams.model.num_heads,
            self.hparams.model.num_layers)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=constraint_data.Token.Pad)

        num_classes = len(constraint_data.Token) + config.data.max_token_length
        self._kappa_metric = CohenKappa(len(constraint_data.Token) + config.data.max_token_length)
        self._accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def configure_optimizers(self):
        base_lr = self.hparams.optim.learning_rate * self.hparams.batch_size / 128

        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr)

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

    def forward(self, batch: modules.TokenInput) -> torch.Tensor:
        return self.model(batch)

    def _compute_loss(self, output: torch.Tensor, batch: modules.TokenInput) -> torch.Tensor:
        output = output[:, :-1].reshape(-1, output.shape[-1])
        target = batch['c_val'][:, 1:].reshape(-1)

        preds = torch.distributions.Categorical(logits=output).sample()

        relevant_idxs = (target != constraint_data.Token.Pad)
        self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])
        self._accuracy_metric(preds[relevant_idxs], target[relevant_idxs])

        return self.criterion(output, target)

    def training_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('validation/loss', loss)
        self.log('validation/kappa', self._kappa_metric)

    def test_step(self, batch: modules.TokenInput, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('test/loss', loss)
        self.log('test/kappa', self._kappa_metric)
        self.log('test/accuracy', self._accuracy_metric)


################################################################


from . import dit

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
        self.prim_model = modules.PrimitiveModel(num_bins, max_entities, embed_dim,
            fc_size, num_heads, num_layers, dropout, use_mask=False,
            linear_decode=False)
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

        # Mask Token Embedding
        self.mask_token_embed = torch.nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

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

        # Mask Embedding
        mask_embedding = self.mask_token_embed.expand(batch_size, -1, -1)

        # p_embeddings contains [Mask Token, Fixed Tokens, Primitives]
        p_embeddings = torch.cat([mask_embedding, fixed_val_embeddings, prim_embeddings], 1)

        # Embed constraint tokens (the noisy input)
        # We need to gather the embeddings corresponding to the current tokens
        # Note: src['c_val'] contains INDICES into p_embeddings (which is conceptually the vocabulary)
        # BUT for the INPUT to the model, we embed them using the same embedding space.
        # Wait, if c_val are indices, we need to map them to embeddings.
        # The indices point to: 0..N-1 (Fixed), N.. (Primitive 1), etc.
        # We can gather from p_embeddings.
        c_val_indices = src['c_val']
        # Mask out padding or invalid indices to avoid gather error, though they should be valid.
        # Ensure indices are within range.

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

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=constraint_data.Token.Pad + 1)

        # Increment num_classes for Mask token
        num_classes = len(constraint_data.Token) + config.data.max_token_length + 1
        self._kappa_metric = CohenKappa(num_classes)
        self._accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        # >>> PRISM >>>
        self.mask_index = 0 # Mask token index is now 0
        self.noise = LogLinearNoise()
        self.antithetic_sampling = True
        self.importance_sampling = False
        self.sampling_eps = 0.001
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

    def forward(self, batch: modules.TokenInput) -> torch.Tensor:
        # For inference/generation, we likely need a different path or loop.
        # This forward is technically not used by training_step which calls self.model directly or implements logic?
        # The previous code called self.model(batch).
        # But now self.model.forward requires 'sigma'.
        # For simple forward without diffusion loop (e.g. debugging), we might fail.
        # Let's leave it for now or assume batch has everything.
        raise NotImplementedError("Use sample_loop or similar for generation.")

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

    def _compute_loss(self, batch: modules.TokenInput) -> torch.Tensor:

        t = self._sample_t(batch["c_val"].shape[0], batch["c_val"].device, _eps_0=self.sampling_eps)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise

        # Prepare input for model

        # Shift input indices to make room for Mask token
        # Mask is 0.
        # Original indices: 0..15 (Fixed), 16.. (Prims)
        # New indices: 0 (Mask), 1..16 (Fixed), 17.. (Prims shifted +1)
        # So we shift EVERYTHING by 1.
        raw_c_val = batch["c_val"]
        shifted_c_val = raw_c_val + 1

        # Apply noise (masking) to shifted indices
        xt, mask_indices = self._sample_xt_x0(shifted_c_val, move_chance)

        src_noisy = batch.copy()
        src_noisy['c_val'] = xt

        # model forward pass
        output = self.model.forward(src_noisy, sigma)  # [B, L, C] -> logits
        target = shifted_c_val  # Target is the shifted Clean indices

        # compute the loss--the reweighting term = 1/t for LogLinearNoise
        reweighting = 1 / t.unsqueeze(-1).expand(-1, target.size(1))
        loss_n = F.cross_entropy(output[mask_indices], target[mask_indices], reduction='none')
        loss_n = loss_n * reweighting[mask_indices]
        loss = loss_n.sum() / (target.shape[0] * target.shape[1])

        with torch.no_grad():
            preds = torch.distributions.Categorical(logits=output).sample()
            relevant_idxs = (target != (constraint_data.Token.Pad + 1))
            self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])
            self._accuracy_metric(preds[relevant_idxs], target[relevant_idxs])

        return loss

    def training_step(self, batch: modules.TokenInput, batch_idx):
        loss = self._compute_loss(batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch: modules.TokenInput, batch_idx):
        loss = self._compute_loss(batch)

        self.log('validation/loss', loss)
        self.log('validation/kappa', self._kappa_metric)

    def test_step(self, batch: modules.TokenInput, batch_idx):
        loss = self._compute_loss(batch)

        self.log('test/loss', loss)
        self.log('test/kappa', self._kappa_metric)
        self.log('test/accuracy', self._accuracy_metric)