"""Autoregressive baseline model for constraint generation."""
import dataclasses

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from sketchgraphs_models.nn.summary import CohenKappa
from .constraint_data import ConstraintTokenInput
from .constraint_models import PrimitiveModel, generate_square_subsequent_mask, ConstraintModelTrainingConfig
from .tokens import ConstraintToken


class ConstraintARModel(torch.nn.Module):
    """Autoregressive generative model of constraints conditioned on primitives.

    Primitives are first embedded via a non-masked PrimitiveModel.
    Constraints are sampled autoregressively using a causal transformer decoder.
    """
    def __init__(self,
                 num_bins: int,
                 max_token_length: int,
                 embed_dim: int,
                 fc_size: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0):
        super().__init__()
        # Primitive model (for dynamic embeddings)
        self.prim_model = PrimitiveModel(num_bins, max_token_length, embed_dim,
            fc_size, num_heads, num_layers, dropout)
        # Value embeddings (only fixed ones)
        num_val_embeddings = len(ConstraintToken)
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim,
            padding_idx=ConstraintToken.Pad)
        # Position embeddings
        num_pos_embeddings = 3 + max_token_length
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim,
            padding_idx=ConstraintToken.Pad)
        # Transformer decoder
        decoder_layers = torch.nn.TransformerDecoderLayer(embed_dim, num_heads, fc_size,
            dropout, batch_first=True)
        self.trans_decoder = torch.nn.TransformerDecoder(decoder_layers, num_layers)
        self.embed_dim = embed_dim

    def _embed_tokens(self, src: dict[str, torch.Tensor]):
        # Embed primitives
        prim_embeddings = self.prim_model(src)
        # Prepend fixed val embeddings (constraint types)
        batch_size = src['c_val'].shape[0]
        seq_len = src['c_val'].shape[1]
        fixed_tokens = np.tile(range(len(ConstraintToken)), (batch_size, 1))
        fixed_tokens = torch.tensor(fixed_tokens).to(src['c_val'].device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)
        prim_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], 1)

        # Gather embeddings for constraint tokens
        c_val_indices = src['c_val'].unsqueeze(-1).expand(-1, -1, self.embed_dim)
        val_embeddings = torch.gather(prim_embeddings, 1, c_val_indices)

        # Generate position indices internally (0, 1, 2, ..., seq_len-1)
        pos_indices = torch.arange(seq_len, device=src['c_val'].device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(pos_indices)
        embeddings = val_embeddings + pos_embeddings
        return embeddings, prim_embeddings

    def forward(self, src: dict[str, torch.Tensor]):
        # Embed
        c_embeddings, p_embeddings = self._embed_tokens(src)
        # Pass to transformer with causal mask
        # Note: generate_square_subsequent_mask expects shape[0] to be seq_len,
        # but c_embeddings is [B, S, D], so we need to use shape[1]
        seq_len = c_embeddings.shape[1]
        tgt_mask = c_embeddings.new_full((seq_len, seq_len), fill_value=float('-inf')).triu_(diagonal=1)
        output = self.trans_decoder(c_embeddings, p_embeddings, tgt_mask)
        # Pointer dot-product
        p_embeddings = p_embeddings.transpose(1, 2)
        output = torch.matmul(output, p_embeddings)
        return output


class ConstraintARModule(L.LightningModule):
    """Lightning module for autoregressive constraint model."""
    hparams: ConstraintModelTrainingConfig

    def __init__(self, config: ConstraintModelTrainingConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = ConstraintARModel(
            num_bins=self.hparams.data.num_position_bins,
            max_token_length=self.hparams.data.max_token_length,
            embed_dim=self.hparams.model.embedding_dimension,
            fc_size=self.hparams.model.hidden_size,
            num_heads=self.hparams.model.num_heads,
            num_layers=self.hparams.model.num_layers)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)

        num_pointer_targets = len(ConstraintToken) + self.hparams.data.max_token_length
        self._kappa_metric = CohenKappa(num_pointer_targets)
        self._accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets)

        # Separate metrics for generative evaluation (on_validation_epoch_end)
        self._kappa_metric_gen = CohenKappa(num_pointer_targets)
        self._accuracy_metric_gen = torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets)

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

    def forward(self, batch: ConstraintTokenInput) -> torch.Tensor:
        return self.model(batch)

    def _compute_loss(self, output: torch.Tensor, batch: ConstraintTokenInput) -> torch.Tensor:
        # AR: predict next token, so shift by 1
        output = output[:, :-1].reshape(-1, output.shape[-1])
        target = batch['c_val'][:, 1:].reshape(-1)

        preds = torch.distributions.Categorical(logits=output).sample()

        relevant_idxs = (target != ConstraintToken.Pad)
        self._kappa_metric(preds[relevant_idxs], target[relevant_idxs])
        self._accuracy_metric(preds[relevant_idxs], target[relevant_idxs])

        return self.criterion(output, target)

    def training_step(self, batch: ConstraintTokenInput, batch_idx: int):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('loss', loss)
        self.log('kappa', self._kappa_metric, prog_bar=True)

        return loss

    def validation_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        # Prefix based on dataloader_idx
        # idx=0: normal validation, idx=1: FC validation
        prefix = 'validation' if dataloader_idx == 0 else 'validation_fc'

        self.log(f'{prefix}/loss', loss, add_dataloader_idx=False)
        self.log(f'{prefix}/kappa', self._kappa_metric, add_dataloader_idx=False)

    def test_step(self, batch: ConstraintTokenInput, batch_idx: int):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)

        self.log('test/loss', loss)
        self.log('test/kappa', self._kappa_metric)
        self.log('test/accuracy', self._accuracy_metric)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Generate samples autoregressively from scratch and compute kappa_gen.

        This is the fair comparison metric with MDD's kappa_gen.
        """
        val_loader = self.trainer.val_dataloaders
        if val_loader is None:
            return

        # Handle list of dataloaders or single dataloader
        if isinstance(val_loader, list):
            val_loader = val_loader[0]

        val_iter = next(iter(val_loader))
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in val_iter.items()}

        # Get target and sequence length
        target = batch['c_val']
        batch_size = target.shape[0]
        max_len = target.shape[1]

        # Start with just START token, rest is zeros (will be filled)
        generated = torch.full((batch_size, max_len), ConstraintToken.Pad,
                               device=self.device, dtype=target.dtype)
        generated[:, 0] = ConstraintToken.Start

        # Autoregressive generation
        for pos in range(max_len - 1):
            # Create batch with current generated tokens
            gen_batch = {
                'val': batch['val'],
                'coord': batch['coord'],
                'pos': batch['pos'],
                'c_val': generated,
            }

            # Forward pass
            output = self.model(gen_batch)  # [B, L, V]

            # Sample next token at position pos
            next_token_logits = output[:, pos, :]  # [B, V]
            next_token = torch.distributions.Categorical(logits=next_token_logits).sample()

            # Fill in position pos+1
            generated[:, pos + 1] = next_token

            # Early stopping if all sequences have generated STOP
            if (generated == ConstraintToken.Stop).any(dim=1).all():
                break

        # Compute kappa_gen
        relevant_idxs = (target != ConstraintToken.Pad)
        self._kappa_metric_gen(generated[relevant_idxs], target[relevant_idxs])
        self._accuracy_metric_gen(generated[relevant_idxs], target[relevant_idxs])

        self.log('validation/kappa_gen', self._kappa_metric_gen)
        self.log('validation/accuracy_gen', self._accuracy_metric_gen)

        # Print comparison on rank 0
        if self.global_rank == 0:
            print("\n" + "="*80)
            print("AR VALIDATION SAMPLE COMPARISON (rank 0)")
            print("="*80)

            i = 0
            def format_seq(tokens):
                result = []
                for tok in tokens:
                    tok_val = int(tok)
                    if tok_val == ConstraintToken.Pad:
                        break
                    elif tok_val == ConstraintToken.Start:
                        result.append("[START]")
                    elif tok_val == ConstraintToken.Stop:
                        result.append("[STOP]")
                        break
                    elif tok_val == ConstraintToken.InnerPad:
                        result.append("[PAD]")
                    elif tok_val < len(ConstraintToken):
                        # Constraint type token (Coincident, Parallel, etc.)
                        result.append(ConstraintToken(tok_val).name)
                    else:
                        result.append(f"R{tok_val - len(ConstraintToken)}")
                return result

            print(f"\n[Sample {i}]:")
            print(f"  GT  : {format_seq(target[i].cpu().numpy())[:30]}...")
            print(f"  Gen : {format_seq(generated[i].cpu().numpy())[:30]}...")
            print("="*80 + "\n")
