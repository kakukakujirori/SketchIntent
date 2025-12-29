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
from .losses import ConstraintIoU
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

        # Metrics for step-by-step evaluation (0 => normal data, 1 => fully constrained data)
        self.kappa_metric = torch.nn.ModuleList([CohenKappa(num_pointer_targets) for _ in range(2)])
        self.accuracy_metric = torch.nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets) for _ in range(2)])
        self.iou_metric = torch.nn.ModuleList([ConstraintIoU() for _ in range(2)])

        # Metrics for full-generation task (0 => normal data, 1 => fully constrained data)
        self.kappa_metric_gen = torch.nn.ModuleList([CohenKappa(num_pointer_targets) for _ in range(2)])
        self.accuracy_metric_gen = torch.nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets) for _ in range(2)])
        self.iou_metric_gen = torch.nn.ModuleList([ConstraintIoU() for _ in range(2)])

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

    def _compute_loss(self, batch: ConstraintTokenInput) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss and return preds/target for metrics.

        Returns:
            (loss, preds, target) where preds and target are [B, L-1] (shifted for AR)
        """
        output = self.model(batch)
        B, L, V = output.shape

        # AR: predict next token, so shift by 1
        logits = output[:, :-1]  # [B, L-1, V]
        target = batch['c_val'][:, 1:]  # [B, L-1]

        loss = self.criterion(logits.reshape(-1, V), target.reshape(-1))

        # Sample predictions for metrics, excluding MASK token
        with torch.no_grad():
            logits_for_sample = logits.clone()
            logits_for_sample[:, :, ConstraintToken.Mask] = -torch.inf
            preds = torch.distributions.Categorical(logits=logits_for_sample).sample()  # [B, L-1]

        return loss, preds, target

    def _compute_metrics(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        dataloader_idx: int = 0,
    ) -> tuple[CohenKappa, ConstraintIoU, torch.Tensor]:
        """Update metrics for the given dataloader.

        Args:
            preds: Predicted tokens [B, L]
            target: Ground truth tokens [B, L]
            dataloader_idx: 0 for normal, 1 for FC

        Returns:
            (kappa_metric, iou_metric, accuracy)
        """
        kappa = self.kappa_metric[dataloader_idx]
        accuracy = self.accuracy_metric[dataloader_idx]
        iou = self.iou_metric[dataloader_idx]

        relevant_idxs = (target != ConstraintToken.Pad)
        kappa(preds[relevant_idxs], target[relevant_idxs])
        accuracy(preds[relevant_idxs], target[relevant_idxs])
        iou(preds, target)

        acc = (preds[relevant_idxs] == target[relevant_idxs]).float().mean()
        return kappa, iou, acc

    def training_step(self, batch: ConstraintTokenInput, batch_idx: int):
        loss, preds, target = self._compute_loss(batch)
        kappa, iou, acc = self._compute_metrics(preds, target, dataloader_idx=0)

        self.log('loss', loss)
        self.log('kappa', kappa, prog_bar=True)

        return loss

    def validation_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target = self._compute_loss(batch)
        kappa, iou, acc = self._compute_metrics(preds, target, dataloader_idx=dataloader_idx)

        # Use add_dataloader_idx=True (default) for side-by-side comparison in table
        self.log('validation/loss', loss)
        self.log('validation/kappa', kappa)
        self.log('validation/iou', iou)

    def test_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target = self._compute_loss(batch)
        kappa, iou, acc = self._compute_metrics(preds, target, dataloader_idx=dataloader_idx)

        # Use add_dataloader_idx=True (default) for side-by-side comparison in table
        self.log('test/loss', loss, sync_dist=True)
        self.log('test/kappa', kappa, sync_dist=True)
        self.log('test/iou', iou, sync_dist=True)
        self.log('test/accuracy', acc, sync_dist=True)

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

        # Compute metrics for validation (loader_idx=0)
        relevant_idxs = (target != ConstraintToken.Pad)
        kappa = self.kappa_metric_gen[0]
        accuracy = self.accuracy_metric_gen[0]
        iou = self.iou_metric_gen[0]
        kappa(generated[relevant_idxs], target[relevant_idxs])
        accuracy(generated[relevant_idxs], target[relevant_idxs])
        iou(generated, target)

        self.log('validation/kappa_gen', kappa)
        self.log('validation/accuracy_gen', accuracy)
        self.log('validation/iou_gen', iou)

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
            print(f"  GT  : {format_seq(target[i].cpu().numpy())}")
            print(f"  Gen : {format_seq(generated[i].cpu().numpy())}")
            print("="*80 + "\n")

    @torch.no_grad()
    def on_test_epoch_end(self):
        """Generate samples autoregressively and compute metrics for test dataloaders."""
        test_loaders = self.trainer.test_dataloaders
        if test_loaders is None:
            return

        # Handle single dataloader vs list
        if not isinstance(test_loaders, list):
            test_loaders = [test_loaders]

        for loader_idx, loader in enumerate(test_loaders):
            try:
                test_iter = next(iter(loader))
            except StopIteration:
                continue

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in test_iter.items()}

            # Get target and sequence length
            target = batch['c_val']
            batch_size = target.shape[0]
            max_len = target.shape[1]

            # Autoregressive generation
            generated = torch.full((batch_size, max_len), ConstraintToken.Pad,
                                   device=self.device, dtype=target.dtype)
            generated[:, 0] = ConstraintToken.Start

            for pos in range(max_len - 1):
                gen_batch = {
                    'val': batch['val'],
                    'coord': batch['coord'],
                    'pos': batch['pos'],
                    'c_val': generated,
                }
                output = self.model(gen_batch)
                next_token_logits = output[:, pos, :]
                next_token = torch.distributions.Categorical(logits=next_token_logits).sample()
                generated[:, pos + 1] = next_token

                if (generated == ConstraintToken.Stop).any(dim=1).all():
                    break

            # Compute metrics (use separate metrics per loader)
            relevant_idxs = (target != ConstraintToken.Pad)
            kappa = self.kappa_metric_gen[loader_idx]
            accuracy = self.accuracy_metric_gen[loader_idx]
            iou = self.iou_metric_gen[loader_idx]
            kappa(generated[relevant_idxs], target[relevant_idxs])
            accuracy(generated[relevant_idxs], target[relevant_idxs])
            iou(generated, target)

            # Use different names for each loader since on_test_epoch_end is called once
            fc_suffix = '' if loader_idx == 0 else '_fc'
            self.log(f'test_gen/kappa{fc_suffix}', kappa, sync_dist=True)
            self.log(f'test_gen/accuracy{fc_suffix}', accuracy, sync_dist=True)
            self.log(f'test_gen/iou{fc_suffix}', iou, sync_dist=True)

            # Print samples on rank 0
            if self.global_rank == 0:
                print("\n" + "="*80)
                loader_label = 'test' if loader_idx == 0 else 'test_fc'
                print(f"AR TEST SAMPLE COMPARISON ({loader_label}, rank 0)")
                print("="*80)

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
                            result.append(ConstraintToken(tok_val).name)
                        else:
                            result.append(f"R{tok_val - len(ConstraintToken)}")
                    return result

                for i in range(5):
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {format_seq(target[i].cpu().numpy())}")
                    print(f"  Gen : {format_seq(generated[i].cpu().numpy())}")
                    print("="*80 + "\n")

            # Solver-based evaluation (rank 0 only)
            if self.global_rank == 0:
                from .solver_feedback import evaluate_constraints_batch, HAS_FREECAD
                if HAS_FREECAD:
                    solver_results = evaluate_constraints_batch(
                        batch['val'],
                        generated,
                        num_bins=self.hparams.data.num_position_bins,
                        fully_constrained_dof=2,
                    )
                    if solver_results:
                        print(f"Solver Report ({loader_label}): {solver_results}")

