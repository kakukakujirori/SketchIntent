"""Autoregressive baseline model for constraint generation."""
import dataclasses

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm

from sketchgraphs_models.nn.summary import CohenKappa
from .constraint_data import ConstraintTokenInput
from .constraint_models import PrimitiveModel, ConstraintModelTrainingConfig
from .iou import ConstraintIoU
from .tokens import ConstraintToken
from .tokenization import CONSTRAINT_TOKEN_LEN


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
            embed_dim=self.hparams.model.net.embedding_dimension,
            fc_size=self.hparams.model.net.hidden_size,
            num_heads=self.hparams.model.net.num_heads,
            num_layers=self.hparams.model.net.num_layers)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)

        num_pointer_targets = len(ConstraintToken) + self.hparams.data.max_token_length

        # Metrics for step-by-step evaluation (0 => normal data, 1 => fully constrained data)
        self.kappa_metric = torch.nn.ModuleList([CohenKappa(num_pointer_targets) for _ in range(2)])
        self.accuracy_metric = torch.nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets) for _ in range(2)])
        self.set_metrics = torch.nn.ModuleList(
            [ConstraintIoU(token_num_per_block=CONSTRAINT_TOKEN_LEN, token_prefix_len=1) for _ in range(2)]
        )

        # Metrics for full-generation task (0 => normal data, 1 => fully constrained data)
        self.kappa_metric_gen = torch.nn.ModuleList([CohenKappa(num_pointer_targets) for _ in range(2)])
        self.accuracy_metric_gen = torch.nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=num_pointer_targets) for _ in range(2)])
        self.set_metrics_gen = torch.nn.ModuleList(
            [ConstraintIoU(token_num_per_block=CONSTRAINT_TOKEN_LEN, token_prefix_len=1) for _ in range(2)]
        )

    @staticmethod
    def _format_constraint_sequence(tokens, max_show: int | None = None) -> list[str]:
        """Format constraint tokens for display."""
        result = []
        for tok in tokens:
            tok_val = int(tok)
            if tok_val == ConstraintToken.Mask:  # NOTE: Unused
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
        generated_from_scratch: bool = False,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Update metrics for the given dataloader.

        Args:
            preds: Predicted tokens [B, L]
            target: Ground truth tokens [B, L]
            generated_from_scratch: If True, use _gen metrics
            dataloader_idx: 0 for normal, 1 for FC

        Returns:
            Dict with 'kappa', 'accuracy', 'iou', 'precision', 'recall'
        """
        if generated_from_scratch:
            kappa = self.kappa_metric_gen[dataloader_idx]
            accuracy = self.accuracy_metric_gen[dataloader_idx]
            set_metric = self.set_metrics_gen[dataloader_idx]
        else:
            kappa = self.kappa_metric[dataloader_idx]
            accuracy = self.accuracy_metric[dataloader_idx]
            set_metric = self.set_metrics[dataloader_idx]

        relevant_idxs = (target != ConstraintToken.Pad)
        kappa(preds[relevant_idxs], target[relevant_idxs])
        accuracy(preds[relevant_idxs], target[relevant_idxs])
        set_metric(preds, target)  # Set metrics work on full sequences (constraint-level)

        metrics = {
            "kappa": kappa.compute(),
            "accuracy": accuracy.compute(),
            **set_metric.compute(),  # iou, precision, recall
        }

        return metrics

    def generate_sample(
        self,
        batch: ConstraintTokenInput,
        max_len: int,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        # Start with just START token, rest is zeros (will be filled)
        batch_size = batch['val'].shape[0]
        generated = torch.full((batch_size, max_len), ConstraintToken.Pad,
                            device=self.device, dtype=dtype)
        generated[:, 0] = ConstraintToken.Start

        # Autoregressive generation
        for pos in tqdm(range(max_len - 1), desc="Sampling"):
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
            next_token_logits[:, ConstraintToken.Mask] = -torch.inf
            next_token = torch.distributions.Categorical(logits=next_token_logits).sample()

            # Fill in position pos+1
            generated[:, pos + 1] = next_token

            # Early stopping if all sequences have generated STOP
            if (generated == ConstraintToken.Stop).any(dim=1).all():
                break

        return generated


    def training_step(self, batch: ConstraintTokenInput, batch_idx: int):
        loss, preds, target = self._compute_loss(batch)
        metrics = self._compute_metrics(preds, target, dataloader_idx=0)

        self.log('train/loss', loss)
        for metric_key, metric_value in metrics.items():
            self.log(f'train/{metric_key}', metric_value, prog_bar=True)

        return loss

    def validation_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target = self._compute_loss(batch)
        metrics = self._compute_metrics(preds, target, dataloader_idx=dataloader_idx)

        suffix = '_fc' if dataloader_idx == 1 else ''
        self.log(f'validation/loss{suffix}', loss, add_dataloader_idx=False, sync_dist=True)
        for metric_key, metric_value in metrics.items():
            self.log(f'validation/{metric_key}{suffix}', metric_value, add_dataloader_idx=False, sync_dist=True)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Generate samples autoregressively from scratch and compute kappa_gen.

        This is the fair comparison metric with MDD's kappa_gen.
        """
        val_loaders = self.trainer.val_dataloaders
        if val_loaders is None:
            return

        # Handle single dataloader vs list
        if not isinstance(val_loaders, list):
            val_loaders = [val_loaders]

        for loader_idx, loader in enumerate(val_loaders):
            try:
                val_iter = next(iter(loader))
            except StopIteration:
                continue

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in val_iter.items()}

            # Get target and sequence length
            target = batch['c_val']
            _, max_len = target.shape
            generated = self.generate_sample(batch, max_len, dtype=target.dtype)

            # Compute metrics for validation
            metrics = self._compute_metrics(generated, target, generated_from_scratch=True, dataloader_idx=loader_idx)

            suffix = '_fc' if loader_idx == 1 else ''
            for metric_name, metric_value in metrics.items():
                self.log(f'validation_gen/{metric_name}{suffix}', metric_value, add_dataloader_idx=False, sync_dist=True)

            # Print comparison on rank 0
            if self.global_rank == 0:
                print("\n" + "="*80)
                print(f"VALIDATION{suffix.upper()} SAMPLE COMPARISON")
                print("="*80)

                for i in range(min(5, generated.size(0))):
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {self._format_constraint_sequence(target[i].cpu().numpy())}")
                    print(f"  Gen : {self._format_constraint_sequence(generated[i].cpu().numpy())}")

                print("="*80 + "\n")

    def test_step(self, batch: ConstraintTokenInput, batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target = self._compute_loss(batch)
        metrics = self._compute_metrics(preds, target, dataloader_idx=dataloader_idx)

        suffix = '_fc' if dataloader_idx == 1 else ''
        self.log(f'test/loss{suffix}', loss, add_dataloader_idx=False, sync_dist=True)
        for metric_key, metric_value in metrics.items():
            self.log(f'test/{metric_key}{suffix}', metric_value, add_dataloader_idx=False, sync_dist=True)

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
            batch_size, max_len = target.shape
            generated = self.generate_sample(batch, max_len, dtype=target.dtype)

            # Compute metrics
            metrics = self._compute_metrics(generated, target, generated_from_scratch=True, dataloader_idx=loader_idx)

            # Use different names for each loader since on_test_epoch_end is called once
            suffix = '' if loader_idx == 0 else '_fc'
            for metric_name, metric_value in metrics.items():
                self.log(f'test_gen/{metric_name}{suffix}', metric_value, add_dataloader_idx=False, sync_dist=True)

            # Print samples on rank 0
            if self.global_rank == 0:
                print("\n" + "="*80)
                print(f"TEST{suffix.upper()} SAMPLE COMPARISON")
                print("="*80)

                for i in range(min(5, generated.size(0))):
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {self._format_constraint_sequence(target[i].cpu().numpy())}")
                    print(f"  Gen : {self._format_constraint_sequence(generated[i].cpu().numpy())}")

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
                        print(f"Solver Report (test{suffix}): {solver_results}")
