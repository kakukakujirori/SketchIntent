import lightning as L
import torch
import torchmetrics
from tqdm import tqdm

from src.metrics import ConstraintIoU, TokenAccuracy
from src.models.components.token_reordering import TokenReordering
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext


class ConstraintARModule(L.LightningModule):
    """Lightning module for autoregressive constraint model."""

    def __init__(
        self,
        context: SketchContext,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)

        self.accuracy_metric = TokenAccuracy(ignore_index=ConstraintToken.Pad)
        self.iou_metrics = ConstraintIoU(context)
        self.token_reordering = TokenReordering(context=context)

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

    def _reset_metrics(self):
        """Reset all metrics to prevent accumulation across epochs."""
        self.accuracy_metric.reset()
        self.iou_metrics.reset()

    def on_train_epoch_start(self):
        self._reset_metrics()

    def on_validation_epoch_start(self):
        self._reset_metrics()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            # total_steps (e.g. for OneCycleLR) is a runtime Trainer property and
            # cannot be resolved statically in Hydra configs, so we inject it here.
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def generate_sample(
        self,
        batch: dict[str, torch.Tensor],
        max_len: int,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        # Start with just START token, rest is zeros (will be filled)
        batch_size = batch['p_val'].shape[0]
        generated = torch.full((batch_size, max_len), ConstraintToken.Pad,
                            device=self.device, dtype=dtype)
        generated[:, 0] = ConstraintToken.Start

        # Autoregressive generation
        for pos in tqdm(range(max_len - 1), desc="Sampling"):
            # Create batch with current generated tokens
            gen_batch = {
                'p_val': batch['p_val'],
                'p_attr': batch['p_attr'],
                'p_pos': batch['p_pos'],
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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        loss, preds, target = self._compute_loss(batch)
        accuracy = self.accuracy_metric(preds, target)

        self.log('train/loss', loss)
        self.log(f'train/accuracy', accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target = self._compute_loss(batch)
        accuracy = self.accuracy_metric(preds, target)

        self.log(f'validation/loss', loss, add_dataloader_idx=False, sync_dist=True)
        self.log(f'validation/accuracy', accuracy, add_dataloader_idx=False, sync_dist=True)

    @torch.no_grad()
    def on_validation_epoch_end(self):
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

            # Get target and sequence length
            target = batch['c_val']
            _, max_len = target.shape
            generated = self.generate_sample(batch, max_len, dtype=target.dtype)

            # Compute constraint-level metrics
            generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
            target_sorted, _ = self.token_reordering(target, None, None, "canonical_with_tail_strip")
            iou_metrics = self.iou_metrics(generated_sorted, target_sorted)
            for metric_name, metric_value in iou_metrics.items():
                self.log(f'validation_gen/{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True)

            # Print comparison on rank 0
            if self.global_rank == 0:
                print("\n" + "="*80)
                print(f"VALIDATION SAMPLE COMPARISON")
                print("="*80)

                for i in range(min(5, generated.size(0))):
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {self._format_constraint_sequence(target_sorted[i].cpu().numpy())}")
                    print(f"  Gen : {self._format_constraint_sequence(generated_sorted[i].cpu().numpy())}")

                print("="*80 + "\n")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        loss, preds, target = self._compute_loss(batch)
        accuracy = self.accuracy_metric(preds, target)

        self.log(f'test/loss', loss, add_dataloader_idx=False, sync_dist=True)
        self.log(f'test/accuracy', accuracy, add_dataloader_idx=False, sync_dist=True)

    @torch.no_grad()
    def on_test_epoch_end(self):
        """Generate samples autoregressively and compute metrics for test dataloaders."""
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
                test_iter = next(iter(loader))
            except StopIteration:
                continue

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in test_iter.items()}

            # Get target and sequence length
            target = batch['c_val']
            batch_size, max_len = target.shape
            generated = self.generate_sample(batch, max_len, dtype=target.dtype)

            # Compute constraint-level metrics
            generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
            target_sorted, _ = self.token_reordering(target, None, None, "canonical_with_tail_strip")
            iou_metrics = self.iou_metrics(generated_sorted, target_sorted)
            for metric_name, metric_value in iou_metrics.items():
                self.log(f'test_gen/{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True)

            # Print samples on rank 0
            if self.global_rank == 0:
                print("\n" + "="*80)
                print(f"TEST SAMPLE COMPARISON")
                print("="*80)

                for i in range(min(5, generated.size(0))):
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {self._format_constraint_sequence(target_sorted[i].cpu().numpy())}")
                    print(f"  Gen : {self._format_constraint_sequence(generated_sorted[i].cpu().numpy())}")

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
