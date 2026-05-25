import inspect
from tqdm import tqdm
from typing import Any

import lightning as L
import torch

from src.metrics import ConstraintBLEU, ConstraintIoU, TokenAccuracy
from src.models.components.constraint_ar_model import ConstraintARModel
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.token_reordering import TokenReordering
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


class ConstraintARModule(L.LightningModule):
    """Lightning module for autoregressive constraint model."""

    def __init__(
        self,
        context: SketchContext,
        model: ConstraintARModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'], logger=False)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)

        self.accuracy_metric = TokenAccuracy(ignore_index=ConstraintToken.Pad)
        self.iou_metrics = ConstraintIoU(context, prefix_len=1)
        self.tokenizer = SketchTokenizer(context)
        self.bleu_metrics = ConstraintBLEU(tokenizer=self.tokenizer)
        self.token_reordering = TokenReordering(context=context, prefix_len=1)  # NOTE: prefix_len is important
        self.solver_feedback = SolverFeedback(context)

    def _reset_metrics(self):
        """Reset all metrics to prevent accumulation across epochs."""
        self.accuracy_metric.reset()
        self.iou_metrics.reset()
        self.bleu_metrics.reset()

    def on_train_epoch_start(self):
        self._reset_metrics()

    def on_validation_epoch_start(self):
        self._reset_metrics()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            # total_steps (e.g. for OneCycleLR) is a runtime Trainer property and
            # cannot be resolved statically in Hydra configs, so we inject it here.
            if 'total_steps' in inspect.signature(self.hparams.scheduler).parameters:
                scheduler = self.hparams.scheduler(
                    optimizer=optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
                )
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
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

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch)

    def _compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @torch.no_grad()
    def generate_sample(
        self,
        batch: dict[str, Any],
        max_len: int,
        dtype: torch.dtype = torch.long,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive token generation using KV cache.

        Complexity: O(L²) total instead of O(L³) (avoids re-running the full
        sequence through the decoder at every step).

        Cross-attention KV vectors are precomputed once from the fixed primitive
        memory.  Self-attention KV vectors are accumulated in pre-allocated
        buffers that are written in-place (no torch.cat allocations).
        """
        batch_size = batch['p_val'].shape[0]
        generated = torch.full((batch_size, max_len), ConstraintToken.Pad, device=self.device, dtype=dtype)
        generated[:, 0] = ConstraintToken.Start
        logprobs = torch.zeros((batch_size, max_len), device=self.device, dtype=torch.float32)

        # ── Step A: compute primitive memory (done ONCE) ──────────────── #
        memory = self.model.compute_memory(batch)   # [B, M, E]

        # ── Step B: pre-compute cross-attention KV from memory (ONCE) ─── #
        # Memory never changes across AR steps → K, V can be cached permanently.
        cross_kv = self.model.compute_cross_kv_cache(memory)  # list[(k,v)] per layer

        # ── Step C: pre-allocate self-attention KV buffers ────────────── #
        # Buffers are written in-place at each step; no repeated cat().
        self_kv = self.model.init_kv_cache(
            batch_size, max_len, device=self.device, dtype=memory.dtype
        )

        stopped = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # ── AR loop ───────────────────────────────────────────────────── #
        for pos in tqdm(range(max_len - 1), desc="Sampling", leave=False, disable=self.training):
            # Embed current token at position pos: [B, 1, E]
            new_emb = self.model.embed_token_at_pos(memory, generated[:, pos], pos)

            # One decoder step — writes to self_kv in-place, O(pos) per step
            dec_out = self.model.decoder_step_cached(new_emb, pos, self_kv, cross_kv)  # [B, 1, E]

            # Pointer logits: [B, 1, E] @ [B, E, M] → [B, M]
            next_token_logits = torch.matmul(dec_out, memory.transpose(1, 2)).squeeze(1)
            next_token_logits[:, ConstraintToken.Mask] = -torch.inf

            dist = torch.distributions.Categorical(logits=next_token_logits)
            next_token  = dist.sample()
            next_logprob = dist.log_prob(next_token)

            # Override with Pad for sequences that already stopped
            next_token = torch.where(
                stopped,
                torch.tensor(ConstraintToken.Pad, device=self.device, dtype=dtype),
                next_token,
            )
            next_logprob = torch.where(
                stopped,
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
                next_logprob,
            )
            generated[:, pos + 1] = next_token
            logprobs[:, pos + 1]  = next_logprob

            stopped = stopped | (next_token == ConstraintToken.Stop)
            if stopped.all():
                break

        return generated, logprobs



    def training_step(self, batch: dict[str, Any], batch_idx: int):
        loss, preds, target = self._compute_loss(batch)
        accuracy = self.accuracy_metric(preds, target)

        self.log('train/ce_loss', loss)
        self.log(f'train/accuracy', accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        loss, preds, ce_target = self._compute_loss(batch)
        accuracy = self.accuracy_metric(preds, ce_target)

        self.log(f'validation/ce_loss', loss, add_dataloader_idx=True, sync_dist=True)
        self.log(f'validation/accuracy', accuracy, add_dataloader_idx=True, sync_dist=True)

        # Generate sample
        full_target = batch['c_val']
        max_len = full_target.shape[1]
        generated, _ = self.generate_sample(batch, max_len, dtype=full_target.dtype)

        # Compute constraint-level metrics
        generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
        target_sorted, _ = self.token_reordering(full_target, None, None, "canonical_with_tail_strip")
        iou_metrics = self.iou_metrics(generated_sorted, target_sorted)
        for metric_name, metric_value in iou_metrics.items():
            self.log(f'validation_gen/{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True, on_epoch=True)

        # Log BLEU (Batch level returned by __call__, aggregated by Lightning if on_epoch=True)
        bleu_results = self.bleu_metrics.evaluate_batch(batch, generated)
        for k, v in bleu_results.items():
            if k == "categories" and isinstance(v, dict):
                for cat_name, cat_scores in v.items():
                    for s_name, s_val in cat_scores.items():
                        self.log(f'validation_gen/bleu_{cat_name}_{s_name}', s_val, add_dataloader_idx=False, sync_dist=True, on_epoch=True)
            elif isinstance(v, torch.Tensor):
                self.log(f'validation_gen/bleu_{k}', v, add_dataloader_idx=False, sync_dist=True, on_epoch=True)

        # Evaluate solver constraints and log solver feedback metrics
        try:
            solver_results = self.solver_feedback.evaluate_constraints_batch(
                batch['p_val'].cpu().numpy(),
                generated.cpu().numpy(),
                batch['sketch'],
                fully_constrained_dof=2,
            )
            if solver_results:
                for metric_name, metric_value in solver_results.items():
                    self.log(f'validation_gen/solver_{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True, on_epoch=True)
        except Exception as e:
            if self.global_rank == 0:
                print(f"[validation_step] Solver evaluation failed: {e}")

        # Save first batch for wandb visualization
        if batch_idx == 0:
            if not hasattr(self, "_val_samples"):
                self._val_samples = []
            self._val_samples.append((target_sorted, generated_sorted, dataloader_idx))

    @torch.no_grad()
    def on_validation_epoch_end(self):
        # Reset metrics
        self._reset_metrics()

        if not hasattr(self, "_val_samples") or not self._val_samples:
            return

        for target_sorted, generated_sorted, loader_idx in self._val_samples:
            if self.global_rank == 0:
                print("\n" + "="*80)
                print(f"VALIDATION SAMPLE COMPARISON")
                print("="*80)

                columns = ["Ground Truth", "Generated"]
                table_data = []

                for i in range(min(5, generated_sorted.size(0))):
                    gt_str = ", ".join(visualize_constraint_sequence(target_sorted[i].cpu().numpy()))
                    gen_str = ", ".join(visualize_constraint_sequence(generated_sorted[i].cpu().numpy()))
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {gt_str}")
                    print(f"  Gen : {gen_str}")
                    table_data.append([gt_str, gen_str])

                print("="*80 + "\n")

                if self.logger is not None and hasattr(self.logger, "log_text"):
                    self.logger.log_text(  # type: ignore
                        key=f"validation/samples_{loader_idx}",
                        columns=columns,
                        data=table_data,
                    )

        self._val_samples.clear()

    def test_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        loss, preds, ce_target = self._compute_loss(batch)
        accuracy = self.accuracy_metric(preds, ce_target)

        self.log(f'test/ce_loss', loss, add_dataloader_idx=True, sync_dist=True)
        self.log(f'test/accuracy', accuracy, add_dataloader_idx=True, sync_dist=True)

        # Generate sample
        full_target = batch['c_val']
        max_len = full_target.shape[1]
        generated, _ = self.generate_sample(batch, max_len, dtype=full_target.dtype)

        # Compute constraint-level metrics
        generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
        target_sorted, _ = self.token_reordering(full_target, None, None, "canonical_with_tail_strip")
        iou_metrics = self.iou_metrics(generated_sorted, target_sorted)
        for metric_name, metric_value in iou_metrics.items():
            self.log(f'test_gen/{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True, on_epoch=True)

        # Log BLEU (Batch level returned by __call__, aggregated by Lightning if on_epoch=True)
        bleu_results = self.bleu_metrics.evaluate_batch(batch, generated)
        for k, v in bleu_results.items():
            if k == "categories" and isinstance(v, dict):
                for cat_name, cat_scores in v.items():
                    for s_name, s_val in cat_scores.items():
                        self.log(f'test_gen/bleu_{cat_name}_{s_name}', s_val, add_dataloader_idx=False, sync_dist=True, on_epoch=True)
            elif isinstance(v, torch.Tensor):
                self.log(f'test_gen/bleu_{k}', v, add_dataloader_idx=False, sync_dist=True, on_epoch=True)

        # Evaluate solver constraints and log solver feedback metrics
        try:
            solver_results = self.solver_feedback.evaluate_constraints_batch(
                batch['p_val'].cpu().numpy(),
                generated.cpu().numpy(),
                batch['sketch'],
                fully_constrained_dof=2,
            )
            if solver_results:
                for metric_name, metric_value in solver_results.items():
                    self.log(f'test_gen/solver_{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True, on_epoch=True)
                if self.global_rank == 0:
                    print(f"Solver Report: {solver_results}")
        except Exception as e:
            if self.global_rank == 0:
                print(f"[test_step] Solver evaluation failed: {e}")

        # Save first batch for wandb visualization
        if batch_idx == 0:
            if not hasattr(self, "_test_samples"):
                self._test_samples = []
            self._test_samples.append((target_sorted, generated_sorted, dataloader_idx))

    @torch.no_grad()
    def on_test_epoch_end(self):
        # Reset metrics
        self._reset_metrics()

        if not hasattr(self, "_test_samples") or not self._test_samples:
            return

        for target_sorted, generated_sorted, loader_idx in self._test_samples:
            if self.global_rank == 0:
                print("\n" + "="*80)
                print(f"TEST SAMPLE COMPARISON")
                print("="*80)

                columns = ["Ground Truth", "Generated"]
                table_data = []

                for i in range(min(5, generated_sorted.size(0))):
                    gt_str = ", ".join(visualize_constraint_sequence(target_sorted[i].cpu().numpy()))
                    gen_str = ", ".join(visualize_constraint_sequence(generated_sorted[i].cpu().numpy()))
                    print(f"\n[Sample {i}]:")
                    print(f"  GT  : {gt_str}")
                    print(f"  Gen : {gen_str}")
                    table_data.append([gt_str, gen_str])

                print("="*80 + "\n")

                if self.logger is not None and hasattr(self.logger, "log_text"):
                    self.logger.log_text(  # type: ignore
                        key=f"test/samples_{loader_idx}",
                        columns=columns,
                        data=table_data,
                    )

        self._test_samples.clear()