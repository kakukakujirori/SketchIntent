import copy
import inspect
from typing import Any

import lightning as L
import torch
import torchmetrics
from tqdm import tqdm

from src.metrics import ConstraintIoU, TokenAccuracy
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.solver_reward import SolverReward
from src.models.components.token_reordering import TokenReordering
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext


class ConstraintARModuleRLOO(L.LightningModule):
    """Lightning module for autoregressive constraint model."""

    def __init__(
        self,
        context: SketchContext,
        model: torch.nn.Module,
        reward: SolverReward,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        ref_update_steps: int = -1,
        group_sample_size: int = 8,
        kl_coeff: float = 0.01,
        sft_coeff: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'reward'], logger=False)

        # models
        self.model = model
        self.reward = reward
        self.ref_model = copy.deepcopy(model)

        self.reward.eval()
        for p in self.reward.parameters():
            p.requires_grad_(False)

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # hyper parameters
        self.ref_update_steps = ref_update_steps
        self.group_sample_size = group_sample_size
        self.kl_coeff = kl_coeff
        self.sft_coeff = sft_coeff

        # losses and metrics
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)
        self.accuracy_metric = TokenAccuracy(ignore_index=ConstraintToken.Pad)
        self.iou_metrics = ConstraintIoU(context, prefix_len=1)  # NOTE: prefix_len is important
        self.token_reordering = TokenReordering(context=context, prefix_len=1)  # NOTE: prefix_len is important
        self.solver_feedback = SolverFeedback(context)

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

    def _compute_ce_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Cross Entropy loss. Logits and target are used for preds sampling."""
        output = self.model(batch)
        B, L, V = output.shape
        logits = output[:, :-1]               # [B, L-1, V]
        target = batch['c_val'][:, 1:]        # [B, L-1]
        loss = self.criterion(logits.reshape(-1, V), target.reshape(-1))
        return loss, logits, target

    def _compute_rloo_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, K = batch['p_val'].shape[0], self.group_sample_size
        _, max_len = batch['c_val'].shape

        expanded = {k: v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor) else [x for x in v for _ in range(K)] for k, v in batch.items()}  # [0, 1, 2, ...] => [0, ..., 0, 1, ..., 1, 2, ..., 2, ...]

        # Sample trajectories
        with torch.no_grad():
            generated, gen_logprobs = self.generate_sample(expanded, max_len, dtype=batch['c_val'].dtype)
            rewards, solver_metrics = self.reward(expanded['p_val'], generated, expanded['sketch'])
            rewards = rewards.to(self.device).float()  # [B*K, L]

            # Process solver metrics from collections.Counter into scalar values for logging
            parsed_solver_metrics = {}
            if "status" in solver_metrics:
                status_counter = solver_metrics["status"]
                total_status = sum(status_counter.values())
                if total_status > 0:
                    for s, c in status_counter.items():
                        # Group all "Error: ..." statuses into a single "Error" key
                        s_name = s.split(':')[0].strip() if s.startswith("Error") else s
                        parsed_solver_metrics[f"solver_{s_name.lower()}_rate"] = c / total_status

            if "dof" in solver_metrics:
                dof_counter = solver_metrics["dof"]
                # Compute mean DoF only for successful constraint sequences (DoF >= 0)
                valid_dof_sum = sum(d * c for d, c in dof_counter.items() if d >= 0)
                valid_dof_count = sum(c for d, c in dof_counter.items() if d >= 0)
                parsed_solver_metrics["solver_valid_dof_mean"] = valid_dof_sum / valid_dof_count if valid_dof_count > 0 else 0.0

        mask = (generated[:, 1:] != ConstraintToken.Pad).float()   # [B*K, L-1]
        n_valid = mask.sum().clamp(min=1)

        # token log probs
        gen_batch = {**expanded, 'c_val': generated}
        output = self.model(gen_batch)                             # [B*K, L, V]
        logits = output[:, :-1, :]                                 # [B*K, L-1, V]

        dist = torch.distributions.Categorical(logits=logits)
        entropy = (dist.entropy() * mask).sum() / n_valid

        token_log_probs = dist.log_prob(generated[:, 1:])          # [B*K, L-1]

        # Ratio
        ratio = torch.exp(token_log_probs - gen_logprobs[:, 1:])
        avg_ratio = (ratio * mask).sum() / n_valid

        # KL penalty
        kl_loss_metric = torch.tensor(0.0, device=self.device)
        adjusted_rewards = rewards.clone()
        if self.kl_coeff > 0.0:
            with torch.no_grad():
                ref_output = self.ref_model({**expanded, 'c_val': generated})
                ref_log_probs = torch.log_softmax(ref_output[:, :-1], dim=-1).gather(
                    dim=-1, index=generated[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                kl_penalty = token_log_probs.detach() - ref_log_probs

            # Subtract KL penalty from rewards
            adjusted_rewards[:, 1:] -= self.kl_coeff * kl_penalty * mask
            kl_loss_metric = (kl_penalty * mask).sum() / n_valid

        # Compute Returns (cumulative sum of future rewards)
        returns = adjusted_rewards.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

        # RLOO advantage
        returns_grouped = returns.view(B, K, max_len)
        baseline   = (returns_grouped.sum(dim=1, keepdim=True) - returns_grouped) / (K - 1)
        advantages = (returns_grouped - baseline).view(B * K, max_len)[:, 1:]  # [B*K, L-1]

        # Policy Gradient loss
        pg_loss = -(advantages.detach() * token_log_probs * mask).sum() / n_valid
        loss = pg_loss

        stop_mask = (generated == ConstraintToken.Stop).float()
        metrics = {
            'pg_loss':        pg_loss,
            'kl_loss':        kl_loss_metric,
            'reward_mean':    rewards.mean(),
            'reward_std':     rewards.std(),
            'reward_at_stop': (rewards * stop_mask).sum() / stop_mask.sum().clamp(min=1),
            'entropy':        entropy,
            'ratio':          avg_ratio,
            **parsed_solver_metrics
        }
        return loss, metrics

    @torch.no_grad()
    def logits_to_preds(self, logits: torch.Tensor) -> torch.Tensor:
        lf = logits.clone()
        lf[:, :, ConstraintToken.Mask] = -torch.inf
        preds = torch.distributions.Categorical(logits=lf).sample()
        return preds

    def generate_sample(
        self,
        batch: dict[str, Any],
        max_len: int,
        dtype: torch.dtype = torch.long,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Start with just START token, rest is zeros (will be filled)
        batch_size = batch['p_val'].shape[0]
        generated = torch.full((batch_size, max_len), ConstraintToken.Pad, device=self.device, dtype=dtype)
        generated[:, 0] = ConstraintToken.Start
        logprobs = torch.zeros((batch_size, max_len), device=self.device, dtype=torch.float32)

        # Autoregressive generation
        for pos in tqdm(range(max_len - 1), desc="Sampling", leave=False, disable=self.training):
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
            dist = torch.distributions.Categorical(logits=next_token_logits)
            next_token = dist.sample()
            next_logprob = dist.log_prob(next_token)

            # Replace with Pad if sequence has already generated STOP
            already_stopped = (generated[:, :pos+1] == ConstraintToken.Stop).any(dim=1)
            next_token = torch.where(
                already_stopped,
                torch.tensor(ConstraintToken.Pad, device=self.device, dtype=dtype),
                next_token,
            )
            next_logprob = torch.where(
                already_stopped,
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
                next_logprob,
            )
            generated[:, pos + 1] = next_token
            logprobs[:, pos + 1] = next_logprob

            # Early stopping if all sequences have generated STOP
            if already_stopped.all():
                break

        return generated, logprobs

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        rl_loss, rl_metrics = self._compute_rloo_loss(batch)

        with torch.set_grad_enabled(self.sft_coeff > 0.0):
            ce_loss, _, _ = self._compute_ce_loss(batch)

        total_loss = rl_loss + ce_loss * self.sft_coeff

        self.log('train/ce_loss', ce_loss)
        self.log('train/rl_loss', rl_loss)
        self.log('train/total_loss', total_loss)
        self.log_dict({f'train/{k}': v for k, v in rl_metrics.items()})

        return total_loss

    def on_train_batch_end(self, outputs, batch: dict[str, Any], batch_idx: int):
        if self.ref_update_steps > 0 and (batch_idx + 1) % self.ref_update_steps == 0:
            # Copy the policy weight to ref
            self.ref_model.load_state_dict(self.model.state_dict())

    def validation_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        loss, logits, ce_target = self._compute_ce_loss(batch)
        preds = self.logits_to_preds(logits)

        accuracy = self.accuracy_metric(preds, ce_target)

        self.log(f'validation/loss', loss, add_dataloader_idx=True, sync_dist=True)
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

        # Evaluate solver constraints and log solver feedback metrics
        try:
            solver_results = self.solver_feedback.evaluate_constraints_batch(
                batch['p_val'],
                generated,
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
        loss, logits, ce_target = self._compute_ce_loss(batch)
        preds = self.logits_to_preds(logits)

        accuracy = self.accuracy_metric(preds, ce_target)

        self.log(f'test/loss', loss, add_dataloader_idx=True, sync_dist=True)
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

        # Evaluate solver constraints and log solver feedback metrics
        try:
            solver_results = self.solver_feedback.evaluate_constraints_batch(
                batch['p_val'],
                generated,
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