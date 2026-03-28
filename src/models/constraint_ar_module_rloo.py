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
            generated = self.generate_sample(expanded, max_len, dtype=batch['c_val'].dtype)
            rewards = self.reward(expanded['p_val'], generated, expanded['sketch']).to(self.device)  # [B*K, L]

        mask = (generated[:, 1:] != ConstraintToken.Pad).float()   # [B*K, L-1]
        n_valid = mask.sum().clamp(min=1)

        # token log probs
        gen_batch = {**expanded, 'c_val': generated}
        output = self.model(gen_batch)                             # [B*K, L, V]
        logits = output[:, :-1, :]                                 # [B*K, L-1, V]
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            dim=-1, index=generated[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                                              # [B*K, L-1]

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
    ) -> torch.Tensor:
        # Start with just START token, rest is zeros (will be filled)
        batch_size = batch['p_val'].shape[0]
        generated = torch.full((batch_size, max_len), ConstraintToken.Pad, device=self.device, dtype=dtype)
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

            # Replace with Pad if sequence has already generated STOP
            already_stopped = (generated[:, :pos+1] == ConstraintToken.Stop).any(dim=1)
            next_token = torch.where(
                already_stopped,
                torch.tensor(ConstraintToken.Pad, device=self.device, dtype=dtype),
                next_token,
            )

            # Fill in position pos+1
            generated[:, pos + 1] = next_token

            # Early stopping if all sequences have generated STOP
            if already_stopped.all():
                break

        return generated

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
        loss, logits, target = self._compute_ce_loss(batch)
        preds = self.logits_to_preds(logits)

        accuracy = self.accuracy_metric(preds, target)

        self.log(f'validation/loss', loss, add_dataloader_idx=True, sync_dist=True)
        self.log(f'validation/accuracy', accuracy, add_dataloader_idx=True, sync_dist=True)

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
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in val_iter.items()}

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
                    print(f"  GT  : {visualize_constraint_sequence(target_sorted[i].cpu().numpy())}")
                    print(f"  Gen : {visualize_constraint_sequence(generated_sorted[i].cpu().numpy())}")

                print("="*80 + "\n")

    def test_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        loss, logits, target = self._compute_ce_loss(batch)
        preds = self.logits_to_preds(logits)

        accuracy = self.accuracy_metric(preds, target)

        self.log(f'test/loss', loss, add_dataloader_idx=True, sync_dist=True)
        self.log(f'test/accuracy', accuracy, add_dataloader_idx=True, sync_dist=True)

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
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in test_iter.items()}

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
                    print(f"  GT  : {visualize_constraint_sequence(target_sorted[i].cpu().numpy())}")
                    print(f"  Gen : {visualize_constraint_sequence(generated_sorted[i].cpu().numpy())}")

                print("="*80 + "\n")

            # Solver-based evaluation (rank 0 only)
            if self.global_rank == 0:
                try:
                    solver_feedback = SolverFeedback(self.hparams.context)
                    solver_results = solver_feedback.evaluate_constraints_batch(
                        batch['p_val'],
                        generated,
                        batch['sketch'],
                        fully_constrained_dof=2,
                    )
                    if solver_results:
                        print(f"Solver Report: {solver_results}")
                except ImportError as e:
                    print(str(e))
