import copy
import inspect
from tqdm import tqdm
from typing import Any, cast

import lightning as L
import torch

from src.models.components.constraint_ar_model import ConstraintARModel
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.token_reordering import TokenReordering
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.models.rewards.composer import RewardComposer
from src.sketch_handlers.constants import ConstraintToken, DIMENSIONAL_CONSTRAINT_TOKEN_IDS
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


class ConstraintARModuleRLOO(L.LightningModule):
    """Lightning module for autoregressive constraint model."""

    def __init__(
        self,
        context: SketchContext,
        model: ConstraintARModel,
        reward_composer: RewardComposer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        ref_update_steps: int = -1,
        group_sample_size: int = 8,
        entropy_bonus: float = 0.01,
        kl_coeff: float = 0.01,
        sft_coeff: float = 0.0,
        relational_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['model', 'reward_composer'],
            logger=False,
        )

        # models
        self.model = model
        self.reward_composer = reward_composer
        self.ref_model = copy.deepcopy(model)

        self.relational_only = relational_only
        if relational_only:
            self.register_buffer(
                "_dim_token_ids",
                torch.tensor(DIMENSIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
                persistent=False,
            )

        # All reward modules are non-trainable.
        self.reward_composer.eval()
        for p in self.reward_composer.parameters():
            p.requires_grad_(False)

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # hyper parameters
        self.ref_update_steps = ref_update_steps
        self.group_sample_size = group_sample_size
        self.entropy_bonus = entropy_bonus
        self.kl_coeff = kl_coeff
        self.sft_coeff = sft_coeff

        # losses and metrics
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)
        self.tokenizer = SketchTokenizer(context)
        self.token_reordering = TokenReordering(context=context, prefix_len=1)  # NOTE: prefix_len is important
        self.solver_feedback = SolverFeedback(context)

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

        expanded = {k: v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor) else [x for x in v for _ in range(K)] for k, v in batch.items()}

        # Step 1: Sample trajectories (GPU)
        with torch.no_grad():
            generated, gen_logprobs = self.generate_sample(expanded, max_len, dtype=batch['c_val'].dtype)

        # Step 2: Build reward context and kick off async rewards (SolverReward)
        # so the CPU solver runs concurrently with the GPU forward pass below.
        p_val_expanded: torch.Tensor = cast(torch.Tensor, expanded['p_val'])
        gt_expanded: torch.Tensor = cast(torch.Tensor, expanded['c_val'])
        ctx = self.reward_composer.build_context(
            prim_tokens=p_val_expanded,
            cons_tokens=generated,
            sketch_original_list=expanded['sketch'],
            gt_tokens=gt_expanded,
        )
        reward_handles = self.reward_composer.submit_async(ctx)

        # Step 3: GPU compute — runs while CPU solver is in flight
        mask = (generated[:, 1:] != ConstraintToken.Pad).float()   # [B*K, L-1]
        n_valid = mask.sum().clamp(min=1)

        # Policy model forward pass (needs gradient for PG loss)
        gen_batch = {**expanded, 'c_val': generated}
        output = self.model(gen_batch)                             # [B*K, L, V]
        logits = output[:, :-1, :]                                 # [B*K, L-1, V]

        dist = torch.distributions.Categorical(logits=logits)
        entropy = (dist.entropy() * mask).sum() / n_valid
        token_log_probs = dist.log_prob(generated[:, 1:])          # [B*K, L-1]

        # Ratio (for monitoring)
        ratio = torch.exp(token_log_probs - gen_logprobs[:, 1:])
        avg_ratio = (ratio * mask).sum() / n_valid

        # KL penalty (computed before reward collection — no dependency on rewards)
        kl_loss_metric = torch.tensor(0.0, device=self.device)
        kl_penalty: torch.Tensor | None = None
        if self.kl_coeff > 0.0:
            with torch.no_grad():
                ref_output = self.ref_model({**expanded, 'c_val': generated})
                ref_log_probs = torch.log_softmax(ref_output[:, :-1], dim=-1).gather(
                    dim=-1, index=generated[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                kl_penalty = token_log_probs.detach() - ref_log_probs
            kl_loss_metric = (kl_penalty * mask).sum() / n_valid

        # Step 4: Collect all reward signals (blocks only on remaining solver time).
        rewards_cpu, reward_outputs = self.reward_composer.collect(reward_handles, ctx)
        rewards = rewards_cpu.to(self.device).float()  # [B*K, L]

        # Step 5: Compute returns / advantages / loss
        adjusted_rewards = rewards.clone()
        if self.kl_coeff > 0.0 and kl_penalty is not None:
            adjusted_rewards[:, 1:] -= self.kl_coeff * kl_penalty * mask

        returns = adjusted_rewards.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        returns_grouped = returns.view(B, K, max_len)
        baseline = (returns_grouped.sum(dim=1, keepdim=True) - returns_grouped) / (K - 1)
        advantages = (returns_grouped - baseline).view(B * K, max_len)[:, 1:]  # [B*K, L-1]

        pg_loss = -(advantages.detach() * token_log_probs * mask).sum() / n_valid
        loss = pg_loss - self.entropy_bonus * entropy

        stop_mask = (generated == ConstraintToken.Stop).float()
        metrics: dict[str, Any] = {
            'pg_loss':        pg_loss,
            'kl_loss':        kl_loss_metric,
            'reward_mean':    rewards.mean(),
            'reward_std':     rewards.std(),
            'reward_at_stop': (rewards * stop_mask).sum() / stop_mask.sum().clamp(min=1),
            'entropy':        entropy,
            'ratio':          avg_ratio,
            **self.reward_composer.flatten_metrics(reward_outputs),
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
        temperature: float = 1.0,
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
            if self.relational_only:
                next_token_logits[:, self._dim_token_ids] = -torch.inf
            next_token_logits = next_token_logits / temperature

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

        self.log(f'validation/ce_loss', loss, add_dataloader_idx=True, sync_dist=True)

        # Generate sample
        full_target = batch['c_val']
        max_len = full_target.shape[1]
        generated, _ = self.generate_sample(batch, max_len, dtype=full_target.dtype)

        # Compute constraint-level metrics
        generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
        target_sorted, _ = self.token_reordering(full_target, None, None, "canonical_with_tail_strip")

        # Evaluate rewards through the composer so SolverReward remains the
        # single source of solver-derived status for downstream gating.
        ctx = self.reward_composer.build_context(
            prim_tokens=batch['p_val'],
            cons_tokens=generated,
            sketch_original_list=batch['sketch'],
            gt_tokens=batch['c_val'],
        )
        try:
            reward_handles = self.reward_composer.submit_async(ctx)
            _, reward_outputs = self.reward_composer.collect(reward_handles, ctx)
            reward_metrics = self.reward_composer.flatten_metrics(reward_outputs)
            for metric_name, metric_value in reward_metrics.items():
                if isinstance(metric_value, torch.Tensor):
                    if metric_value.ndim != 0:
                        continue
                    metric_value = metric_value.item()
                if not isinstance(metric_value, (int, float)):
                    continue
                self.log(
                    f'validation_gen/{metric_name}',
                    float(metric_value),
                    add_dataloader_idx=False,
                    sync_dist=True,
                    on_epoch=True,
                )
        except Exception as e:
            if self.global_rank == 0:
                print(f"[validation_step] reward evaluation failed: {e}")

        # Save first batch for wandb visualization
        if batch_idx == 0:
            if not hasattr(self, "_val_samples"):
                self._val_samples = []
            self._val_samples.append((target_sorted, generated_sorted, dataloader_idx))

    @torch.no_grad()
    def on_validation_epoch_end(self):

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

        self.log(f'test/loss', loss, add_dataloader_idx=True, sync_dist=True)

        # Generate sample
        full_target = batch['c_val']
        max_len = full_target.shape[1]
        generated, _ = self.generate_sample(batch, max_len, dtype=full_target.dtype)

        # Compute constraint-level metrics
        generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
        target_sorted, _ = self.token_reordering(full_target, None, None, "canonical_with_tail_strip")

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
