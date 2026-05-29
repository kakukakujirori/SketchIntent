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
from src.models.rewards.composer import (
    CONSTRAINT_MIX_KEY,
    RewardComposer,
    RewardValComposer,
)
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
        reward_val_composer: RewardValComposer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        ref_update_steps: int = -1,
        group_sample_size: int = 8,
        entropy_bonus: float = 0.01,
        kl_coeff: float = 0.01,
        sft_coeff: float = 0.0,
        relational_only: bool = False,
        train_sample_temperature: float = 1.0,
        val_sample_temperature: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['model', 'reward_composer', 'reward_val_composer'],
            logger=False,
        )

        # models
        self.model = model
        self.reward_composer = reward_composer
        self.reward_val_composer = reward_val_composer
        self.ref_model = copy.deepcopy(model)

        self.relational_only = relational_only
        if relational_only:
            self.register_buffer(
                "_dim_token_ids",
                torch.tensor(DIMENSIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
                persistent=False,
            )

        # All reward / eval modules are non-trainable.
        for composer in (self.reward_composer, self.reward_val_composer):
            composer.eval()
            for p in composer.parameters():
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
        # Sampling temperature for RLOO rollouts (train) and validation
        # generation.  Raised above 1.0 to restore candidate diversity after a
        # rel-only warm start collapses the dimensional-token logits.
        self.train_sample_temperature = float(train_sample_temperature)
        self.val_sample_temperature = float(val_sample_temperature)

        # losses and metrics
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)
        self.tokenizer = SketchTokenizer(context)
        self.token_reordering = TokenReordering(context=context, prefix_len=1)  # NOTE: prefix_len is important
        self.solver_feedback = SolverFeedback(context)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # Compile the full-sequence forward() for CE loss / ref-model passes.
            self.model = torch.compile(self.model)
            self.ref_model = torch.compile(self.ref_model)
            # Compile generate_one_step separately for the AR sampling loop.
            # mode="default" avoids CUDAGraphs (which conflict with KV-cache
            # in-place writes); Inductor still fuses kernels and optimises.
            self._compiled_step = torch.compile(
                self.model.generate_one_step,
                mode="default",
                fullgraph=False,
            )

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

    def _compute_group_memory(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        expanded: dict[str, Any],
        group_size: int,
    ) -> torch.Tensor:
        """Compute primitive memory once per prompt when it is semantically safe.

        With active dropout, repeating a B-sized memory tensor would share one
        dropout mask across the K rollouts.  In that case we keep the old
        expanded computation so each rollout keeps independent stochasticity.
        """
        active_dropout = model.training and any(
            isinstance(m, torch.nn.Dropout) and m.p > 0.0
            for m in model.modules()
        )
        if active_dropout:
            return model.compute_memory(expanded)
        return model.compute_memory(batch).repeat_interleave(group_size, dim=0)

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
            sample_memory = self._compute_group_memory(
                self.model, batch, expanded, K
            )
            generated, gen_logprobs = self.generate_sample(
                expanded,
                max_len,
                dtype=batch['c_val'].dtype,
                temperature=self.train_sample_temperature,
                memory=sample_memory,
            )
            del sample_memory

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
        policy_memory = self._compute_group_memory(self.model, batch, expanded, K)
        output = self.model(gen_batch, memory=policy_memory)        # [B*K, L, V]
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
                ref_memory = self._compute_group_memory(
                    self.ref_model, batch, expanded, K
                )
                ref_output = self.ref_model(
                    {**expanded, 'c_val': generated},
                    memory=ref_memory,
                )
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
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive token generation using KV cache.

        Complexity: O(L²) total instead of O(L³) (avoids re-running the full
        sequence through the decoder at every step).

        Cross-attention KV vectors are precomputed once from the fixed primitive
        memory.  Self-attention KV vectors are accumulated in pre-allocated
        buffers that are written in-place (no torch.cat allocations).

        All tensor shapes inside the loop are **static** (full-length KV buffers
        + causal attention masks), enabling torch.compile / Inductor to capture
        a single graph that is replayed at every step.
        """
        if memory is None:
            batch_size = batch['p_val'].shape[0]
            # ── Step A: compute primitive memory (done ONCE) ──────────────── #
            memory = self.model.compute_memory(batch)   # [B, M, E]
        else:
            batch_size = memory.shape[0]

        generated = torch.full((batch_size, max_len), ConstraintToken.Pad, device=self.device, dtype=dtype)
        generated[:, 0] = ConstraintToken.Start
        logprobs = torch.zeros((batch_size, max_len), device=self.device, dtype=torch.float32)

        # ── Step B: pre-compute cross-attention KV from memory (ONCE) ─── #
        cross_kv = self.model.compute_cross_kv_cache(memory)

        # ── Step C: pre-allocate self-attention KV buffers ────────────── #
        self_kv = self.model.init_kv_cache(
            batch_size, max_len, device=self.device, dtype=memory.dtype
        )

        # ── Step D: pre-build causal masks (fixed shapes) ────────────── #
        causal_masks = ConstraintARModel.build_causal_masks(
            max_len, device=self.device, dtype=memory.dtype,
        )  # [max_len, 1, 1, max_len]

        stopped = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Choose step function: compiled (training) or eager (val/test)
        step_fn = getattr(self, '_compiled_step', self.model.generate_one_step)

        # Pre-allocate position tensors to avoid CPU-GPU transfer overhead per step
        pos_tensors = torch.arange(max_len, device=self.device, dtype=torch.long)

        # ── AR loop ───────────────────────────────────────────────────── #
        for pos in tqdm(range(max_len - 1), desc="Sampling", leave=False, disable=self.training):
            causal_mask = causal_masks[pos]  # [1, 1, max_len] — fixed shape
            pos_tensor = pos_tensors[pos]

            next_token_logits = step_fn(
                memory, generated[:, pos], pos_tensor, self_kv, cross_kv, causal_mask,
            )  # [B, M]

            next_token_logits[:, ConstraintToken.Mask] = -torch.inf
            if self.relational_only:
                next_token_logits[:, self._dim_token_ids] = -torch.inf
            next_token_logits = next_token_logits / temperature

            dist = torch.distributions.Categorical(logits=next_token_logits)
            next_token  = dist.sample()
            next_logprob = dist.log_prob(next_token)

            # Override with Pad for sequences that already stopped
            next_token = next_token.masked_fill(stopped, ConstraintToken.Pad)
            next_logprob = next_logprob.masked_fill(stopped, 0.0)

            generated[:, pos + 1] = next_token
            logprobs[:, pos + 1]  = next_logprob

            stopped = stopped | (next_token == ConstraintToken.Stop)
            # Check stopped.all() infrequently to avoid severe CPU-GPU sync bottleneck
            if pos % 16 == 0 and stopped.all().item():
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
        generated, _ = self.generate_sample(
            batch,
            max_len,
            dtype=full_target.dtype,
            temperature=self.val_sample_temperature,
        )

        # Compute constraint-level metrics
        generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
        target_sorted, _ = self.token_reordering(full_target, None, None, "canonical_with_tail_strip")

        try:
            eval_metrics = self.reward_val_composer.compute(
                batch['p_val'], generated, batch['sketch'], batch['c_val']
            )
            for metric_name, metric_value in eval_metrics.items():
                # Keys starting with "_" carry non-scalar payloads (e.g. the
                # per-type constraint-mix for the bar chart); accumulate those
                # separately instead of logging them as scalars.
                if metric_name.startswith("_"):
                    if metric_name == CONSTRAINT_MIX_KEY:
                        self._accumulate_constraint_mix(metric_value)
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

    def _accumulate_constraint_mix(self, payload: dict[str, Any]) -> None:
        """Sum the per-type constraint-mix payload across validation batches."""
        acc = getattr(self, "_val_constraint_mix", None)
        if acc is None:
            acc = {
                "names": list(payload["names"]),
                "families": list(payload["families"]),
                "counts": [float(c) for c in payload["counts"]],
                "n_total": 0,
                "n_sketches": 0,
            }
            self._val_constraint_mix = acc
        for j, c in enumerate(payload["counts"]):
            acc["counts"][j] += float(c)
        acc["n_total"] += int(payload["n_total"])
        acc["n_sketches"] += int(payload["n_sketches"])

    def _log_constraint_mix_chart(self) -> None:
        """Log the per-type constraint-mix Table + bar chart to W&B (rank 0).

        Scalars (``eval_*_ratio_mean``) already cover the aggregate families; this
        chart shows the full per-type distribution without flooding the dashboard
        with one scalar per constraint type.  Non-W&B loggers (or a charting
        failure) are skipped so validation never breaks on visualization.
        """
        acc = getattr(self, "_val_constraint_mix", None)
        self._val_constraint_mix = None
        if acc is None or self.global_rank != 0:
            return
        logger = self.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        try:
            import wandb

            n_sketches = max(1, int(acc["n_sketches"]))
            n_total = max(1, int(acc["n_total"]))
            step = int(self.global_step)
            rows = []
            bar_rows = []
            for name, family, count in zip(
                acc["names"], acc["families"], acc["counts"]
            ):
                mean_count = float(count) / n_sketches
                fraction = float(count) / n_total
                rows.append([step, name, family, mean_count, fraction])
                bar_rows.append([name, fraction])
            table = wandb.Table(
                columns=["step", "constraint_type", "family", "mean_count", "fraction"],
                data=rows,
            )
            bar = wandb.plot.bar(
                wandb.Table(columns=["constraint_type", "fraction"], data=bar_rows),
                "constraint_type",
                "fraction",
                title="Generated constraint type mix (fraction)",
            )
            logger.experiment.log(  # type: ignore[attr-defined]
                {
                    "validation/constraint_type_mix_table": table,
                    "validation/constraint_type_mix_bar": bar,
                }
            )
        except Exception as e:  # pragma: no cover - visualization must never break val
            print(f"[on_validation_epoch_end] constraint-mix chart logging failed: {e}")

    @torch.no_grad()
    def on_validation_epoch_end(self):

        self._log_constraint_mix_chart()

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
        generated, _ = self.generate_sample(
            batch,
            max_len,
            dtype=full_target.dtype,
            temperature=self.val_sample_temperature,
        )

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
