import copy
import inspect
from typing import Any, cast

import numpy as np
import lightning as L
import torch
from tqdm import tqdm

from src.models.components.constraint_ar_model import ConstraintARModel
from src.models.components.dfeas_reward import DFeasReward
from src.models.components.relstruct_reward import RelStructReward
from src.models.components.response_metric_reward import ResponseMetricReward
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.solver_reward import SolverReward
from src.models.components.token_reordering import TokenReordering
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.sketch_handlers.constants import ConstraintToken, DIMENSIONAL_CONSTRAINT_TOKEN_IDS
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


class ConstraintARModuleRLOO(L.LightningModule):
    """Lightning module for autoregressive constraint model."""

    def __init__(
        self,
        context: SketchContext,
        model: ConstraintARModel,
        reward: SolverReward,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        ref_update_steps: int = -1,
        group_sample_size: int = 8,
        entropy_bonus: float = 0.01,
        kl_coeff: float = 0.01,
        sft_coeff: float = 0.0,
        gt_reward_coeff: float = 0.0,
        penalty_duplicate: float = -1.0,
        dfeas_reward: DFeasReward | None = None,
        dfeas_apply_to: str = "converged",
        relstruct_reward: RelStructReward | None = None,
        response_metric_reward: ResponseMetricReward | None = None,
        relational_only: bool = False,
    ):
        super().__init__()
        if dfeas_apply_to not in ("converged", "fc"):
            raise ValueError(f"unknown dfeas_apply_to: {dfeas_apply_to}")
        self.save_hyperparameters(
            ignore=[
                'model',
                'reward',
                'dfeas_reward',
                'relstruct_reward',
                'response_metric_reward',
            ],
            logger=False,
        )

        # models
        self.model = model
        self.reward = reward
        self.dfeas_reward = dfeas_reward
        self.relstruct_reward = relstruct_reward
        self.response_metric_reward = response_metric_reward
        self.ref_model = copy.deepcopy(model)

        self.relational_only = relational_only
        if relational_only:
            self.register_buffer(
                "_dim_token_ids",
                torch.tensor(DIMENSIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
                persistent=False,
            )

        self.reward.eval()
        for p in self.reward.parameters():
            p.requires_grad_(False)

        if self.dfeas_reward is not None:
            self.dfeas_reward.eval()
            for p in self.dfeas_reward.parameters():
                p.requires_grad_(False)

        if self.relstruct_reward is not None:
            self.relstruct_reward.eval()
            for p in self.relstruct_reward.parameters():
                p.requires_grad_(False)

            # If RelStructReward owns the dim-count penalty, zero out
            # SolverReward's penalty_num_dimensional_constraints to avoid
            # double-counting (plan §6.1 / config comment "set
            # penalty_num_dimensional_constraints=0 when use_dim_penalty=true").
            if getattr(self.relstruct_reward, "use_dim_penalty", False):
                prev = getattr(self.reward, "penalty_num_dimensional_constraints", 0.0)
                if prev != 0.0:
                    print(
                        f"[ConstraintARModuleRLOO] RelStructReward.use_dim_penalty=True "
                        f"-> overriding SolverReward.penalty_num_dimensional_constraints "
                        f"{prev} -> 0.0 to avoid double penalty."
                    )
                    self.reward.penalty_num_dimensional_constraints = 0.0

        if self.response_metric_reward is not None:
            self.response_metric_reward.eval()
            for p in self.response_metric_reward.parameters():
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
        self.gt_reward_coeff = gt_reward_coeff
        self.penalty_duplicate = penalty_duplicate
        self.dfeas_apply_to = dfeas_apply_to
        self._block_size: int = context.align_constraint_token_length_to  # tokens per constraint
        self._prefix_len: int = 1  # START token

        # losses and metrics
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ConstraintToken.Pad)
        self.tokenizer = SketchTokenizer(context)
        self.token_reordering = TokenReordering(context=context, prefix_len=1)  # NOTE: prefix_len is important
        self.solver_feedback = SolverFeedback(context)

    def _reset_metrics(self):
        """Reset all metrics to prevent accumulation across epochs."""
        pass

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

    @torch.no_grad()
    def _compute_gt_match_reward(
        self,
        generated: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token GT-match reward for each generated sequence.

        For every constraint block in *generated*, checks whether an identical
        block exists in *gt* (after intra-ref canonical sorting on both sides).
        Matching is **one-to-one**: each GT block can be consumed by at most one
        generated block (greedy first-match in scan order).  This prevents
        duplicate-spam from farming the same GT block N times.

        Args:
            generated: [BK, L] generated token sequences (includes START token at pos 0).
            gt: [BK, L] ground-truth token sequences (same shape, already expanded K times).

        Returns:
            reward: [BK, L] float32 tensor (0 everywhere except matched block tails).
        """
        BK, L = generated.shape
        bs = self._block_size
        prefix = self._prefix_len

        # ── extract body blocks (skip prefix START token) ────────────────────
        # body region: [prefix, prefix + N*bs)  where N = (L - prefix) // bs
        N = (L - prefix) // bs
        body_end = prefix + N * bs

        # gen_blocks: [BK, N, bs]
        gen_blocks = generated[:, prefix:body_end].reshape(BK, N, bs)
        # gt_blocks:  [BK, M, bs]  (M may differ from N in theory, but same L → M == N)
        gt_blocks = gt[:, prefix:body_end].reshape(BK, N, bs)

        # ── intra-ref sort (canonical ordering within each block) ─────────────
        # We leverage reward.sort_intra_refs on numpy; it's called once per sequence
        # so the overhead is acceptable.
        gen_np = generated.cpu().numpy()
        gt_np  = gt.cpu().numpy()

        gen_sorted_np = np.stack(
            [self.reward.sort_intra_refs(gen_np[k]) for k in range(BK)], axis=0
        )  # [BK, L]
        gt_sorted_np = np.stack(
            [self.reward.sort_intra_refs(gt_np[k]) for k in range(BK)], axis=0
        )  # [BK, L]

        gen_sorted = torch.from_numpy(gen_sorted_np).to(generated.device)
        gt_sorted  = torch.from_numpy(gt_sorted_np).to(gt.device)

        gen_body = gen_sorted[:, prefix:body_end].reshape(BK, N, bs)  # [BK, N, bs]
        gt_body  = gt_sorted[:, prefix:body_end].reshape(BK, N, bs)   # [BK, M, bs]
        M = gt_body.shape[1]  # equals N

        # ── valid-block masks ─────────────────────────────────────────────────
        # A block is invalid (STOP / PAD) if ANY token == Stop or Pad
        pad_id  = int(ConstraintToken.Pad)
        stop_id = int(ConstraintToken.Stop)

        # gen_invalid: [BK, N] — True if the generated block is Stop/Pad
        gen_invalid = (
            (gen_body == pad_id) | (gen_body == stop_id)
        ).any(dim=-1)

        # gt_invalid: [BK, M] — True if the GT block is Stop/Pad
        gt_invalid = (
            (gt_body == pad_id) | (gt_body == stop_id)
        ).any(dim=-1)

        # ── pairwise block match (vectorised) ─────────────────────────────────
        # Compare every gen block against every gt block for each sample.
        # gen_exp: [BK, N, 1, bs]  vs  gt_exp: [BK, 1, M, bs]
        gen_exp = gen_body.unsqueeze(2)  # [BK, N, 1, bs]
        gt_exp  = gt_body.unsqueeze(1)   # [BK, 1, M, bs]

        # block_match: [BK, N, M]  — True if whole block matches
        block_match = (gen_exp == gt_exp).all(dim=-1)

        # Disallow matches with invalid blocks on either side
        block_match = block_match & (~gt_invalid.unsqueeze(1))
        block_match = block_match & (~gen_invalid.unsqueeze(2))

        # ── greedy one-to-one assignment ──────────────────────────────────────
        # Each GT block can be matched at most once. Iterate gen blocks in scan
        # order and consume the first available GT match. N is small (~32), so
        # the Python loop over N is cheap.
        used = torch.zeros(BK, M, dtype=torch.bool, device=generated.device)
        matched = torch.zeros(BK, N, dtype=torch.bool, device=generated.device)
        row_idx = torch.arange(BK, device=generated.device)
        for n in range(N):
            cand = block_match[:, n, :] & ~used          # [BK, M]
            has_match = cand.any(dim=-1)                  # [BK]
            first_idx = cand.int().argmax(dim=-1)         # [BK]; 0 if no match
            matched[:, n] = has_match
            # Mark consumed GT block (only where a match actually exists)
            used[row_idx[has_match], first_idx[has_match]] = True

        # ── assign reward to the LAST token of each matched block ─────────────
        reward = torch.zeros(BK, L, dtype=torch.float32, device=generated.device)
        if matched.any():
            # last position of block n in the full sequence:  prefix + n*bs + (bs-1)
            block_tail_positions = torch.arange(N, device=generated.device) * bs + (bs - 1) + prefix
            # [BK, N] positions → scatter into reward
            # For matched blocks, add gt_reward_coeff at the tail position
            # We scatter a filled coefficient tensor masked by `matched`
            reward_to_add = matched.float() * self.gt_reward_coeff  # [BK, N]
            reward.scatter_add_(
                dim=1,
                index=block_tail_positions.unsqueeze(0).expand(BK, -1),
                src=reward_to_add,
            )

        return reward  # [BK, L]

    @torch.no_grad()
    def _compute_duplicate_penalty(self, generated: torch.Tensor) -> torch.Tensor:
        """Penalise duplicate constraint blocks within a sequence.

        FreeCAD's solver returns only a single id in ``RedundantConstraints``
        even when N identical constraints are added, so SolverReward's
        ``penalty_per_problematic_constraint`` only fires once regardless of
        N.  This method closes that loophole by checking the canonical-sorted
        generated blocks ourselves and applying ``penalty_duplicate`` to every
        non-first occurrence at the block's tail token.

        Args:
            generated: [BK, L] generated token sequences.

        Returns:
            penalty: [BK, L] float32 tensor (<= 0 everywhere).
        """
        BK, L = generated.shape
        bs = self._block_size
        prefix = self._prefix_len
        N = (L - prefix) // bs
        body_end = prefix + N * bs

        pad_id  = int(ConstraintToken.Pad)
        stop_id = int(ConstraintToken.Stop)

        gen_np = generated.cpu().numpy()
        gen_sorted_np = np.stack(
            [self.reward.sort_intra_refs(gen_np[k]) for k in range(BK)], axis=0
        )
        gen_sorted = torch.from_numpy(gen_sorted_np).to(generated.device)
        gen_body = gen_sorted[:, prefix:body_end].reshape(BK, N, bs)

        gen_invalid = (
            (gen_body == pad_id) | (gen_body == stop_id)
        ).any(dim=-1)  # [BK, N]

        penalty = torch.zeros(BK, L, dtype=torch.float32, device=generated.device)
        block_tail_positions = (
            torch.arange(N, device=generated.device) * bs + (bs - 1) + prefix
        )

        # Per-sample loop — BK*N ≈ 256*32 = 8k iterations, negligible.
        gen_body_cpu = gen_body.cpu().numpy()
        gen_invalid_cpu = gen_invalid.cpu().numpy()
        tail_cpu = block_tail_positions.cpu().numpy()
        for k in range(BK):
            seen: set[tuple[int, ...]] = set()
            for n in range(N):
                if gen_invalid_cpu[k, n]:
                    continue
                key = tuple(gen_body_cpu[k, n].tolist())
                if key in seen:
                    penalty[k, int(tail_cpu[n])] += self.penalty_duplicate
                else:
                    seen.add(key)
        return penalty

    def _compute_rloo_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, K = batch['p_val'].shape[0], self.group_sample_size
        _, max_len = batch['c_val'].shape

        expanded = {k: v.repeat_interleave(K, dim=0) if isinstance(v, torch.Tensor) else [x for x in v for _ in range(K)] for k, v in batch.items()}

        # Step 1: Sample trajectories (GPU)
        with torch.no_grad():
            generated, gen_logprobs = self.generate_sample(expanded, max_len, dtype=batch['c_val'].dtype)

        # Step 2: Kick off solver asynchronously (CPU workers start now,
        # running concurrently with the GPU forward passes below)
        p_val_expanded: torch.Tensor = cast(torch.Tensor, expanded['p_val'])
        solver_handle = self.reward.submit_async(
            p_val_expanded, generated, expanded['sketch']
        )

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

        # KL penalty (computed before collect_results — no dependency on rewards)
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

        # Step 4: Collect solver results (blocks only for remaining solver time)
        rewards_cpu, solver_metrics, per_sample_status = self.reward.collect_results(solver_handle)
        rewards = rewards_cpu.to(self.device).float()  # [B*K, L]

        # Step 4b: GT-match reward (added on top of solver reward)
        if self.gt_reward_coeff > 0.0:
            gt_expanded = cast(torch.Tensor, expanded['c_val'])  # [B*K, L]
            gt_match_reward = self._compute_gt_match_reward(generated, gt_expanded)
            rewards = rewards + gt_match_reward

        # Step 4b': duplicate-block penalty.  FreeCAD reports only a single
        # RedundantConstraints id for N identical constraints, so SolverReward's
        # per-constraint penalty does not scale with N.  This closes the
        # spam-the-same-constraint loophole regardless of the reward mix.
        duplicate_penalty: torch.Tensor | None = None
        if self.penalty_duplicate < 0.0:
            duplicate_penalty = self._compute_duplicate_penalty(generated)
            rewards = rewards + duplicate_penalty

        # Step 4c: d_feas reward (first-order edit feasibility / intent preservation)
        # Conflicting / redundant sketches have ill-defined first-order
        # responses, so d_feas is only meaningful on solver-converged samples.
        # For intent shaping, it can also be restricted to FC samples so the
        # reward does not compete with early DoF closure.
        dfeas_metrics: dict[str, float | int] = {}
        if self.dfeas_reward is not None:
            if self.dfeas_apply_to == "fc":
                converged_mask = [s == "FC" for s in per_sample_status]
            else:
                converged_mask = [s in ("FC", "Under") for s in per_sample_status]
            dfeas_scores_cpu, dfeas_metrics = self.dfeas_reward(
                p_val_expanded, generated, expanded['sketch'],
                converged_mask=converged_mask,
            )
            dfeas_scores = dfeas_scores_cpu.to(self.device).float()
            rewards = rewards + dfeas_scores

        # Step 4d: RelStruct reward (plan A″ — leak_prime + d_relstruct + gain_intrinsic)
        relstruct_metrics: dict[str, float | int] = {}
        if self.relstruct_reward is not None:
            relstruct_scores_cpu, relstruct_metrics = self.relstruct_reward(
                p_val_expanded, generated, expanded['sketch'],
                per_sample_status=per_sample_status,
            )
            rewards = rewards + relstruct_scores_cpu.to(self.device).float()

        # Step 4e: direct response-metric reward (Hθ / d_SVD).
        response_metric_metrics: dict[str, float | int] = {}
        if self.response_metric_reward is not None:
            response_metric_scores_cpu, response_metric_metrics = self.response_metric_reward(
                p_val_expanded, generated, expanded['sketch'],
                per_sample_status=per_sample_status,
            )
            rewards = rewards + response_metric_scores_cpu.to(self.device).float()

        # Parse solver metrics (Counter → scalar)
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

        # Step 5: Compute returns / advantages / loss
        adjusted_rewards = rewards.clone()
        if self.kl_coeff > 0.0 and kl_penalty is not None:
            # Subtract KL penalty from rewards
            adjusted_rewards[:, 1:] -= self.kl_coeff * kl_penalty * mask

        # Compute Returns (cumulative sum of future rewards)
        returns = adjusted_rewards.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

        # RLOO advantage
        returns_grouped = returns.view(B, K, max_len)
        baseline   = (returns_grouped.sum(dim=1, keepdim=True) - returns_grouped) / (K - 1)
        advantages = (returns_grouped - baseline).view(B * K, max_len)[:, 1:]  # [B*K, L-1]

        # Policy Gradient loss
        pg_loss = -(advantages.detach() * token_log_probs * mask).sum() / n_valid

        # Entropy bonus
        loss = pg_loss - self.entropy_bonus * entropy

        stop_mask = (generated == ConstraintToken.Stop).float()

        gt_match_reward_metric = torch.tensor(0.0, device=self.device)
        if self.gt_reward_coeff > 0.0:
            # gt_match_reward was added into rewards already; recover it for logging
            gt_match_reward_metric = gt_match_reward.sum() / n_valid  # type: ignore[possibly-undefined]

        duplicate_penalty_metric = torch.tensor(0.0, device=self.device)
        if duplicate_penalty is not None:
            duplicate_penalty_metric = duplicate_penalty.sum() / n_valid

        metrics = {
            'pg_loss':            pg_loss,
            'kl_loss':            kl_loss_metric,
            'reward_mean':        rewards.mean(),
            'reward_std':         rewards.std(),
            'reward_at_stop':     (rewards * stop_mask).sum() / stop_mask.sum().clamp(min=1),
            'entropy':            entropy,
            'ratio':              avg_ratio,
            'gt_match_reward':    gt_match_reward_metric,
            'duplicate_penalty':  duplicate_penalty_metric,
            **parsed_solver_metrics,
            **dfeas_metrics,
            **relstruct_metrics,
            **response_metric_metrics,
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

        self.log(f'validation/loss', loss, add_dataloader_idx=True, sync_dist=True)

        # Generate sample
        full_target = batch['c_val']
        max_len = full_target.shape[1]
        generated, _ = self.generate_sample(batch, max_len, dtype=full_target.dtype)

        # Compute constraint-level metrics
        generated_sorted, _ = self.token_reordering(generated, None, None, "canonical_with_tail_strip")
        target_sorted, _ = self.token_reordering(full_target, None, None, "canonical_with_tail_strip")

        # Evaluate solver constraints and log solver feedback metrics.
        # per_sample_status is reused for gating RelStructReward bonuses, so
        # both signals come from a single solver evaluation pass.
        per_sample_status: list[str] | None = None
        try:
            solver_results, per_sample_status = self.solver_feedback.evaluate_constraints_batch(
                batch['p_val'].cpu().numpy(),
                generated.cpu().numpy(),
                batch['sketch'],
                fully_constrained_dof=2,
                return_per_sample_status=True,
            )
            if solver_results:
                for metric_name, metric_value in solver_results.items():
                    self.log(f'validation_gen/solver_{metric_name}', metric_value, add_dataloader_idx=False, sync_dist=True, on_epoch=True)
        except Exception as e:
            if self.global_rank == 0:
                print(f"[validation_step] Solver evaluation failed: {e}")

        # Evaluate d_feas and log
        if self.dfeas_reward is not None:
            try:
                _, dfeas_val_results = self.dfeas_reward(
                    batch['p_val'],
                    generated,
                    batch['sketch'],
                )
                for metric_name, metric_value in dfeas_val_results.items():
                    self.log(f'validation_gen/{metric_name}', float(metric_value), add_dataloader_idx=False, sync_dist=True, on_epoch=True)
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[validation_step] DFeas evaluation failed: {e}")

        # Evaluate RelStruct reward components and log
        if self.relstruct_reward is not None:
            try:
                _, relstruct_val_results = self.relstruct_reward(
                    batch['p_val'],
                    generated,
                    batch['sketch'],
                    per_sample_status=per_sample_status,
                )
                for metric_name, metric_value in relstruct_val_results.items():
                    self.log(f'validation_gen/{metric_name}', float(metric_value), add_dataloader_idx=False, sync_dist=True, on_epoch=True)
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[validation_step] RelStruct evaluation failed: {e}")

        # Evaluate direct response-metric reward components and log
        if self.response_metric_reward is not None:
            try:
                _, response_metric_val_results = self.response_metric_reward(
                    batch['p_val'],
                    generated,
                    batch['sketch'],
                    per_sample_status=per_sample_status,
                )
                for metric_name, metric_value in response_metric_val_results.items():
                    self.log(f'validation_gen/{metric_name}', float(metric_value), add_dataloader_idx=False, sync_dist=True, on_epoch=True)
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[validation_step] ResponseMetric evaluation failed: {e}")

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
