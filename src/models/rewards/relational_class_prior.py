"""RelationalClassPriorReward — exploration scaffold for rare relations.

This reward is **not** part of the ResponseMetric family.  Its only job is to
nudge the policy into *sampling* rare relational constraint tokens (``Equal``,
``Tangent``, ``Mirror``, ``Parallel``, ``Perpendicular`` and ``Coincident``) so
that those candidates appear in the rollout distribution at all.  A selection
reward such as ``ResponseIntentReward`` can only prefer candidates the policy
actually emits; if rare relations have ~0 probability, no amount of selection
shaping surfaces them.

Because it is a scaffold, it must be applied identically to the ``S`` and
``S + ResponseIntentReward`` arms of every comparison — never to the intent arm
alone.  See ``notebooks/rloo_response_metric_reward_research_plan.md`` §4.3.

The bonus is a small, capped, per-type token-count bonus placed at the first
STOP of each generated sequence (a sequence-level signal), gated on solver
status so we do not reward relations on already-broken sketches.
"""
from __future__ import annotations

import torch

from src.metrics.constraint_stats import constraint_head_counts
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.sketch_handlers.constants import (
    ConstraintToken,
    KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS,
)
from src.sketch_handlers.context import SketchContext


_GATE_MODES = ("always", "converged", "fc")

# ``coeff_by_type`` is derived from the human-GT relation frequencies by
# ``scripts/derive_relational_class_prior_coeffs.py`` (reproducible); paste its
# output into the configs.


class RelationalClassPriorReward(BaseReward):
    """Capped per-type count bonus that puts rare relations in the candidate set."""

    name = "rel_prior"

    def __init__(
        self,
        context: SketchContext,
        coeff_by_type: dict[str, float],
        gate: str = "converged",
        cap_total: float = 0.5,
        cap_per_type: int = 2,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """Create a relational class-prior reward.

        Args:
            context: Sketch tokenization context (for the constraint block size).
            coeff_by_type: Mapping from ``ConstraintToken`` name (e.g.
                ``"Tangent"``) to the per-occurrence bonus coefficient.  All
                coefficients must be non-negative — this reward only ever adds.
                Required (supplied from config) so the coefficients are recorded
                in the Hydra run config; derive them with
                ``scripts/derive_relational_class_prior_coeffs.py``.
            gate: Solver-status gate.  ``converged`` scores FC/Under samples,
                ``fc`` scores only FC, ``always`` scores everything.
            cap_total: Upper bound on the per-sequence bonus, so the prior can
                never out-scale ``SolverReward``.
            cap_per_type: Each constraint type contributes at most this many
                occurrences, so spamming one relation cannot farm the bonus.
            weight: Composer-level multiplier from ``BaseReward``.
            enabled: Composer-level on/off switch from ``BaseReward``.
        """
        super().__init__(weight=weight, enabled=enabled)
        if gate not in _GATE_MODES:
            raise ValueError(f"gate must be one of {_GATE_MODES}, got {gate!r}")
        if cap_total < 0.0:
            raise ValueError(f"cap_total must be >= 0, got {cap_total}")
        if cap_per_type < 0:
            raise ValueError(f"cap_per_type must be >= 0, got {cap_per_type}")
        if not coeff_by_type:
            raise ValueError("coeff_by_type must be a non-empty mapping")

        self.gate = gate
        self.cap_total = float(cap_total)
        self.cap_per_type = int(cap_per_type)
        self._block_size = int(context.align_constraint_token_length_to)
        self._prefix_len = 1

        type_ids: list[int] = []
        coeffs: list[float] = []
        key_flags: list[float] = []
        key_set = {int(t) for t in KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS}
        # ``coeff_by_type`` may arrive as an OmegaConf DictConfig; ``.items()``
        # still yields plain (str, float) pairs.
        for name, coeff in dict(coeff_by_type).items():
            try:
                token = ConstraintToken[name]
            except KeyError as exc:
                raise ValueError(
                    f"unknown constraint type {name!r} in coeff_by_type; "
                    f"valid names are {[t.name for t in ConstraintToken]}"
                ) from exc
            if float(coeff) < 0.0:
                raise ValueError(
                    f"coeff_by_type[{name!r}] must be >= 0, got {coeff}"
                )
            type_ids.append(int(token))
            coeffs.append(float(coeff))
            key_flags.append(1.0 if int(token) in key_set else 0.0)

        # Buffers keep the resolved type ids / coeffs alongside the module so a
        # config diff (which types, what weight) is fully reproducible.
        self.register_buffer(
            "_type_ids", torch.tensor(type_ids, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_coeffs", torch.tensor(coeffs, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "_key_mask",
            torch.tensor(key_flags, dtype=torch.float32),
            persistent=False,
        )

    def _gate_pass(self, status: str | None) -> bool:
        if self.gate == "always" or status is None:
            return True
        if self.gate == "converged":
            return status in ("FC", "Under")
        if self.gate == "fc":
            return status == "FC"
        return False

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        cons_tokens = ctx.cons_tokens
        per_sample_solver_status = ctx.per_sample_solver_status
        B, L = cons_tokens.shape

        # Per-type head counts via the shared head-of-block counter (same logic
        # the constraint-mix logging uses).  Move to CPU for the small bonus
        # math so the returned scores match the CPU tensors every other reward
        # produces (the composer sums on CPU before moving to device).
        counts = constraint_head_counts(
            cons_tokens, self._block_size, self._type_ids, prefix_len=self._prefix_len
        ).to("cpu")                                                       # (B, n)
        coeffs = self._coeffs.to("cpu")                                   # (n,)
        key_mask = self._key_mask.to("cpu")                              # (n,)

        capped = counts.clamp(max=self.cap_per_type).to(torch.float32)    # (B, n)
        per_type_bonus = capped * coeffs.view(1, -1)                      # (B, n)
        bonus = per_type_bonus.sum(dim=1).clamp(max=self.cap_total)       # (B,)
        # Diagnostic: how much of the (pre-total-cap) bonus comes from the key
        # relations we most care about surfacing.
        key_rel_bonus = (per_type_bonus * key_mask.view(1, -1)).sum(dim=1)  # (B,)

        # STOP positions = placement target for the sequence-level bonus.
        eq_stop = (cons_tokens == int(ConstraintToken.Stop)).int()
        any_stop = eq_stop.any(dim=-1)
        first_stop = torch.where(
            any_stop,
            eq_stop.argmax(dim=-1),
            torch.full((B,), L - 1, device=cons_tokens.device, dtype=torch.long),
        ).to("cpu")

        scores = torch.zeros((B, L), dtype=torch.float32)
        for i in range(B):
            status = (
                per_sample_solver_status[i]
                if per_sample_solver_status is not None
                else None
            )
            if not self._gate_pass(status):
                continue
            scores[i, int(first_stop[i].item())] += float(bonus[i].item())

        metrics: dict[str, float] = {
            "rel_prior_bonus_mean": float(bonus.mean().item()),
            "rel_prior_key_rel_bonus_mean": float(key_rel_bonus.mean().item()),
        }
        return RewardOutput(scores=scores, metrics=metrics)
