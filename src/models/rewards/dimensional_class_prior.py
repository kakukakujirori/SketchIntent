"""DimensionalClassPriorReward — signed prior over dimensional constraint types.

This is the dimensional counterpart to ``RelationalClassPriorReward``.  It is a
small sequence-level scaffold for class composition, not a geometric response
metric and not a replacement for dimensional-count regularization.  Typical use
is a light CAD prior such as preferring axis-aligned ``DistanceX/Y`` over
unconstrained ``Distance``.
"""
from __future__ import annotations

import torch

from src.metrics.constraint_stats import constraint_head_counts
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.sketch_handlers.constants import (
    ConstraintToken,
    DIMENSIONAL_CONSTRAINT_TOKEN_IDS,
)
from src.sketch_handlers.context import SketchContext


_GATE_MODES = ("always", "converged", "fc")


class DimensionalClassPriorReward(BaseReward):
    """Capped signed per-type prior over dimensional constraint classes."""

    name = "dim_prior"

    def __init__(
        self,
        context: SketchContext,
        coeff_by_type: dict[str, float],
        gate: str = "converged",
        cap_positive_total: float = 0.15,
        cap_negative_total: float = 0.0,
        cap_per_type: int = 2,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """Create a dimensional class-prior reward.

        Args:
            context: Sketch tokenization context; used for constraint block size.
            coeff_by_type: Mapping from dimensional ``ConstraintToken`` names
                such as ``"Distance"``, ``"DistanceX"``, or ``"Radius"`` to
                per-occurrence coefficients.  Positive values add a bonus,
                negative values add a penalty.
            gate: Solver-status gate.  ``converged`` scores FC/Under samples,
                ``fc`` scores only FC, ``always`` scores everything.
            cap_positive_total: Upper bound on the positive per-sequence bonus.
            cap_negative_total: Lower bound on the negative per-sequence
                penalty.  This must be a non-positive value, e.g. ``-0.15``.
            cap_per_type: Each dimensional type contributes at most this many
                occurrences, so spamming one type cannot farm the score.
            weight: Composer-level multiplier from ``BaseReward``.
            enabled: Composer-level on/off switch from ``BaseReward``.
        """
        super().__init__(weight=weight, enabled=enabled)
        if gate not in _GATE_MODES:
            raise ValueError(f"gate must be one of {_GATE_MODES}, got {gate!r}")
        if cap_positive_total < 0.0:
            raise ValueError(
                f"cap_positive_total must be >= 0, got {cap_positive_total}"
            )
        if cap_negative_total > 0.0:
            raise ValueError(
                f"cap_negative_total must be <= 0, got {cap_negative_total}"
            )
        if cap_per_type < 0:
            raise ValueError(f"cap_per_type must be >= 0, got {cap_per_type}")
        if not coeff_by_type:
            raise ValueError("coeff_by_type must be a non-empty mapping")

        self.gate = gate
        self.cap_positive_total = float(cap_positive_total)
        self.cap_negative_total = float(cap_negative_total)
        self.cap_per_type = int(cap_per_type)
        self._block_size = int(context.align_constraint_token_length_to)
        self._prefix_len = 1

        dim_tokens = tuple(ConstraintToken(int(t)) for t in DIMENSIONAL_CONSTRAINT_TOKEN_IDS)
        valid_dim_names = {t.name for t in dim_tokens}
        coeff_map: dict[str, float] = {}
        for name, coeff in dict(coeff_by_type).items():
            if name not in valid_dim_names:
                raise ValueError(
                    f"unknown dimensional constraint type {name!r} in coeff_by_type; "
                    f"valid names are {sorted(valid_dim_names)}"
                )
            coeff_map[name] = float(coeff)
        self._coeff_map = coeff_map
        self._scored_types = tuple(ConstraintToken[name] for name in coeff_map)
        self.register_buffer(
            "_type_token_ids",
            torch.tensor([int(t) for t in self._scored_types], dtype=torch.long),
            persistent=False,
        )
        self._type_names = tuple(t.name for t in self._scored_types)
        self.register_buffer(
            "_coeffs",
            torch.tensor([coeff_map[name] for name in self._type_names], dtype=torch.float32),
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

    def _score_from_counts(
        self, counts: torch.Tensor
    ) -> tuple[float, float, float]:
        capped = counts.clamp(max=self.cap_per_type).to(dtype=torch.float32)
        contribs = capped * self._coeffs.to(capped.device)
        positive = float(contribs.clamp(min=0.0).sum().item())
        negative = float(contribs.clamp(max=0.0).sum().item())
        positive = min(positive, self.cap_positive_total)
        negative = max(negative, self.cap_negative_total)
        return positive + negative, positive, negative

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        cons_tokens = ctx.cons_tokens
        per_sample_solver_status = ctx.per_sample_solver_status
        B, L = cons_tokens.shape

        counts = constraint_head_counts(
            cons_tokens,
            self._block_size,
            self._type_token_ids,
            prefix_len=self._prefix_len,
        )

        eq_stop = (cons_tokens == int(ConstraintToken.Stop)).int()
        any_stop = eq_stop.any(dim=-1)
        first_stop = torch.where(
            any_stop,
            eq_stop.argmax(dim=-1),
            torch.full((B,), L - 1, device=cons_tokens.device, dtype=torch.long),
        )

        scores = torch.zeros((B, L), dtype=torch.float32)
        score_sum = 0.0
        positive_sum = 0.0
        negative_sum = 0.0
        type_count_sum = {name: 0 for name in self._type_names}

        for i in range(B):
            status = (
                per_sample_solver_status[i]
                if per_sample_solver_status is not None
                else None
            )
            if not self._gate_pass(status):
                continue
            score, positive, negative = self._score_from_counts(counts[i])
            scores[i, int(first_stop[i].item())] += score
            score_sum += score
            positive_sum += positive
            negative_sum += negative
            for j, name in enumerate(self._type_names):
                type_count_sum[name] += int(counts[i, j].item())

        denom = max(1, B)
        metrics: dict[str, float] = {
            "dim_prior_score_mean": score_sum / denom,
            "dim_prior_bonus_mean": positive_sum / denom,
            "dim_prior_penalty_mean": negative_sum / denom,
        }
        for name, count in type_count_sum.items():
            metrics[f"dim_prior_subtype_{name}_count_mean"] = count / denom
        return RewardOutput(scores=scores, metrics=metrics)
