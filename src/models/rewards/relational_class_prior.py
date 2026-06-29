"""RelationalClassPriorReward — exploration scaffold for rare relations.

This reward is **not** part of the ResponseMetric family.  Its only job is to
nudge the policy into *sampling* rare relational constraint tokens (``Equal``,
``Tangent``, ``Mirror``, ``Parallel``, ``Perpendicular``, ``Coincident`` …) so
that those candidates appear in the rollout distribution at all.  A selection
reward such as ``ResponseIntentReward`` can only prefer candidates the policy
actually emits; if rare relations have ~0 probability, no amount of selection
shaping surfaces them.

Because it is a scaffold, it must be applied identically to the ``S`` and
``S + ResponseIntentReward`` arms of every comparison — never to the intent arm
alone.  See ``notebooks/rloo_response_metric_reward_research_plan.md`` §4.3.

The bonus is a small, capped, per-(sub)type count bonus placed at the first STOP
of each generated sequence (a sequence-level signal), gated on solver status so
we do not reward relations on already-broken sketches.

Sub-type awareness
------------------
Several SketchGraphs tokens are polymorphic: ``Coincident`` is point-point
(polyline stitching, low intent) / point-on-object / collinear, ``Equal`` is
length / radius, ``Tangent`` is point / edge.  Their design intent differs by
sub-case, so ``coeff_by_type`` accepts **either** a plain head-token name
(``"Coincident"``) **or** a sub-type label (``"Coincident.point_on_object"``,
see :data:`src.metrics.constraint_subtypes.CONSTRAINT_SUBTYPE_LABELS`).  For a
given constraint the most specific matching key wins: its sub-type label if that
label is in ``coeff_by_type``, otherwise its plain type name.  This lets a config
boost just the high-intent sub-case while keeping a small catch-all on the base
type, e.g.::

    coeff_by_type:
      Coincident: 0.012                # fallback (point_point / collinear)
      Coincident.point_on_object: 0.10 # boost the high-intent sub-case

Resolving sub-types requires the reconstructed sketch (reference → entity-type),
so this reward decodes each generated sequence into a ``datalib.Sketch``.  That
reconstruction is shared with ``ResponseIntentReward`` via ``ctx.pred_sketches``
and is strictly cheaper than the ``SolverReward`` solve already run on every
sample.  Per-sub-type counts are logged regardless of which keys are weighted,
so the sub-type distribution is observable before any sub-type coefficient is set.
"""
from __future__ import annotations

import logging

import torch

from src.metrics.constraint_subtypes import (
    CONSTRAINT_SUBTYPE_LABELS,
    classify_constraint_subtype,
    subtype_base,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.models.rewards.utils.sketch_builder import (
    first_stop_and_pad,
    get_or_build_pred_sketch,
)
from src.sketch_handlers.constants import (
    ConstraintToken,
    KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


logger = logging.getLogger(__name__)

_GATE_MODES = ("always", "converged", "fc")


class RelationalClassPriorReward(BaseReward):
    """Capped per-(sub)type count bonus that puts rare relations in the candidate set."""

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
            context: Sketch tokenization context (used to decode generated
                sequences into sketches for sub-type resolution).
            coeff_by_type: Mapping from a ``ConstraintToken`` name (e.g.
                ``"Tangent"``) **or** a sub-type label (e.g.
                ``"Coincident.point_on_object"``, see
                ``CONSTRAINT_SUBTYPE_LABELS``) to the per-occurrence bonus
                coefficient.  All coefficients must be non-negative — this reward
                only ever adds.  Required (supplied from config) so the
                coefficients are recorded in the Hydra run config.
            gate: Solver-status gate.  ``converged`` scores FC/Under samples,
                ``fc`` scores only FC, ``always`` scores everything.
            cap_total: Upper bound on the per-sequence bonus, so the prior can
                never out-scale ``SolverReward``.
            cap_per_type: Each effective (sub)type key contributes at most this
                many occurrences, so spamming one relation cannot farm the bonus.
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

        self.context = context
        self.gate = gate
        self.cap_total = float(cap_total)
        self.cap_per_type = int(cap_per_type)
        self.tokenizer = SketchTokenizer(context)

        valid_subtype_labels = set(CONSTRAINT_SUBTYPE_LABELS)
        valid_type_names = {t.name for t in ConstraintToken}
        # ``coeff_by_type`` may arrive as an OmegaConf DictConfig; ``.items()``
        # still yields plain (str, float) pairs.
        coeff_map: dict[str, float] = {}
        for name, coeff in dict(coeff_by_type).items():
            if "." in name:
                if name not in valid_subtype_labels:
                    raise ValueError(
                        f"unknown constraint sub-type {name!r} in coeff_by_type; "
                        f"valid sub-types are {sorted(valid_subtype_labels)}"
                    )
            elif name not in valid_type_names:
                raise ValueError(
                    f"unknown constraint type {name!r} in coeff_by_type; "
                    f"valid names are {sorted(valid_type_names)}"
                )
            if float(coeff) < 0.0:
                raise ValueError(
                    f"coeff_by_type[{name!r}] must be >= 0, got {coeff}"
                )
            coeff_map[name] = float(coeff)

        self._coeff_map = coeff_map
        # Base type names whose design intent we most want to surface (used only
        # for the ``key_rel_bonus`` diagnostic); a sub-type key counts as "key"
        # when its base type is a key relation.
        self._key_bases = {
            ConstraintToken(int(t)).name for t in KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS
        }

    def _gate_pass(self, status: str | None) -> bool:
        if self.gate == "always" or status is None:
            return True
        if self.gate == "converged":
            return status in ("FC", "Under")
        if self.gate == "fc":
            return status == "FC"
        return False

    def _bonus_for_sketch(self, pred_sketch) -> tuple[float, float, dict[str, int]]:
        """Return ``(bonus, key_rel_bonus, subtype_counts)`` for one sketch.

        ``bonus`` is the total-capped per-sequence bonus; ``key_rel_bonus`` is
        the pre-total-cap contribution from key relations (diagnostic);
        ``subtype_counts`` tallies the polymorphic sub-type labels for logging.
        """
        # Effective-key counts for the weighted bonus, plus sub-type counts for
        # the (coeff-independent) diagnostic — both from one classify per
        # constraint.
        key_counts: dict[str, int] = {}
        subtype_counts: dict[str, int] = {}
        for constr in pred_sketch.constraints.values():
            sub = classify_constraint_subtype(constr, pred_sketch)
            if sub is not None:
                subtype_counts[sub] = subtype_counts.get(sub, 0) + 1
            base = getattr(constr.type, "name", "Unknown")
            # Most specific matching key wins; plain base is the fallback.
            if sub is not None and sub in self._coeff_map:
                key = sub
            elif base in self._coeff_map:
                key = base
            else:
                continue
            key_counts[key] = key_counts.get(key, 0) + 1

        bonus = 0.0
        key_rel_bonus = 0.0
        for key, count in key_counts.items():
            capped = min(count, self.cap_per_type)
            contrib = capped * self._coeff_map[key]
            bonus += contrib
            if subtype_base(key) in self._key_bases:
                key_rel_bonus += contrib
        bonus = min(bonus, self.cap_total)
        return bonus, key_rel_bonus, subtype_counts

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        cons_tokens = ctx.cons_tokens
        B, L = cons_tokens.shape
        per_sample_solver_status = ctx.per_sample_solver_status

        prim_np = ctx.prim_tokens.detach().cpu().numpy()
        cons_np = cons_tokens.detach().cpu().numpy()

        scores = torch.zeros((B, L), dtype=torch.float32)
        bonus_sum = 0.0
        key_rel_sum = 0.0
        subtype_total: dict[str, int] = {}

        for i in range(B):
            status = (
                per_sample_solver_status[i]
                if per_sample_solver_status is not None
                else None
            )
            if not self._gate_pass(status):
                continue

            cons_i, first_stop_idx = first_stop_and_pad(cons_np[i], L)
            try:
                pred_sketch = get_or_build_pred_sketch(
                    ctx,
                    i,
                    self.tokenizer,
                    cons_i,
                    prim_np[i],
                    ctx.sketch_original_list[i],
                )
            except Exception as exc:
                logger.debug(
                    "rel_prior: skipped sample %d (build failure: %s)", i, exc
                )
                continue

            bonus, key_rel_bonus, subtype_counts = self._bonus_for_sketch(
                pred_sketch
            )
            scores[i, first_stop_idx] += bonus
            bonus_sum += bonus
            key_rel_sum += key_rel_bonus
            for label, c in subtype_counts.items():
                subtype_total[label] = subtype_total.get(label, 0) + c

        # Means are taken over the whole batch (gated-out / failed samples count
        # as 0) so the values are comparable across configs and match the
        # per-sequence ``reward_contrib`` convention.
        denom = max(1, B)
        metrics: dict[str, float] = {
            "rel_prior_bonus_mean": bonus_sum / denom,
            "rel_prior_key_rel_bonus_mean": key_rel_sum / denom,
        }
        # Per-sub-type mean count (per sample), so the polymorphic split is
        # observable in train and val even when no sub-type coeff is set.  Keys
        # are sanitized ("." -> "_") to avoid W&B treating them as sections.
        for label in CONSTRAINT_SUBTYPE_LABELS:
            safe = label.replace(".", "_")
            metrics[f"rel_prior_subtype_{safe}_count_mean"] = (
                subtype_total.get(label, 0) / denom
            )
        return RewardOutput(scores=scores, metrics=metrics)
