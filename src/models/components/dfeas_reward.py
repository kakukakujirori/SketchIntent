"""DFeasReward — first-order edit-feasibility reward for RLOO.

Implements a reward signal based on ``d_feas`` (see
``notebooks/summary_histcad_response_metric_v3.md``).  Given a predicted
constraint set for a sketch, the source GT sketch acts as the design-intent
reference:

  - Q_G = dimensional constraints of the GT sketch (edit probes)
  - R_G = relational constraints of the GT sketch (preservation set)

For each probe q in Q_G we form the predicted response S_P^(q) on the
Pred sketch and accumulate the two residuals defined in v3:

  T_P^(q) = J_q S_P^(q) - I             gain residual (edit attainability)
  L_P^(q) = J_{R_G} S_P^(q)             relation leak (relational preservation)

The per-probe scalar is sqrt(gain_ms + leak_ms) (row-balanced), averaged over
probes, then capped and negated to become a reward.

This module is intentionally CAD-solver-free; the only heavy work is per-sketch
torch.func.jacfwd plus two SVDs (one on J_x of pred, one per probe of size k_q
× n_p).  Cost is comparable to or cheaper than FreeCAD evaluation.

The reward is placed at the first STOP token of the predicted sequence to match
the existing SolverReward convention.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import sketchgraphs.data as datalib
from src.metrics.edit_response_distance import (
    _is_dimensional,
    _is_relational,
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
)
from src.models.components.solver_feedback import SolverFeedback
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import complete_constraint_values


def _stack_relation_jacobians(rels, state, context):
    rows: list[torch.Tensor] = []
    total = 0
    for rel in rels:
        j = constraint_jacobian(rel, state, context)
        if j is None or j.shape[0] == 0:
            continue
        rows.append(j)
        total += j.shape[0]
    if not rows:
        return None, 0
    return torch.cat(rows, dim=0), total


def _compute_d_feas(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    context: SketchContext,
    eps: float,
    cap: float,
) -> tuple[float, float, float, int, int]:
    """Return (d_feas, gain_mean, leak_mean, n_probes_used, n_relations)."""
    edit_probes = [c for c in gt_sketch.constraints.values() if _is_dimensional(c)]
    rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]
    if not edit_probes:
        return 0.0, 0.0, 0.0, 0, len(rels)

    try:
        pred_native = compute_native_response(pred_sketch, context, eps=eps)
    except Exception:
        return cap, cap, 0.0, 0, len(rels)

    J_R_pred, n_R = _stack_relation_jacobians(rels, pred_native.state, context)

    d_feas_sum = 0.0
    gain_sum = 0.0
    leak_sum = 0.0
    count = 0
    for probe in edit_probes:
        try:
            S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps)
            if S_pr is None:
                continue
            Jq_pr = constraint_jacobian(probe, pred_native.state, context)
            if Jq_pr is None:
                continue
            k_q = max(1, Jq_pr.shape[0])
            eye = torch.eye(
                Jq_pr.shape[0], dtype=Jq_pr.dtype, device=Jq_pr.device
            )
            gain_residual = Jq_pr @ S_pr - eye
            gain_ms = float(torch.linalg.norm(gain_residual) ** 2) / k_q
            leak_ms = 0.0
            if J_R_pred is not None and J_R_pred.shape[0] > 0:
                leak = J_R_pred @ S_pr
                leak_ms = float(torch.linalg.norm(leak) ** 2) / max(1, n_R)
            d_per = math.sqrt(gain_ms + leak_ms)
            if not math.isfinite(d_per):
                d_per = cap
            d_feas_sum += min(d_per, cap)
            gain_sum += math.sqrt(gain_ms)
            leak_sum += math.sqrt(leak_ms)
            count += 1
        except Exception:
            continue

    if count == 0:
        return 0.0, 0.0, 0.0, 0, len(rels)
    d_feas = min(d_feas_sum / count, cap)
    return d_feas, gain_sum / count, leak_sum / count, count, len(rels)


class DFeasReward(torch.nn.Module):
    """Reward module computing -d_feas(pred, gt) per sample.

    The reward is a single scalar per sample placed at the first STOP token of
    the predicted constraint sequence (matching SolverReward's placement of the
    sequence-level reward).

    Args:
        context: tokeniser / sketch handler context.
        eps: SVD rank cutoff used inside d_feas (matches v3 default 1e-9).
        d_feas_cap: clip individual d_feas values at this cap. Heavy-tail
            outliers in v3 reached >1e5; cap=5 keeps reward magnitude
            comparable to the solver reward.
        coeff: external multiplier; the returned reward at STOP is
            ``-coeff * min(d_feas, cap)``.
    """

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        d_feas_cap: float = 5.0,
        coeff: float = 1.0,
    ):
        super().__init__()
        self.context = context
        self.eps = eps
        self.d_feas_cap = d_feas_cap
        self.coeff = coeff
        self.tokenizer = SketchTokenizer(context)

    def _build_pred_sketch(
        self,
        cons_tokens_np: np.ndarray,
        prim_tokens_np: np.ndarray,
        sketch_original: datalib.Sketch,
    ) -> datalib.Sketch:
        # Passing reference_sketch=sketch_original makes the decoded entities
        # adopt gt's entity keys by position, so constraint references in pred
        # and gt resolve to the same physical entity (required for d_feas).
        pred_sketch, _, _ = self.tokenizer.detokenize(
            {"p_val": prim_tokens_np, "c_val": cons_tokens_np},
            allow_invalid_constraints=True,
            reference_sketch=sketch_original,
        )
        if sketch_original is not None:
            pred_sketch = SolverFeedback.overwrite_primitives_with_original(
                pred_sketch, sketch_original
            )
        pred_sketch, _ = complete_constraint_values(
            pred_sketch, skip_invalid_constraints=True
        )
        return pred_sketch

    @torch.no_grad()
    def forward(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch],
        converged_mask: list[bool] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        B, L = cons_tokens.shape
        assert len(sketch_original_list) == B
        scores = torch.zeros((B, L), dtype=torch.float32)

        prim_np = prim_tokens.detach().cpu().numpy()
        cons_np = cons_tokens.detach().cpu().numpy()

        d_feas_sum = 0.0
        gain_sum = 0.0
        leak_sum = 0.0
        valid_n = 0
        skipped_build = 0
        skipped_no_probes = 0
        skipped_not_converged = 0

        for i in range(B):
            if converged_mask is not None and not converged_mask[i]:
                skipped_not_converged += 1
                continue

            cons_i = cons_np[i].copy()
            stop_pos = np.where(cons_i == int(ConstraintToken.Stop))[0]
            first_stop_idx = int(stop_pos[0]) if len(stop_pos) > 0 else (L - 1)
            if 0 <= first_stop_idx < L - 1:
                cons_i[first_stop_idx + 1:] = int(ConstraintToken.Pad)

            gt_sketch = sketch_original_list[i]
            try:
                pred_sketch = self._build_pred_sketch(cons_i, prim_np[i], gt_sketch)
            except Exception:
                skipped_build += 1
                continue

            d_feas, gain, leak, n_probes, _ = _compute_d_feas(
                pred_sketch, gt_sketch, self.context,
                eps=self.eps, cap=self.d_feas_cap,
            )
            if n_probes == 0:
                skipped_no_probes += 1
                continue

            scores[i, first_stop_idx] -= self.coeff * float(d_feas)
            d_feas_sum += d_feas
            gain_sum += gain
            leak_sum += leak
            valid_n += 1

        metrics: dict[str, float | int] = {
            "d_feas_mean": d_feas_sum / max(1, valid_n),
            "d_feas_gain_mean": gain_sum / max(1, valid_n),
            "d_feas_leak_mean": leak_sum / max(1, valid_n),
            "d_feas_valid_n": valid_n,
            "d_feas_skipped_build": skipped_build,
            "d_feas_skipped_no_probes": skipped_no_probes,
            "d_feas_skipped_not_converged": skipped_not_converged,
        }
        return scores, metrics
