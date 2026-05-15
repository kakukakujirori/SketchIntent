"""Finite optimized feasibility diagnostic for 2D sketch constraints.

This is a finite-edit counterpart to ``d_feas`` where the predicted sketch's
existing dimensional values are treated as freely optimizable parameters.  For
each source GT dimensional probe, we set a finite target value, then solve the
predicted sketch with:

  - Pred's relational constraints kept fixed, and
  - the edited GT probe inserted as the target constraint.

Pred's own dimensional constraints are intentionally omitted from the solve:
their values are assumed to be re-optimized after the edit.  The final score
then measures target miss plus finite change in source-GT relation residuals.
This is therefore a reachability diagnostic, not a replacement for local
``d_feas`` edit-knob semantics.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import sketchgraphs.data as datalib

from src.metrics.edit_feasibility_finite import (
    _constraint_signature,
    _relation_leak_per_delta,
)
from src.metrics.edit_response_distance import _is_dimensional, _is_relational
from src.metrics.histcad_2d import (
    HistCADConfig,
    _bbox_diag,
    _evaluate_observable,
    _newton_solve,
    _set_quantity,
)
from src.sketch_handlers.sketch_solver import SketchState


@dataclass
class FiniteOptimizedFeasibilityProbeResult:
    probe_index: int
    probe_signature: tuple
    target_factor: float
    q0_gt: float
    q0_pred: float
    target_value: float
    delta_value: float
    solved: bool
    solve_status: str
    solve_residual: float
    achieved_value: float | None
    gain: float
    leak: float
    distance: float
    n_relation_rows: int
    n_pred_relations: int
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class FiniteOptimizedFeasibilityResult:
    distance: float
    gain: float
    leak: float
    per_probe: list[FiniteOptimizedFeasibilityProbeResult]
    n_pred_dimensions: int
    n_pred_relations: int
    n_source_relations: int
    n_probe_targets: int
    skipped_probe_targets: int
    solve_failures: int


@dataclass
class FiniteOptimizedFeasibilityConfig:
    target_factors: tuple[float, ...] = (0.8, 1.2)
    min_delta_abs: float = 1e-9
    zero_delta_scale: float = 0.01
    zero_angle_delta: float = math.radians(5.0)
    solver_failure_gain: float = 1.0


def _pred_relation_constraints(sketch: datalib.Sketch) -> dict[str, datalib.Constraint]:
    return {
        cid: copy.deepcopy(c)
        for cid, c in sketch.constraints.items()
        if _is_relational(c)
    }


def _target_for_factor(
    probe: datalib.Constraint,
    q0_gt: float,
    factor: float,
    bbox: float,
    config: FiniteOptimizedFeasibilityConfig,
) -> float:
    target = q0_gt * factor
    if abs(target - q0_gt) >= config.min_delta_abs:
        return target

    sign = -1.0 if factor < 1.0 else 1.0
    if probe.constraint_type.name == "Angle":
        return q0_gt + sign * config.zero_angle_delta
    return q0_gt + sign * max(bbox * config.zero_delta_scale, config.min_delta_abs)


def _build_optimized_solve_sketch(
    pred_sketch: datalib.Sketch,
    probe: datalib.Constraint,
    target_value: float,
) -> tuple[datalib.Sketch, int]:
    constraints = _pred_relation_constraints(pred_sketch)
    new_cid = "_finite_optimized_target"
    i = 0
    while new_cid in constraints:
        new_cid = f"_finite_optimized_target_{i}"
        i += 1
    constraints[new_cid] = _set_quantity(
        copy.deepcopy(probe), target_value
    )._replace(identifier=new_cid)
    return datalib.Sketch(
        entities=copy.deepcopy(pred_sketch.entities),
        constraints=constraints,
    ), len(constraints) - 1


def compute_finite_optimized_feasibility(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    *,
    edit_weights: Sequence[float] | None = None,
    config: FiniteOptimizedFeasibilityConfig | None = None,
    solver_config: HistCADConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> FiniteOptimizedFeasibilityResult:
    """Compute the finite optimized reachability diagnostic.

    The finite target is derived from the source GT probe value by multiplying
    it by each configured target factor.  The solve itself starts from Pred's
    current geometry and enforces only Pred relations plus the target probe.
    """
    cfg = config or FiniteOptimizedFeasibilityConfig()
    solver_cfg = solver_config or HistCADConfig()
    if edit_weights is None:
        edit_weights = [1.0] * len(edit_probes)
    if len(edit_weights) != len(edit_probes):
        raise ValueError("len(edit_weights) must match len(edit_probes)")

    gt_state = SketchState(gt_sketch, dtype=dtype)
    pred_state = SketchState(pred_sketch, dtype=dtype)
    bbox = _bbox_diag(gt_sketch, dtype=dtype)
    n_pred_dim = sum(1 for c in pred_sketch.constraints.values() if _is_dimensional(c))
    n_pred_rel = sum(1 for c in pred_sketch.constraints.values() if _is_relational(c))

    total = 0.0
    total_gain = 0.0
    total_leak = 0.0
    total_weight = 0.0
    skipped = 0
    solve_failures = 0
    per_probe: list[FiniteOptimizedFeasibilityProbeResult] = []

    for qi, (probe, weight) in enumerate(zip(edit_probes, edit_weights)):
        sig = _constraint_signature(probe)
        q0_gt = _evaluate_observable(probe, gt_state, context)
        q0_pred = _evaluate_observable(probe, pred_state, context)

        for factor in cfg.target_factors:
            if q0_gt is None or q0_pred is None:
                reason = "probe_unevaluable"
                skipped += 1
                per_probe.append(FiniteOptimizedFeasibilityProbeResult(
                    probe_index=qi,
                    probe_signature=sig,
                    target_factor=factor,
                    q0_gt=float("nan") if q0_gt is None else q0_gt,
                    q0_pred=float("nan") if q0_pred is None else q0_pred,
                    target_value=float("nan"),
                    delta_value=float("nan"),
                    solved=False,
                    solve_status=reason,
                    solve_residual=float("inf"),
                    achieved_value=None,
                    gain=math.nan,
                    leak=math.nan,
                    distance=math.nan,
                    n_relation_rows=0,
                    n_pred_relations=n_pred_rel,
                    skipped=True,
                    skip_reason=reason,
                ))
                continue

            target = _target_for_factor(probe, q0_gt, factor, bbox, cfg)
            delta_abs = abs(target - q0_pred)
            if delta_abs < cfg.min_delta_abs:
                reason = "near_zero_pred_edit"
                skipped += 1
                per_probe.append(FiniteOptimizedFeasibilityProbeResult(
                    probe_index=qi,
                    probe_signature=sig,
                    target_factor=factor,
                    q0_gt=q0_gt,
                    q0_pred=q0_pred,
                    target_value=target,
                    delta_value=target - q0_pred,
                    solved=False,
                    solve_status=reason,
                    solve_residual=float("inf"),
                    achieved_value=None,
                    gain=math.nan,
                    leak=math.nan,
                    distance=math.nan,
                    n_relation_rows=0,
                    n_pred_relations=n_pred_rel,
                    skipped=True,
                    skip_reason=reason,
                ))
                continue

            solve_sketch, solve_n_pred_rel = _build_optimized_solve_sketch(
                pred_sketch, probe, target
            )
            solved_state, status, residual, _iters = _newton_solve(
                solve_sketch, solver_cfg, context, dtype=dtype
            )
            solved = status == "converged" and solved_state is not None
            if not solved:
                solve_failures += 1
                achieved = None
                gain = cfg.solver_failure_gain
                leak = 0.0
                n_relation_rows = 0
            else:
                achieved = _evaluate_observable(probe, solved_state, context)
                gain = (
                    cfg.solver_failure_gain
                    if achieved is None
                    else abs(achieved - target) / max(delta_abs, cfg.min_delta_abs)
                )
                leak, n_relation_rows = _relation_leak_per_delta(
                    relation_probes,
                    pred_state,
                    solved_state,
                    context,
                    delta_abs,
                )

            distance = math.sqrt(gain * gain + leak * leak)
            per_probe.append(FiniteOptimizedFeasibilityProbeResult(
                probe_index=qi,
                probe_signature=sig,
                target_factor=factor,
                q0_gt=q0_gt,
                q0_pred=q0_pred,
                target_value=target,
                delta_value=target - q0_pred,
                solved=solved,
                solve_status=status,
                solve_residual=residual,
                achieved_value=achieved,
                gain=gain,
                leak=leak,
                distance=distance,
                n_relation_rows=n_relation_rows,
                n_pred_relations=solve_n_pred_rel,
            ))
            total += float(weight) * distance
            total_gain += float(weight) * gain
            total_leak += float(weight) * leak
            total_weight += float(weight)

    return FiniteOptimizedFeasibilityResult(
        distance=total / total_weight if total_weight > 0 else 0.0,
        gain=total_gain / total_weight if total_weight > 0 else 0.0,
        leak=total_leak / total_weight if total_weight > 0 else 0.0,
        per_probe=per_probe,
        n_pred_dimensions=n_pred_dim,
        n_pred_relations=n_pred_rel,
        n_source_relations=len(relation_probes),
        n_probe_targets=len(per_probe),
        skipped_probe_targets=skipped,
        solve_failures=solve_failures,
    )
