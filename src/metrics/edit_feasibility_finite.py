"""Finite native-dimension feasibility metric for 2D sketch constraints.

This module is the finite-solve counterpart of the first-order ``d_feas``
diagnostic used in ``notebooks/exp_constrained_response_metric.py``.

The key distinction from HistCAD-style hard edits is that we do not insert the
external edit probe ``q`` as a new constraint.  Instead, for each requested
finite edit ``dq`` we compute the native dimension update

    dp* = pinv(J_q R_P) dq

from the linear response of the predicted sketch, apply those updates to the
predicted sketch's own dimensional constraints, and then let FreeCAD solve.
This preserves the "are the predicted dimensions useful edit knobs?" semantics
of linear ``d_feas`` under a finite solver.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter

from src.metrics.edit_response_distance import (
    _abs_pinv,
    compute_native_response,
    constraint_jacobian,
)
from src.metrics.histcad_2d import (
    HistCADConfig,
    _bbox_diag,
    _compute_delta,
    _evaluate_observable,
    _set_quantity,
    _solve_via_freecad,
)
from src.sketch_handlers.sketch_solver import SketchState


@dataclass
class NativeFiniteFeasibilityProbeResult:
    probe_index: int
    probe_signature: tuple
    delta_sign: float
    delta_value: float
    q0: float
    target_value: float
    solved: bool
    solve_status: str
    achieved_value: float | None
    gain: float
    leak: float
    distance: float
    n_relation_rows: int
    n_native_dimensions: int
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class NativeFiniteFeasibilityResult:
    distance: float
    gain: float
    leak: float
    per_probe: list[NativeFiniteFeasibilityProbeResult]
    n_dim_pred: int
    n_relations: int
    n_probe_signs: int
    skipped_probe_signs: int
    solve_failures: int


def _ref_strings(constraint: datalib.Constraint) -> tuple[str, ...]:
    return tuple(
        str(p.value)
        for p in constraint.parameters
        if isinstance(p, LocalReferenceParameter)
    )


def _refs_present(constraint: datalib.Constraint, state: SketchState) -> bool:
    for ref in _ref_strings(constraint):
        eid = ref.split(".", 1)[0]
        if eid.startswith("-"):
            continue
        if eid not in state.slices:
            return False
    return True


def _constraint_signature(constraint: datalib.Constraint) -> tuple:
    return (
        constraint.constraint_type.name,
        tuple(sorted(_ref_strings(constraint))),
    )


def _residual_vector(
    constraint: datalib.Constraint,
    state: SketchState,
    context,
) -> torch.Tensor | None:
    handler = context.get_constraint_handler(constraint.constraint_type.name)
    if handler is None or not _refs_present(constraint, state):
        return None
    try:
        return handler.residual(constraint, state).detach().reshape(-1)
    except Exception:
        return None


def _relation_leak_per_delta(
    relation_probes: Sequence[datalib.Constraint],
    base_state: SketchState,
    solved_state: SketchState,
    context,
    delta_abs: float,
) -> tuple[float, int]:
    """Return row-balanced finite relation residual change per unit edit."""
    sq_sum = 0.0
    n_rows = 0
    denom = max(delta_abs, 1e-12)
    for relation in relation_probes:
        r0 = _residual_vector(relation, base_state, context)
        r1 = _residual_vector(relation, solved_state, context)
        if r0 is None or r1 is None or r0.shape != r1.shape:
            continue
        diff = (r1 - r0) / denom
        sq_sum += float(torch.linalg.norm(diff) ** 2)
        n_rows += int(diff.numel())
    if n_rows == 0:
        return 0.0, 0
    return math.sqrt(sq_sum / float(n_rows)), n_rows


def _apply_native_dimension_delta(
    sketch: datalib.Sketch,
    dim_constraint_ids: Sequence[str],
    dp: torch.Tensor,
    context,
) -> datalib.Sketch:
    """Shift each native dimensional constraint target by the corresponding dp."""
    out = copy.deepcopy(sketch)
    flat_dp = dp.detach().cpu().reshape(-1).tolist()
    for cid, delta in zip(dim_constraint_ids, flat_dp):
        constraint = out.constraints.get(cid)
        if constraint is None:
            continue
        handler = context.get_constraint_handler(constraint.constraint_type.name)
        if handler is None:
            continue
        try:
            value = float(handler.get_constraint_props(constraint)["value"])
        except Exception:
            continue
        out.constraints[cid] = _set_quantity(constraint, value + float(delta))
    return out


def compute_native_finite_feasibility(
    pred_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    converter,
    config: HistCADConfig | None = None,
    edit_weights: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-9,
    solver_failure_gain: float = 1.0,
) -> NativeFiniteFeasibilityResult:
    """Finite native-dimension version of ``d_feas``.

    For each edit probe and delta sign:
      1. compute ``dp* = pinv(J_q R_P) dq`` using the predicted sketch's native
         dimension response;
      2. apply ``dp*`` to the predicted sketch's existing dimensional targets;
      3. solve with FreeCAD;
      4. score finite gain miss and finite relation residual change.

    Solve failures contribute ``gain=solver_failure_gain`` and ``leak=0`` so
    that failed finite edits are penalized without dropping the probe.
    """
    cfg = config or HistCADConfig()
    if edit_weights is None:
        edit_weights = [1.0] * len(edit_probes)
    if len(edit_weights) != len(edit_probes):
        raise ValueError("len(edit_weights) must match len(edit_probes)")

    native = compute_native_response(pred_sketch, context, dtype=dtype, eps=eps)
    base_state = native.state
    bbox = _bbox_diag(pred_sketch, dtype=dtype)

    total = 0.0
    total_gain = 0.0
    total_leak = 0.0
    total_weight = 0.0
    skipped = 0
    solve_failures = 0
    per_probe: list[NativeFiniteFeasibilityProbeResult] = []

    for qi, (probe, weight) in enumerate(zip(edit_probes, edit_weights)):
        sig = _constraint_signature(probe)
        q0 = _evaluate_observable(probe, base_state, context)
        Jq = constraint_jacobian(probe, base_state, context)
        if q0 is None or Jq is None or Jq.shape[0] != 1:
            reason = "probe_unevaluable" if q0 is None or Jq is None else "multirow_probe"
            for sign in cfg.delta_signs:
                skipped += 1
                per_probe.append(NativeFiniteFeasibilityProbeResult(
                    probe_index=qi,
                    probe_signature=sig,
                    delta_sign=sign,
                    delta_value=0.0,
                    q0=float("nan") if q0 is None else q0,
                    target_value=float("nan"),
                    solved=False,
                    solve_status=reason,
                    achieved_value=None,
                    gain=math.nan,
                    leak=math.nan,
                    distance=math.nan,
                    n_relation_rows=0,
                    n_native_dimensions=len(native.dim_constraint_ids),
                    skipped=True,
                    skip_reason=reason,
                ))
            continue

        for sign in cfg.delta_signs:
            delta = _compute_delta(cfg, sign, q0, bbox)
            delta_abs = abs(delta)
            target = q0 + delta

            if native.R.shape[1] == 0:
                gain = solver_failure_gain
                leak = 0.0
                distance = math.sqrt(gain * gain + leak * leak)
                per_probe.append(NativeFiniteFeasibilityProbeResult(
                    probe_index=qi,
                    probe_signature=sig,
                    delta_sign=sign,
                    delta_value=delta,
                    q0=q0,
                    target_value=target,
                    solved=False,
                    solve_status="no_native_dimensions",
                    achieved_value=q0,
                    gain=gain,
                    leak=leak,
                    distance=distance,
                    n_relation_rows=0,
                    n_native_dimensions=0,
                ))
                total += float(weight) * distance
                total_gain += float(weight) * gain
                total_leak += float(weight) * leak
                total_weight += float(weight)
                continue

            A = Jq @ native.R
            dq = torch.tensor([[delta]], dtype=A.dtype, device=A.device)
            dp = _abs_pinv(A, eps=eps) @ dq
            edited = _apply_native_dimension_delta(
                pred_sketch, native.dim_constraint_ids, dp, context
            )
            solved_sketch, status = _solve_via_freecad(edited, converter)

            if solved_sketch is None:
                solve_failures += 1
                gain = solver_failure_gain
                leak = 0.0
                achieved = None
                solved = False
                n_relation_rows = 0
            else:
                solved = True
                solved_state = SketchState(solved_sketch, dtype=dtype)
                achieved = _evaluate_observable(probe, solved_state, context)
                if achieved is None:
                    gain = solver_failure_gain
                else:
                    gain = abs(achieved - target) / max(delta_abs, 1e-12)
                leak, n_relation_rows = _relation_leak_per_delta(
                    relation_probes,
                    base_state,
                    solved_state,
                    context,
                    delta_abs,
                )

            distance = math.sqrt(gain * gain + leak * leak)
            per_probe.append(NativeFiniteFeasibilityProbeResult(
                probe_index=qi,
                probe_signature=sig,
                delta_sign=sign,
                delta_value=delta,
                q0=q0,
                target_value=target,
                solved=solved,
                solve_status=status,
                achieved_value=achieved,
                gain=gain,
                leak=leak,
                distance=distance,
                n_relation_rows=n_relation_rows,
                n_native_dimensions=len(native.dim_constraint_ids),
            ))
            total += float(weight) * distance
            total_gain += float(weight) * gain
            total_leak += float(weight) * leak
            total_weight += float(weight)

    return NativeFiniteFeasibilityResult(
        distance=total / total_weight if total_weight > 0 else 0.0,
        gain=total_gain / total_weight if total_weight > 0 else 0.0,
        leak=total_leak / total_weight if total_weight > 0 else 0.0,
        per_probe=per_probe,
        n_dim_pred=len(native.dim_constraint_ids),
        n_relations=len(relation_probes),
        n_probe_signs=len(per_probe),
        skipped_probe_signs=skipped,
        solve_failures=solve_failures,
    )
