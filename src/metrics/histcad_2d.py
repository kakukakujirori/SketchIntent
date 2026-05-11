"""
histcad_2d.py — 2D port of HistCAD-style edit-based evaluation.

HistCAD (arXiv 2602.19171) は 3D parametric CAD sequence に対して、
parameter edit 後に target が達成され、preserved constraints が保たれるかを
binary 的に評価する。本モジュールはその核心の 2D 版を実装する。

Per-sketch metric trio
----------------------
各 (edit probe q, sign s ∈ {±1}) について、

  1. q の現在値 q0 を sketch 上で評価
  2. target q* = q0 + s · Δ を設定 (Δ は config.delta_basis で決定)
  3. q の制約をターゲット q* に書き換えた sketch を Newton-style に解く
  4. **ER (Edit Reachability)**:
        Newton が converge した上で |h(x*) - q*| ≤ tol
  5. **cPCSR (Conditional Preserved Constraint Satisfaction Rate)**:
        ER=True 時のみ、preserved relation 集合 R_pres について
        |r_i(x*)| < tol_i の割合を計算
  6. **OES (Overall Editable Success)**:
        ER × cPCSR (ER=False 時は 0 として probe 全体で平均)

Per-pair API (compute_histcad_2d_pair)
--------------------------------------
Pred を solve 対象、GT 由来の rel constraint を preserved 集合とする
HistCAD 風 pair-wise scoring。`relation_source` を変えると intersection /
union / pred-side も選べる。

設計の意図
----------
- `HistCADConfig` で全閾値・編集スケール・relation source・newton 設定を外出し。
- 提案 metric (edit_response_distance) と同じ probe / signature 機構を再利用し、
  q を共通 edit probe として扱う (parameterization-invariance のため)。
- 提案 metric が連続応答距離なのに対し、本モジュールは binary preservation。
  両者を同一実験で並べて出すことで、HistCAD-style では同点だが提案 metric では
  差が出る case を抽出するのが研究目的 (notebooks/plan_histcad.md §13.2)。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

import torch
import sketchgraphs.data as datalib
from sketchgraphs.data import QuantityParameter

from src.sketch_handlers.sketch_solver import SketchState, _StateWithQ
from src.metrics.edit_response_distance import (
    constraint_signature,
    _ref_strings,
    _is_relational,
)


VALUE_WATERMARK = "VALUE_ASSIGNED:"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HistCADConfig:
    """All knobs for HistCAD-style scoring. Defaults are the proposed defaults;
    every field is intended to be tweakable from the experiment driver."""

    # --- edit perturbation ---
    delta_basis: str = "bbox_diag"        # "bbox_diag" | "current" | "absolute"
    delta_scale: float = 0.05             # used for "bbox_diag" / "current"
    delta_absolute: float = 1.0           # used for "absolute"
    delta_signs: tuple[float, ...] = (-1.0, +1.0)
    delta_min_abs: float = 1e-6           # floor on |Δ| to avoid 0-edits

    # --- Newton solver ---
    newton_max_iter: int = 25
    newton_tol: float = 1e-6              # ||r|| convergence threshold
    newton_damping: float = 1.0           # base step scaling
    newton_step_max: float = 1e3          # cap on per-component |δq|
    newton_lstsq_rcond: float | None = None
    newton_diverge_factor: float = 5.0    # bail if ||r||_t > factor * ||r||_{t-1}
    newton_stall_ratio: float = 0.99      # if ||r||_t > stall_ratio * ||r||_{t-1} ...
    newton_stall_iters: int = 3           # ... for this many consecutive iters, bail

    # --- Edit Reachability (ER) judgement ---
    er_residual_tol: float = 1e-4         # ||r(x*)|| ≤ tol post-solve
    er_target_tol: float = 1e-3           # |h(x*) - q*| / scale ≤ tol
    er_check_target: bool = True
    er_check_residual: bool = True

    # --- Conditional Preserved Constraint Satisfaction (cPCSR) ---
    relation_tol_default: float = 1e-3
    relation_tol_per_type: dict[str, float] = field(default_factory=dict)

    # --- pair-wise: which relation set is "preserved" ---
    relation_source: str = "gt"           # "gt" | "pred" | "intersection" | "union"


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class HistCADProbeResult:
    probe_index: int
    probe_signature: tuple
    delta_sign: float
    delta_value: float
    q0: float
    target_value: float
    er: bool
    er_reason: str
    converged: bool
    final_residual: float
    achieved_value: float | None
    relations_preserved: list[bool]
    cpcsr: float | None                   # None when ER=False (conditional)


@dataclass
class HistCADResult:
    per_probe: list[HistCADProbeResult]
    er_rate: float                        # mean(er over all probe×sign trials)
    cpcsr_rate: float                     # mean cPCSR over only-ER trials
    oes_rate: float                       # mean(er * cpcsr; cpcsr→0 if ER=False)
    n_probes: int                         # total trials
    n_relations: int                      # |R_pres|


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_diag(sketch: datalib.Sketch, dtype: torch.dtype = torch.float64) -> float:
    state = SketchState(sketch, dtype=dtype)
    points: list[torch.Tensor] = []
    for eid, ent in sketch.entities.items():
        if isinstance(ent, datalib.Line):
            points.append(state.get_point(f"{eid}.start").detach())
            points.append(state.get_point(f"{eid}.end").detach())
        elif isinstance(ent, datalib.Circle):
            points.append(state.get_point(f"{eid}.center").detach())
        elif isinstance(ent, datalib.Arc):
            points.append(state.get_point(f"{eid}.start").detach())
            points.append(state.get_point(f"{eid}.end").detach())
            points.append(state.get_point(f"{eid}.center").detach())
        elif isinstance(ent, datalib.Point):
            points.append(state.get_point(eid).detach())
    if not points:
        return 1.0
    P = torch.stack(points)
    diag = float(torch.linalg.norm(P.max(dim=0).values - P.min(dim=0).values))
    return max(diag, 1e-6)


def _refs_present(constraint: datalib.Constraint, state: SketchState) -> bool:
    for ref in _ref_strings(constraint):
        eid = ref.split(".", 1)[0]
        if eid.startswith("-"):
            continue                       # external axis is fine
        if eid not in state.slices:
            return False
    return True


def _evaluate_observable(
    constraint: datalib.Constraint,
    state: SketchState,
    context,
) -> float | None:
    """h(x) for a dim constraint = residual + target (target read from constraint).

    Returns None when the handler / refs / value are unavailable on this state.
    """
    handler = context.get_constraint_handler(constraint.constraint_type.name)
    if handler is None or not _refs_present(constraint, state):
        return None
    try:
        r = handler.residual(constraint, state).reshape(-1)
        target = handler.get_constraint_props(constraint)["value"]
        return float(r[0]) + float(target)
    except Exception:
        return None


def _set_quantity(constraint: datalib.Constraint, target_value: float) -> datalib.Constraint:
    """Return a copy of `constraint` whose Quantity parameter (length/angle)
    has its value replaced by `target_value`. Other parameters untouched."""
    new_params = []
    replaced = False
    for p in constraint.parameters:
        if isinstance(p, QuantityParameter) and p.parameterId in ("length", "angle"):
            new_params.append(
                QuantityParameter(
                    parameterId=p.parameterId,
                    value=target_value,
                    expression=VALUE_WATERMARK + str(target_value),
                )
            )
            replaced = True
        else:
            new_params.append(p)
    if not replaced:
        # fallback: append a length parameter (rare — most dim constraints already have one)
        new_params.append(
            QuantityParameter(
                parameterId="length",
                value=target_value,
                expression=VALUE_WATERMARK + str(target_value),
            )
        )
    return constraint._replace(parameters=new_params)


def _apply_edit(
    sketch: datalib.Sketch,
    probe: datalib.Constraint,
    target_value: float,
    new_cid_seed: str = "_histcad_edit",
) -> datalib.Sketch:
    """Deep-copy `sketch`, then either rewrite the matching probe constraint's
    target to `target_value`, or insert `probe` as a new constraint with that
    target if no signature match exists."""
    new_sketch = copy.deepcopy(sketch)
    sig = constraint_signature(probe)

    matched_cid: str | None = None
    for cid, c in new_sketch.constraints.items():
        if constraint_signature(c) == sig:
            matched_cid = cid
            break

    if matched_cid is not None:
        new_sketch.constraints[matched_cid] = _set_quantity(
            new_sketch.constraints[matched_cid], target_value
        )
    else:
        new_cid = new_cid_seed
        i = 0
        while new_cid in new_sketch.constraints:
            new_cid = f"{new_cid_seed}_{i}"
            i += 1
        new_constraint = _set_quantity(probe, target_value)._replace(identifier=new_cid)
        new_sketch.constraints[new_cid] = new_constraint
    return new_sketch


def _stack_residuals_and_jacobian(
    sketch: datalib.Sketch,
    state: SketchState,
    context,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Stack r(state.q) and J = ∂r/∂q for every constraint in `sketch`.

    Implementation note: builds one composite residual function and computes
    its Jacobian in a single `torch.func.jacfwd` call. This is materially
    faster than per-constraint jacfwd because torch.func batches a single set
    of n_dof tangent vectors through one forward pass over all constraints,
    instead of repeating that setup for each constraint.

    Constraints with missing handlers / unresolved refs are skipped before the
    composite is built. Per-constraint runtime errors inside the composite are
    swallowed by returning a zero residual for that constraint.
    """
    n_dof = state.num_dof()
    if n_dof == 0:
        return None, None

    # Pre-resolve handlers / sizes to keep the composite fn straight-line.
    selected: list[tuple[datalib.Constraint, object, int]] = []
    for cid, c in sketch.constraints.items():
        handler = context.get_constraint_handler(c.constraint_type.name)
        if handler is None or not _refs_present(c, state):
            continue
        # Probe the residual size on the current state once so the composite fn
        # can return a fixed-shape zero on errors.
        try:
            r0 = handler.residual(c, state).reshape(-1)
        except Exception:
            continue
        selected.append((c, handler, int(r0.shape[0])))

    if not selected:
        return None, None

    def fn_all(q: torch.Tensor) -> torch.Tensor:
        s = _StateWithQ(state, q)
        parts: list[torch.Tensor] = []
        for _c, _h, _k in selected:
            try:
                r = _h.residual(_c, s).reshape(-1)
                if r.shape[0] != _k:
                    r = torch.zeros(_k, dtype=q.dtype, device=q.device)
            except Exception:
                r = torch.zeros(_k, dtype=q.dtype, device=q.device)
            parts.append(r)
        return torch.cat(parts)

    q0 = state.q.detach()
    try:
        R = fn_all(q0)
        J = torch.func.jacfwd(fn_all)(q0)
    except Exception:
        return None, None
    if J.ndim == 1:
        J = J.unsqueeze(0)
    return R, J


def _newton_solve(
    sketch: datalib.Sketch,
    config: HistCADConfig,
    context,
    dtype: torch.dtype = torch.float64,
) -> tuple[SketchState | None, str, float, int]:
    """Damped Gauss-Newton on the full constraint system.

    Updates an internal q vector; returns a SketchState whose q is set to the
    final iterate. status ∈ {"converged", "max_iter", "diverged", "lstsq_failed",
    "no_dof", "no_residuals"}.
    """
    state = SketchState(sketch, dtype=dtype)
    if state.num_dof() == 0:
        return state, "no_dof", 0.0, 0

    q = state.q.detach().clone()
    last_norm = float("inf")
    stall_streak = 0

    for it in range(config.newton_max_iter):
        # Evaluate residuals + jacobian at the current q.
        state.q.data = q
        R, J = _stack_residuals_and_jacobian(sketch, state, context)
        if R is None:
            return state, "no_residuals", float("inf"), it

        norm = float(torch.linalg.norm(R))
        if norm < config.newton_tol:
            return state, "converged", norm, it
        if it > 0:
            if norm > config.newton_diverge_factor * last_norm:
                return state, "diverged", norm, it
            if norm > config.newton_stall_ratio * last_norm:
                stall_streak += 1
                if stall_streak >= config.newton_stall_iters:
                    return state, "stalled", norm, it
            else:
                stall_streak = 0
        last_norm = norm

        try:
            sol = torch.linalg.lstsq(
                J, R, rcond=config.newton_lstsq_rcond
            ).solution
        except Exception:
            return state, "lstsq_failed", norm, it
        delta = -config.newton_damping * sol
        delta = torch.clamp(delta, -config.newton_step_max, config.newton_step_max)
        q = q + delta

    state.q.data = q
    return state, "max_iter", last_norm, config.newton_max_iter


def _evaluate_relations(
    state: SketchState,
    relation_probes: Sequence[datalib.Constraint],
    context,
    config: HistCADConfig,
) -> list[bool]:
    """For each relation probe, mark True iff its residual norm < tol on `state`."""
    out: list[bool] = []
    for r in relation_probes:
        handler = context.get_constraint_handler(r.constraint_type.name)
        if handler is None or not _refs_present(r, state):
            out.append(False)
            continue
        try:
            res = handler.residual(r, state).reshape(-1)
            tol = config.relation_tol_per_type.get(
                r.constraint_type.name, config.relation_tol_default
            )
            out.append(bool(float(torch.linalg.norm(res)) < tol))
        except Exception:
            out.append(False)
    return out


def _compute_delta(
    config: HistCADConfig,
    sign: float,
    q0: float,
    bbox_diag: float,
) -> float:
    if config.delta_basis == "bbox_diag":
        d = sign * config.delta_scale * bbox_diag
    elif config.delta_basis == "current":
        d = sign * config.delta_scale * abs(q0)
    elif config.delta_basis == "absolute":
        d = sign * config.delta_absolute
    else:
        raise ValueError(f"unknown delta_basis: {config.delta_basis}")
    if abs(d) < config.delta_min_abs:
        d = config.delta_min_abs * (1.0 if sign >= 0 else -1.0)
    return d


# ---------------------------------------------------------------------------
# Single-sketch evaluation
# ---------------------------------------------------------------------------

def compute_histcad_2d(
    sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    config: HistCADConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> HistCADResult:
    """Compute (ER, cPCSR, OES) on a single sketch given exogenous probe sets.

    For each probe q × each sign s ∈ config.delta_signs:
      - read q's current observable q0 on `sketch`
      - target q* = q0 + Δ(s)
      - rewrite (or add) the probe constraint with that target, then solve
      - judge ER from solver convergence + target attainment
      - if ER, evaluate residual of every relation probe at the solved state
    """
    cfg = config or HistCADConfig()
    base_state = SketchState(sketch, dtype=dtype)
    bbox = _bbox_diag(sketch, dtype=dtype)

    per_probe: list[HistCADProbeResult] = []
    er_count = 0
    cpcsr_sum_for_er = 0.0
    oes_sum = 0.0
    n_relations = len(relation_probes)

    for qi, q in enumerate(edit_probes):
        sig = constraint_signature(q)
        q0 = _evaluate_observable(q, base_state, context)
        if q0 is None:
            for sign in cfg.delta_signs:
                per_probe.append(HistCADProbeResult(
                    probe_index=qi, probe_signature=sig,
                    delta_sign=sign, delta_value=0.0, q0=float("nan"),
                    target_value=float("nan"),
                    er=False, er_reason="probe_unevaluable_on_sketch",
                    converged=False, final_residual=float("inf"),
                    achieved_value=None,
                    relations_preserved=[False] * n_relations,
                    cpcsr=None,
                ))
            continue

        for sign in cfg.delta_signs:
            delta = _compute_delta(cfg, sign, q0, bbox)
            target = q0 + delta
            edited = _apply_edit(sketch, q, target)

            final_state, status, residual, _it = _newton_solve(
                edited, cfg, context, dtype=dtype
            )

            converged = status == "converged"
            er = True
            er_reason = ""
            achieved: float | None = None

            if cfg.er_check_residual and not converged:
                er = False
                er_reason = f"newton:{status}"
            if er and final_state is not None and cfg.er_check_target:
                achieved = _evaluate_observable(q, final_state, context)
                if achieved is None:
                    er = False
                    er_reason = "obs_eval_failed"
                else:
                    miss = abs(achieved - target) / max(bbox, 1e-12)
                    if miss > cfg.er_target_tol:
                        er = False
                        er_reason = f"target_miss:{miss:.3g}"

            relations_preserved: list[bool]
            if final_state is not None:
                relations_preserved = _evaluate_relations(
                    final_state, relation_probes, context, cfg
                )
            else:
                relations_preserved = [False] * n_relations

            cpcsr: float | None
            if er:
                if n_relations == 0:
                    cpcsr = 1.0                         # vacuous preservation
                else:
                    cpcsr = sum(relations_preserved) / n_relations
                er_count += 1
                cpcsr_sum_for_er += cpcsr
                oes_sum += cpcsr
            else:
                cpcsr = None
                # OES contribution = 0

            per_probe.append(HistCADProbeResult(
                probe_index=qi, probe_signature=sig,
                delta_sign=sign, delta_value=delta, q0=q0,
                target_value=target,
                er=er, er_reason=er_reason,
                converged=converged, final_residual=residual,
                achieved_value=achieved,
                relations_preserved=relations_preserved,
                cpcsr=cpcsr,
            ))

    n_total = len(per_probe)
    er_rate = (er_count / n_total) if n_total else 0.0
    cpcsr_rate = (cpcsr_sum_for_er / er_count) if er_count else 0.0
    oes_rate = (oes_sum / n_total) if n_total else 0.0

    return HistCADResult(
        per_probe=per_probe,
        er_rate=er_rate,
        cpcsr_rate=cpcsr_rate,
        oes_rate=oes_rate,
        n_probes=n_total,
        n_relations=n_relations,
    )


# ---------------------------------------------------------------------------
# Pair-wise (Pred vs GT-derived preserved relations)
# ---------------------------------------------------------------------------

def select_preserved_relations(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    relation_source: str,
) -> list[datalib.Constraint]:
    gt_rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]
    pred_rels = [c for c in pred_sketch.constraints.values() if _is_relational(c)]

    if relation_source == "gt":
        return gt_rels
    if relation_source == "pred":
        return pred_rels
    if relation_source == "intersection":
        gt_sigs = {constraint_signature(c) for c in gt_rels}
        pred_sigs = {constraint_signature(c) for c in pred_rels}
        common = gt_sigs & pred_sigs
        return [c for c in gt_rels if constraint_signature(c) in common]
    if relation_source == "union":
        seen: set[tuple] = set()
        out: list[datalib.Constraint] = []
        for c in list(gt_rels) + list(pred_rels):
            sig = constraint_signature(c)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(c)
        return out
    raise ValueError(f"unknown relation_source: {relation_source}")


def compute_histcad_2d_pair(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    context,
    config: HistCADConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> HistCADResult:
    """Pair-wise HistCAD: solve edits on Pred, judge cPCSR using
    config.relation_source-derived preserved relations."""
    cfg = config or HistCADConfig()
    rels = select_preserved_relations(pred_sketch, gt_sketch, cfg.relation_source)
    return compute_histcad_2d(
        sketch=pred_sketch,
        edit_probes=edit_probes,
        relation_probes=rels,
        context=context,
        config=cfg,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# FreeCAD-backed solver path (uses the native Sketcher solver via the
# converter; much more robust than the internal Newton implementation).
# ---------------------------------------------------------------------------

def _solve_via_freecad(sketch: datalib.Sketch, converter):
    """Try to solve `sketch` via FreeCAD's native sketcher solver.

    Bypasses `complete_constraint_values` (which `converter.to_freecad`
    calls internally) because that helper RECOMPUTES dim constraint values
    from current geometry and would clobber the post-edit target written by
    `_apply_edit`. We therefore replicate the converter's two-stage build
    (primitives → constraints → recompute → solve) directly so the edited
    target survives into FreeCAD.

    Returns (post_solved_sketch, status_string).
    """
    from src.sketch_handlers.freecad_utils import ensure_freecad_imported
    FreeCAD, _, _ = ensure_freecad_imported()
    doc = None
    try:
        doc, fc_sketch, prim_map = converter.to_freecad_primitives(
            sketch, doc_name="Sketch", skip_invalid_primitives=False,
        )
        _, invalid = converter.to_freecad_constraints(
            sketch, fc_sketch, prim_map, skip_invalid_constraints=False,
        )
        doc.recompute()
        rc = int(fc_sketch.solve())
        if rc != 0:
            return None, f"solver_rc:{rc}"
        if invalid:
            return None, f"invalid_constraints:{len(invalid)}"
        return converter.from_freecad(fc_sketch), "converged"
    except Exception as e:
        return None, ("exception:" + str(e))[:120]
    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                pass


def compute_histcad_2d_freecad(
    sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    converter,
    config: HistCADConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> HistCADResult:
    """Same protocol as `compute_histcad_2d`, but solving each edit through
    FreeCAD's native Sketcher rather than the internal Newton.

    The internal Newton implementation is fragile on near-fully-constrained
    sketches (the `--newton-stall-iters` heuristic bails on systems FreeCAD
    handles trivially). Use this variant when the per-pair `histcad_er` from
    `compute_histcad_2d_pair` is suspiciously low.
    """
    cfg = config or HistCADConfig()
    base_state = SketchState(sketch, dtype=dtype)
    bbox = _bbox_diag(sketch, dtype=dtype)

    per_probe: list[HistCADProbeResult] = []
    er_count = 0
    cpcsr_sum = 0.0
    oes_sum = 0.0
    n_relations = len(relation_probes)

    for qi, q in enumerate(edit_probes):
        sig = constraint_signature(q)
        q0 = _evaluate_observable(q, base_state, context)
        if q0 is None:
            for sign in cfg.delta_signs:
                per_probe.append(HistCADProbeResult(
                    probe_index=qi, probe_signature=sig,
                    delta_sign=sign, delta_value=0.0, q0=float("nan"),
                    target_value=float("nan"),
                    er=False, er_reason="probe_unevaluable_on_sketch",
                    converged=False, final_residual=float("inf"),
                    achieved_value=None,
                    relations_preserved=[False] * n_relations,
                    cpcsr=None,
                ))
            continue

        for sign in cfg.delta_signs:
            delta = _compute_delta(cfg, sign, q0, bbox)
            target = q0 + delta
            edited = _apply_edit(sketch, q, target)
            new_sketch, status = _solve_via_freecad(edited, converter)

            converged = status == "converged"
            er = converged
            er_reason = "" if converged else f"freecad:{status}"
            achieved: float | None = None
            new_state: SketchState | None = (
                SketchState(new_sketch, dtype=dtype) if new_sketch is not None else None
            )

            if er and cfg.er_check_target and new_state is not None:
                achieved = _evaluate_observable(q, new_state, context)
                if achieved is None:
                    er = False
                    er_reason = "obs_eval_failed"
                else:
                    miss = abs(achieved - target) / max(bbox, 1e-12)
                    if miss > cfg.er_target_tol:
                        er = False
                        er_reason = f"target_miss:{miss:.3g}"

            relations_preserved = (
                _evaluate_relations(new_state, relation_probes, context, cfg)
                if new_state is not None else [False] * n_relations
            )

            cpcsr: float | None
            if er:
                cpcsr = (sum(relations_preserved) / n_relations) if n_relations else 1.0
                er_count += 1
                cpcsr_sum += cpcsr
                oes_sum += cpcsr
            else:
                cpcsr = None

            per_probe.append(HistCADProbeResult(
                probe_index=qi, probe_signature=sig,
                delta_sign=sign, delta_value=delta, q0=q0,
                target_value=target,
                er=er, er_reason=er_reason,
                converged=converged, final_residual=0.0,
                achieved_value=achieved,
                relations_preserved=relations_preserved,
                cpcsr=cpcsr,
            ))

    n_total = len(per_probe)
    return HistCADResult(
        per_probe=per_probe,
        er_rate=(er_count / n_total) if n_total else 0.0,
        cpcsr_rate=(cpcsr_sum / er_count) if er_count else 0.0,
        oes_rate=(oes_sum / n_total) if n_total else 0.0,
        n_probes=n_total,
        n_relations=n_relations,
    )


def compute_histcad_2d_pair_freecad(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    context,
    converter,
    config: HistCADConfig | None = None,
    dtype: torch.dtype = torch.float64,
) -> HistCADResult:
    """Pair-wise HistCAD evaluation using the FreeCAD solver."""
    cfg = config or HistCADConfig()
    rels = select_preserved_relations(pred_sketch, gt_sketch, cfg.relation_source)
    return compute_histcad_2d_freecad(
        sketch=pred_sketch,
        edit_probes=edit_probes,
        relation_probes=rels,
        context=context,
        converter=converter,
        config=cfg,
        dtype=dtype,
    )
