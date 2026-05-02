"""
sketch_solver.py — Differentiable sketch state and Jacobian / null-space utilities.

This module provides:
  - SketchState: wraps a datalib.Sketch and exposes differentiable torch.Tensor
    parameters for every primitive, keyed by the same reference strings used
    in constraint handlers (e.g. "0", "0.start", "0.end", "0.center").
  - compute_jacobian: stacks all constraint residuals and computes dr/dq via
    torch.func.jacfwd at the current geometry (q₀).
  - compute_null_space: extracts ker(J) via SVD.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import sketchgraphs.data as datalib

if TYPE_CHECKING:
    from src.sketch_handlers.constraints.base import ConstraintHandler


# ---------------------------------------------------------------------------
# Parameter layout for each primitive type
# ---------------------------------------------------------------------------
# Line   : [pntX, pntY, dirX, dirY, startParam, endParam]  -> size 6
#   start = pnt + startParam * dir
#   end   = pnt + endParam   * dir
# Circle : [xCenter, yCenter, radius]                       -> size 3
# Arc    : [xCenter, yCenter, radius, startParam, endParam] -> size 5
#   start/end points are derived from xDir=1,yDir=0 frame stored in entity;
#   we treat the parameterisation as: point = center + r*(cos θ, sin θ)
#   where θ_start / θ_end are the startParam / endParam angles (radians).
# Point  : [x, y]                                           -> size 2

_PRIMITIVE_PARAM_SIZE = {
    'Line': 6,
    'Circle': 3,
    'Arc': 5,
    'Point': 2,
}


# ---------------------------------------------------------------------------
# SketchState
# ---------------------------------------------------------------------------

class SketchState:
    """
    Wraps a datalib.Sketch to provide differentiable torch parameter tensors.

    After construction, callers can retrieve tensors via helper methods
    like `get_point`, `get_line_dir`, `get_radius`, etc.  All tensors share
    the underlying parameter vector `self.q`, so computing gradients w.r.t.
    `self.q` propagates through all constraints.

    Attributes
    ----------
    sketch : datalib.Sketch
    q      : torch.Tensor, shape (num_dof,), requires_grad=True
    entity_ids : list[str]   ordered entity IDs (external axes excluded)
    slices : dict[str, slice]  mapping entity_id -> slice in q
    """

    def __init__(self, sketch: datalib.Sketch, dtype: torch.dtype = torch.float64):
        self.sketch = sketch
        self.dtype = dtype

        # Build flat parameter vector q
        params: list[float] = []
        self.slices: dict[str, slice] = {}
        self.entity_ids: list[str] = []

        for eid, entity in sketch.entities.items():
            ename = entity.__class__.__name__
            p = _entity_to_params(entity)
            if p is None:
                continue  # unsupported type – skip
            start = len(params)
            params.extend(p)
            self.slices[eid] = slice(start, len(params))
            self.entity_ids.append(eid)

        self.q = torch.tensor(params, dtype=dtype, requires_grad=False)

    # ------------------------------------------------------------------
    # Low-level slice accessors (return views into self.q)
    # ------------------------------------------------------------------

    def _slice(self, eid: str) -> torch.Tensor:
        sl = self.slices.get(eid)
        if sl is None:
            raise KeyError(f"SketchState: unknown entity '{eid}'")
        return self.q[sl]

    # ------------------------------------------------------------------
    # Point / subnode accessors (ref strings like "0", "0.start", etc.)
    # ------------------------------------------------------------------

    def get_point(self, ref: str) -> torch.Tensor:
        """
        Return a (2,) tensor [x, y] for the point referenced by `ref`.

        Supported ref formats:
          - "eid.start"   / "eid.end"    for Line / Arc
          - "eid.center"                 for Circle / Arc
          - "eid"                        for Point entity
          - "eid"  (Line/Circle/Arc)     returns edge midpoint (for edge-level refs)
        """
        if '.' in ref:
            eid, sub = ref.split('.', 1)
            entity = self.sketch.entities.get(eid)
            if entity is None:
                raise KeyError(f"SketchState.get_point: entity '{eid}' not found")
            return self._subnode_point(entity, sub)
        else:
            eid = ref
            entity = self.sketch.entities.get(eid)
            if entity is None:
                # Could be external axis ref like '-1' – return zero tensor
                return torch.zeros(2, dtype=self.dtype)
            if isinstance(entity, datalib.Point):
                p = self._slice(eid)
                return p  # [x, y]
            elif isinstance(entity, datalib.Line):
                # Edge-level ref: return midpoint (pntX, pntY)
                p = self._slice(eid)
                return p[0:2]  # [pntX, pntY]
            elif isinstance(entity, (datalib.Circle, datalib.Arc)):
                p = self._slice(eid)
                return p[0:2]  # [xCenter, yCenter]
            else:
                raise ValueError(f"Unsupported entity type: {type(entity)}")

    def _subnode_point(self, entity: datalib.Entity, sub: str) -> torch.Tensor:
        eid = entity.entityId
        p = self._slice(eid)

        if isinstance(entity, datalib.Line):
            # p = [pntX, pntY, dirX, dirY, startParam, endParam]
            pnt = p[0:2]
            d = p[2:4]
            if sub == 'start':
                return pnt + p[4] * d
            elif sub == 'end':
                return pnt + p[5] * d
            else:
                raise ValueError(f"Invalid Line subnode: {sub}")

        elif isinstance(entity, datalib.Circle):
            # p = [xCenter, yCenter, radius]
            if sub == 'center':
                return p[0:2]
            else:
                raise ValueError(f"Invalid Circle subnode: {sub}")

        elif isinstance(entity, datalib.Arc):
            # p = [xCenter, yCenter, radius, startParam, endParam]
            cx, cy = p[0], p[1]
            r = p[2]
            if sub == 'center':
                return p[0:2]
            elif sub == 'start':
                # Use the stored xDir/yDir from the entity to rotate param angles
                # θ_base = atan2(yDir, xDir); θ_start = θ_base + startParam
                # (same convention as Arc._point_at_angle_offset)
                theta_base = math.atan2(entity.yDir, entity.xDir)
                if entity.clockwise:
                    theta = theta_base - p[3]
                else:
                    theta = theta_base + p[3]
                return torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)])
            elif sub == 'end':
                theta_base = math.atan2(entity.yDir, entity.xDir)
                if entity.clockwise:
                    theta = theta_base - p[4]
                else:
                    theta = theta_base + p[4]
                return torch.stack([cx + r * torch.cos(theta), cy + r * torch.sin(theta)])
            else:
                raise ValueError(f"Invalid Arc subnode: {sub}")

        elif isinstance(entity, datalib.Point):
            # Point has no subnodes
            raise ValueError(f"Point entity has no subnode: {sub}")

        else:
            raise ValueError(f"Unsupported entity type: {type(entity)}")

    # ------------------------------------------------------------------
    # Convenience accessors for constraint residual helpers
    # ------------------------------------------------------------------

    def get_line_params(self, eid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pnt[2], dir[2], startParam[scalar], endParam[scalar])."""
        p = self._slice(eid)
        return p[0:2], p[2:4], p[4], p[5]

    def get_arc_params(self, eid: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (center[2], radius[scalar], startParam[scalar], endParam[scalar])."""
        p = self._slice(eid)
        return p[0:2], p[2], p[3], p[4]

    def get_circle_params(self, eid: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (center[2], radius[scalar])."""
        p = self._slice(eid)
        return p[0:2], p[2]

    def get_radius(self, eid: str) -> torch.Tensor:
        entity = self.sketch.entities[eid]
        if isinstance(entity, (datalib.Circle, datalib.Arc)):
            return self._slice(eid)[2]
        raise ValueError(f"Entity '{eid}' has no radius")

    def num_dof(self) -> int:
        return len(self.q)


# ---------------------------------------------------------------------------
# Helper: entity -> flat parameter list
# ---------------------------------------------------------------------------

def _entity_to_params(entity: datalib.Entity) -> list[float] | None:
    if isinstance(entity, datalib.Line):
        return [entity.pntX, entity.pntY, entity.dirX, entity.dirY,
                entity.startParam, entity.endParam]
    elif isinstance(entity, datalib.Circle):
        return [entity.xCenter, entity.yCenter, entity.radius]
    elif isinstance(entity, datalib.Arc):
        return [entity.xCenter, entity.yCenter, entity.radius,
                entity.startParam, entity.endParam]
    elif isinstance(entity, datalib.Point):
        return [entity.x, entity.y]
    else:
        return None


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian(
    sketch: datalib.Sketch,
    constraint_ids: list[str],
    context,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, SketchState]:
    """
    Compute the constraint Jacobian J = dr/dq at the current geometry q₀.

    Parameters
    ----------
    sketch : datalib.Sketch  (entities must have real coordinate values)
    constraint_ids : list of constraint IDs to include
    context : SketchContext  (provides get_constraint_handler)
    dtype : torch floating type

    Returns
    -------
    J : torch.Tensor, shape (num_residuals, num_dof)
        Jacobian of the stacked residuals w.r.t. q at q₀.
    state : SketchState
        The state used for computation (holds q₀ and slices).
    """
    state = SketchState(sketch, dtype=dtype)
    n_dof = state.num_dof()

    if n_dof == 0:
        return torch.zeros(0, 0, dtype=dtype), state

    rows: list[torch.Tensor] = []

    for cid in constraint_ids:
        constraint = sketch.constraints.get(cid)
        if constraint is None:
            continue
        handler_name = constraint.constraint_type.name
        handler: ConstraintHandler | None = context.get_constraint_handler(handler_name)
        if handler is None:
            continue

        # Compute Jacobian of this constraint's residual w.r.t. q
        def _residual_fn(q: torch.Tensor) -> torch.Tensor:
            # Temporarily substitute q into state and evaluate residual
            s = _StateWithQ(state, q)
            try:
                return handler.residual(constraint, s)
            except Exception:
                return torch.zeros(1, dtype=q.dtype)

        try:
            jac = torch.func.jacfwd(_residual_fn)(state.q.detach())
            # jac: (k, n_dof) where k = residual size
            if jac.ndim == 1:
                jac = jac.unsqueeze(0)
            rows.append(jac)
        except Exception:
            continue

    if not rows:
        return torch.zeros(0, n_dof, dtype=dtype), state

    J = torch.cat(rows, dim=0)
    return J, state


# ---------------------------------------------------------------------------
# Helper class: SketchState with a substituted q vector (for jacfwd)
# ---------------------------------------------------------------------------

class _StateWithQ(SketchState):
    """A SketchState that uses an externally supplied q tensor (for autodiff)."""

    def __init__(self, base: SketchState, q: torch.Tensor):
        # Don't call super().__init__ – copy structure only
        self.sketch = base.sketch
        self.dtype = base.dtype
        self.slices = base.slices
        self.entity_ids = base.entity_ids
        self.q = q  # <- the differentiable tensor from jacfwd


# ---------------------------------------------------------------------------
# Null space computation
# ---------------------------------------------------------------------------

def compute_null_space(
    J: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the null space of J via SVD.

    Parameters
    ----------
    J : (m, n)
    eps : singular value threshold for null space

    Returns
    -------
    N : (n, k)  orthonormal basis of ker(J), k = dim(ker J)
    """
    if J.shape[0] == 0:
        # No constraints -> entire space is null space
        n = J.shape[1]
        return torch.eye(n, dtype=J.dtype)

    # J = U S Vh  -> null space = rows of Vh where S < eps
    _, S, Vh = torch.linalg.svd(J, full_matrices=True)
    null_mask = torch.cat([S < eps, torch.ones(Vh.shape[0] - len(S), dtype=torch.bool)])
    N = Vh[null_mask].T  # (n, k)
    return N


def compute_row_space(
    J: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the row space (range) of J via SVD.

    The row space is the orthogonal complement of the null space.
    It represents the set of directions that the constraints actively restrict.

    Parameters
    ----------
    J : (m, n)
    eps : singular value threshold for rank determination

    Returns
    -------
    R : (n, r)  orthonormal basis of row(J), r = rank(J)
        Returns (n, 0) empty tensor if J has no rows or rank 0.
    """
    if J.shape[0] == 0:
        # No constraints -> row space is trivial (zero-dimensional)
        return torch.zeros(J.shape[1], 0, dtype=J.dtype)

    _, S, Vh = torch.linalg.svd(J, full_matrices=True)
    row_mask = torch.cat([S >= eps, torch.zeros(Vh.shape[0] - len(S), dtype=torch.bool)])
    R = Vh[row_mask].T  # (n, r)
    return R

