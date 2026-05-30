"""Rule-based geometric constraint detector ("規則付与器").

Model-free: given a ``datalib.Sketch`` carrying *real* primitive geometry, this
module enumerates candidate relational constraints and keeps the ones whose
handler residual is ~0 at the sketch geometry.  It reuses each
``ConstraintHandler.residual`` (the exact same geometric definition the solver
and the d_H_angle metric use), so a detected predicate matches the rest of the
system by construction.

Why a *geometric predicate* detector rather than an RL enumerator: relations
such as ``Parallel``/``Perpendicular`` are DoF-substitutable by
``Horizontal``+``Horizontal`` / ``Horizontal``+``Vertical``, so a DoF-driven
generator never surfaces them.  A predicate test ("are these two lines actually
parallel?") is DoF-blind and adds the relation regardless of redundancy — which
is precisely the property needed to populate the pseudo-GT with the
design-intent relations.  See ``notebooks/rloo_response_metric_reward_research_plan.md`` §14.6.

Detection is grouped into three families so a single interpretable tolerance
governs each:

- ``orientation`` (Horizontal / Vertical / Parallel / Perpendicular): the
  residual is ``|d1||d2|·sin θ`` (or ``·cos θ`` for Perpendicular); dividing by
  the product of the involved line lengths yields ``|sin Δθ|``, thresholded by
  an angular tolerance.
- ``position`` (Coincident point-point / point-on-object, Concentric, Midpoint,
  Tangent, Mirror): the residual is mostly length-like; dividing by the sketch
  bounding-box diagonal yields a dimensionless fraction.  ``Tangent.point`` also
  has a dot/cross component with length^2 units, normalized by the involved line
  length / radius product.
- ``size`` (Equal length / radius): the residual is a magnitude difference;
  dividing by the larger magnitude yields a relative fraction.

``Mirror`` is limited to an existing line entity used as the symmetry axis; this
does not search for latent axes not represented in the sketch.  Line-line
``Coincident`` (Collinear) is intentionally excluded:
its residual mixes a cross-product and an offset on different scales, and it
overlaps ``Parallel``; it is rare (~0.3% of the relational data) and ambiguous.

The detector returns *all* candidates with their normalized margins (no
thresholding) so the fiverr recall/precision harness can sweep tolerances; use
``filter_candidates`` / ``augment_sketch`` to apply a tolerance and write the
accepted constraints back onto a sketch.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

import sketchgraphs.data as datalib

from src.metrics.constraint_subtypes import classify_constraint_label
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_solver import SketchState

__all__ = [
    "Candidate",
    "DEFAULT_TOLERANCES",
    "DETECTABLE_TYPES",
    "detect_candidates",
    "filter_candidates",
    "augment_sketch",
]


# Families and their default tolerances (the units differ per family — see the
# module docstring).  These are *initial* values; the fiverr recall/precision
# harness calibrates them per type.
DEFAULT_TOLERANCES: dict[str, float] = {
    "orientation": math.sin(math.radians(2.0)),  # within ~2 deg of axis/parallel/perp
    "position": 0.01,                            # within 1% of the bbox diagonal
    "size": 0.02,                                # within 2% relative magnitude
}

# Relational types this detector can produce.
DETECTABLE_TYPES: tuple[str, ...] = (
    "Coincident",   # point-point and point-on-object (PointOnObject) sub-cases
    "Horizontal",
    "Vertical",
    "Parallel",
    "Perpendicular",
    "Equal",        # length (line-line) and radius (circle/arc) sub-cases
    "Concentric",
    "Midpoint",
    "Tangent",      # edge-edge and point-specified tangency
    "Mirror",       # point-point about an existing line axis
)

_EPS = 1e-9
# PointOnObject / Tangent are defined (by the handlers' residuals) against the
# *infinite* line, so a contact on the line's extension is valid — we do NOT
# restrict it to the segment.  The one exception is a point sitting on a line's
# own endpoint: that is a vertex coincidence, emitted as point-point, so we skip
# it as a point-on-object to avoid double-counting corners.
_ENDPOINT_GUARD_FRAC = 1e-2
_MIDPOINT3_PREFILTER_FRAC = 5e-2


@dataclass(frozen=True)
class Candidate:
    """One detected candidate constraint.

    Attributes
    ----------
    type_name : str
        Constraint head-type name (e.g. ``"Parallel"``, ``"Coincident"``).
    refs : tuple[str, ...]
        Reference strings in the sketch's own id convention (e.g.
        ``("0.start", "1.end")``, ``("3", "4")``).  Symmetric relations are
        stored with sorted refs; ``Midpoint`` keeps ``(point, line)`` order.
    family : str
        ``"orientation"`` | ``"position"`` | ``"size"`` — selects the tolerance.
    margin : float
        Normalized ``|residual|`` (smaller = more clearly satisfied).  Directly
        comparable to ``DEFAULT_TOLERANCES[family]``.
    subtype : str
        Effective sub-type label from
        :func:`src.metrics.constraint_subtypes.classify_constraint_label`
        (e.g. ``"Coincident.point_on_object"``, ``"Equal.radius"``,
        ``"Tangent.edge"``), else the plain type name.
    """

    type_name: str
    refs: tuple[str, ...]
    family: str
    margin: float
    subtype: str


# --------------------------------------------------------------------------- #
#  Geometry helpers (all read the same SketchState the residuals consume)
# --------------------------------------------------------------------------- #

def _np_point(state: SketchState, ref: str) -> np.ndarray:
    return state.get_point(ref).detach().cpu().numpy().astype(np.float64)


def _line_length(state: SketchState, eid: str) -> float:
    s = _np_point(state, f"{eid}.start")
    e = _np_point(state, f"{eid}.end")
    return float(np.linalg.norm(e - s))


def _radius(state: SketchState, eid: str) -> float:
    return float(state.get_radius(eid).detach().cpu())


def _bbox_diagonal(state: SketchState, sketch: datalib.Sketch) -> float:
    """Length scale for position-family normalization: the bbox diagonal.

    Bounds lines by their endpoints and circles/arcs by ``center ± radius`` on
    both axes, matching the geometric extent used elsewhere.
    """
    pts: list[np.ndarray] = []
    for eid, ent in sketch.entities.items():
        if isinstance(ent, datalib.Point):
            pts.append(_np_point(state, eid))
        elif isinstance(ent, datalib.Line):
            pts.append(_np_point(state, f"{eid}.start"))
            pts.append(_np_point(state, f"{eid}.end"))
        elif isinstance(ent, (datalib.Circle, datalib.Arc)):
            c = _np_point(state, f"{eid}.center")
            r = _radius(state, eid)
            pts.extend([c + [r, 0.0], c - [r, 0.0], c + [0.0, r], c - [0.0, r]])
    if not pts:
        return 1.0
    arr = np.stack(pts)
    diag = float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
    return max(diag, _EPS)


def _residual_norm(handler, constraint, state: SketchState) -> float:
    """L2 norm of a constraint's residual (the raw, un-normalized geometric gap)."""
    r = handler.residual(constraint, state)
    return float(torch.linalg.norm(r.detach()))


# --------------------------------------------------------------------------- #
#  Reference enumeration
# --------------------------------------------------------------------------- #

def _point_refs(sketch: datalib.Sketch) -> list[str]:
    """Point-like refs eligible for coincidence/midpoint endpoints.

    Point entities (bare id) + Line/Arc endpoints (``eid.start`` / ``eid.end``).
    Circle/Arc centers are deliberately excluded here — center coincidence is
    ``Concentric``, handled separately.
    """
    refs: list[str] = []
    for eid, ent in sketch.entities.items():
        if isinstance(ent, datalib.Point):
            refs.append(eid)
        elif isinstance(ent, (datalib.Line, datalib.Arc)):
            refs.append(f"{eid}.start")
            refs.append(f"{eid}.end")
    return refs


def _endpoint_refs(sketch: datalib.Sketch) -> list[str]:
    refs: list[str] = []
    for eid, ent in sketch.entities.items():
        if isinstance(ent, (datalib.Line, datalib.Arc)):
            refs.append(f"{eid}.start")
            refs.append(f"{eid}.end")
    return refs


def _mirror_point_refs(sketch: datalib.Sketch) -> list[str]:
    """Point-like refs eligible for Mirror, including curve centres."""
    refs = _point_refs(sketch)
    for eid, ent in sketch.entities.items():
        if isinstance(ent, (datalib.Circle, datalib.Arc)):
            refs.append(f"{eid}.center")
    return refs


def _midpoint_point_refs(sketch: datalib.Sketch) -> list[str]:
    """Point-like refs eligible to be the explicit midpoint reference."""
    return _mirror_point_refs(sketch)


def _entities_of(sketch: datalib.Sketch, types: tuple[type, ...]) -> list[str]:
    return [eid for eid, ent in sketch.entities.items() if isinstance(ent, types)]


def _ref_base(ref: str) -> str:
    return ref.split(".", 1)[0]


def _canonical_refs(type_name: str, refs: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical ref ordering for dedup/matching.

    ``Midpoint`` is asymmetric: keep ``(point, line)`` order for 2-ref form,
    and sort only the two endpoint refs for 3-ref form.
    ``Mirror`` keeps the axis as the last ref and only sorts the reflected pair.
    All other detected relations are symmetric — sort.
    """
    if type_name == "Midpoint" and len(refs) == 3:
        return tuple(sorted(refs[:2])) + (refs[2],)
    if type_name == "Midpoint":
        return refs
    if type_name == "Mirror" and len(refs) == 3:
        return tuple(sorted(refs[:2])) + (refs[2],)
    return tuple(sorted(refs))


# --------------------------------------------------------------------------- #
#  Detection
# --------------------------------------------------------------------------- #

def detect_candidates(
    sketch: datalib.Sketch,
    context: SketchContext | None = None,
    *,
    types: set[str] | None = None,
) -> list[Candidate]:
    """Enumerate candidate relational constraints with normalized margins.

    Returns *every* geometrically-plausible candidate (no tolerance applied) so
    callers can sweep thresholds.  Use :func:`filter_candidates` to apply a
    tolerance.

    Parameters
    ----------
    sketch : datalib.Sketch
        Must carry real primitive geometry (the augmenter runs on the input
        sketch, not on quantized tokens).
    context : SketchContext, optional
        Provides the constraint handlers.  A default ``SketchContext()`` is
        built if omitted.
    types : set[str], optional
        Restrict detection to these head-type names (default: all of
        :data:`DETECTABLE_TYPES`).
    """
    if context is None:
        context = SketchContext()
    wanted = set(DETECTABLE_TYPES) if types is None else (set(types) & set(DETECTABLE_TYPES))
    if not wanted:
        return []

    state = SketchState(sketch)
    bbox = _bbox_diagonal(state, sketch)

    lines = _entities_of(sketch, (datalib.Line,))
    curves = _entities_of(sketch, (datalib.Circle, datalib.Arc))
    point_refs = _point_refs(sketch)
    endpoint_refs = _endpoint_refs(sketch)

    line_len = {eid: _line_length(state, eid) for eid in lines}
    curve_r = {eid: _radius(state, eid) for eid in curves}

    out: list[Candidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def _emit(type_name: str, refs: tuple[str, ...], family: str, raw: float, scale: float) -> None:
        if scale <= _EPS:
            return
        key = (type_name, _canonical_refs(type_name, refs))
        if key in seen:
            return
        seen.add(key)
        handler = context.get_constraint_handler(type_name)
        try:
            constraint = handler.create_constraint(type_name, refs)
            subtype = classify_constraint_label(constraint, sketch)
        except Exception:
            subtype = type_name
        out.append(
            Candidate(
                type_name=type_name,
                refs=_canonical_refs(type_name, refs),
                family=family,
                margin=raw / scale,
                subtype=subtype,
            )
        )

    def _raw(type_name: str, refs: tuple[str, ...]) -> float | None:
        handler = context.get_constraint_handler(type_name)
        try:
            constraint = handler.create_constraint(type_name, refs)
            return _residual_norm(handler, constraint, state)
        except Exception:
            return None

    def _residual_vec(type_name: str, refs: tuple[str, ...]) -> np.ndarray | None:
        handler = context.get_constraint_handler(type_name)
        try:
            constraint = handler.create_constraint(type_name, refs)
            return handler.residual(constraint, state).detach().cpu().numpy().reshape(-1).astype(np.float64)
        except Exception:
            return None

    def _tangent_scale(refs: tuple[str, ...]) -> float:
        scales: list[float] = []
        for ref in refs:
            eid = _ref_base(ref)
            ent = sketch.entities.get(eid)
            if isinstance(ent, datalib.Line):
                scales.append(line_len.get(eid, 0.0))
            elif isinstance(ent, (datalib.Circle, datalib.Arc)):
                scales.append(curve_r.get(eid, 0.0))
        if len(scales) < 2:
            return 0.0
        return max(scales[0] * scales[1], _EPS)

    def _tangent_point_margin(refs: tuple[str, ...]) -> float | None:
        if not any("." in r for r in refs):
            return None
        ents = [sketch.entities.get(_ref_base(r)) for r in refs]
        if not any(isinstance(ent, (datalib.Circle, datalib.Arc)) for ent in ents):
            return None
        residual = _residual_vec("Tangent", refs)
        if residual is None or residual.size == 0:
            return None
        if residual.size == 1:
            return abs(float(residual[0])) / bbox

        # Tangent.point residuals are [contact distance(s), tangent area].
        # Distance components are length-like; the final dot/cross component has
        # length^2 units and is normalized by the involved line length/radius.
        dist_margin = float(np.linalg.norm(residual[:-1])) / bbox
        tangent_margin = abs(float(residual[-1])) / _tangent_scale(refs)
        return float(math.hypot(dist_margin, tangent_margin))

    def _nearby_point_refs(query: np.ndarray, cell_size: float, buckets: dict[tuple[int, int], list[str]]) -> list[str]:
        key = tuple(np.floor(query / cell_size).astype(int).tolist())
        out_refs: list[str] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out_refs.extend(buckets.get((key[0] + dx, key[1] + dy), ()))
        return out_refs

    # -- orientation: Horizontal / Vertical (single line) -------------------- #
    for type_name in ("Horizontal", "Vertical"):
        if type_name not in wanted:
            continue
        for eid in lines:
            raw = _raw(type_name, (eid,))
            if raw is not None:
                _emit(type_name, (eid,), "orientation", raw, line_len[eid])

    # -- orientation: Parallel / Perpendicular (line pairs) ------------------ #
    for type_name in ("Parallel", "Perpendicular"):
        if type_name not in wanted:
            continue
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                raw = _raw(type_name, (a, b))
                if raw is not None:
                    _emit(type_name, (a, b), "orientation", raw, line_len[a] * line_len[b])

    # -- size: Equal length (line pairs) ------------------------------------- #
    if "Equal" in wanted:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                raw = _raw("Equal", (a, b))
                if raw is not None:
                    _emit("Equal", (a, b), "size", raw, max(line_len[a], line_len[b]))
        # -- size: Equal radius (circle/arc pairs) --------------------------- #
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                a, b = curves[i], curves[j]
                raw = _raw("Equal", (a, b))
                if raw is not None:
                    _emit("Equal", (a, b), "size", raw, max(curve_r[a], curve_r[b]))

    # -- position: Concentric (circle/arc pairs) ----------------------------- #
    if "Concentric" in wanted:
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                a, b = curves[i], curves[j]
                raw = _raw("Concentric", (a, b))
                if raw is not None:
                    _emit("Concentric", (a, b), "position", raw, bbox)

    # -- position: Coincident point-point ------------------------------------ #
    if "Coincident" in wanted:
        for i in range(len(point_refs)):
            for j in range(i + 1, len(point_refs)):
                ra, rb = point_refs[i], point_refs[j]
                if _ref_base(ra) == _ref_base(rb):
                    continue  # two subnodes of the same entity never coincide
                raw = _raw("Coincident", (ra, rb))
                if raw is not None:
                    _emit("Coincident", (ra, rb), "position", raw, bbox)

        # -- position: Coincident point-on-object (PointOnObject) ------------ #
        # A point lies ON an edge (the line is infinite, so contacts on its
        # extension are valid — see the _ENDPOINT_GUARD note above; only a point
        # sitting on the edge's own endpoint is skipped, as that is a corner
        # already covered by the point-point case).
        for pr in point_refs:
            pr_xy = _np_point(state, pr)
            for edge in lines:
                if _ref_base(pr) == edge:
                    continue
                s_pt = _np_point(state, f"{edge}.start")
                e_pt = _np_point(state, f"{edge}.end")
                # A point on the edge's own endpoint is a vertex coincidence
                # (emitted as point-point), not point-on-object — skip it so corners
                # aren't double-counted.  Contacts on the line's extension are kept.
                if min(np.linalg.norm(pr_xy - s_pt), np.linalg.norm(pr_xy - e_pt)) < _ENDPOINT_GUARD_FRAC * bbox:
                    continue
                raw = _raw("Coincident", (pr, edge))
                if raw is not None:
                    _emit("Coincident", (pr, edge), "position", raw, bbox)
            for edge in curves:
                if _ref_base(pr) == edge:
                    continue
                raw = _raw("Coincident", (pr, edge))
                if raw is not None:
                    _emit("Coincident", (pr, edge), "position", raw, bbox)

    # -- position: Midpoint (point is midpoint of a line) -------------------- #
    if "Midpoint" in wanted:
        midpoint_refs = _midpoint_point_refs(sketch)
        for pr in midpoint_refs:
            for line in lines:
                if _ref_base(pr) == line:
                    continue
                raw = _raw("Midpoint", (pr, line))
                if raw is not None:
                    _emit("Midpoint", (pr, line), "position", raw, bbox)

        # 3-ref form: endpoints of one non-line edge plus an explicit midpoint
        # ref.  We intentionally do not enumerate every pair of arbitrary
        # points: grid-like sketches create thousands of geometrically true but
        # intent-weak midpoints.  Arc endpoint midpoint is the structural 3-ref
        # form emitted by the FreeCAD converter, and it is only O(A * P).
        arc_refs = _entities_of(sketch, (datalib.Arc,))
        ref_xy = {ref: _np_point(state, ref) for ref in midpoint_refs}
        cell_size = max(_MIDPOINT3_PREFILTER_FRAC * bbox, _EPS)
        buckets: dict[tuple[int, int], list[str]] = {}
        for ref, xy in ref_xy.items():
            key = tuple(np.floor(xy / cell_size).astype(int).tolist())
            buckets.setdefault(key, []).append(ref)
        max_raw = _MIDPOINT3_PREFILTER_FRAC * bbox

        for arc in arc_refs:
            a, b = f"{arc}.start", f"{arc}.end"
            target = (_np_point(state, a) + _np_point(state, b)) * 0.5
            for mid in _nearby_point_refs(target, cell_size, buckets):
                if mid in (a, b):
                    continue
                raw = _raw("Midpoint", (a, b, mid))
                if raw is not None and raw <= max_raw:
                    _emit("Midpoint", (a, b, mid), "position", raw, bbox)

    # -- position: Tangent (edge-edge) --------------------------------------- #
    if "Tangent" in wanted:
        # line-curve (the line is treated as infinite for tangency, matching the
        # handler residual — the contact may lie on the line's extension)
        for line in lines:
            for curve in curves:
                raw = _raw("Tangent", (line, curve))
                if raw is not None:
                    _emit("Tangent", (line, curve), "position", raw, bbox)
        # curve-curve
        for i in range(len(curves)):
            for j in range(i + 1, len(curves)):
                a, b = curves[i], curves[j]
                raw = _raw("Tangent", (a, b))
                if raw is not None:
                    _emit("Tangent", (a, b), "position", raw, bbox)

        # point-specified tangency: at least one endpoint ref identifies the
        # contact point.  This covers FreeCAD's Tangent(e1, p1, e2[, p2]) forms.
        edges = lines + curves
        for pr in endpoint_refs:
            base = _ref_base(pr)
            for edge in edges:
                if edge == base:
                    continue
                margin = _tangent_point_margin((pr, edge))
                if margin is not None:
                    _emit("Tangent", (pr, edge), "position", margin, 1.0)
        for i in range(len(endpoint_refs)):
            for j in range(i + 1, len(endpoint_refs)):
                a, b = endpoint_refs[i], endpoint_refs[j]
                if _ref_base(a) == _ref_base(b):
                    continue
                margin = _tangent_point_margin((a, b))
                if margin is not None:
                    _emit("Tangent", (a, b), "position", margin, 1.0)

    # -- position: Mirror (point pair symmetric about an existing line axis) ---- #
    if "Mirror" in wanted:
        mirror_points = _mirror_point_refs(sketch)
        for i in range(len(mirror_points)):
            for j in range(i + 1, len(mirror_points)):
                a, b = mirror_points[i], mirror_points[j]
                for axis in lines:
                    if axis in (_ref_base(a), _ref_base(b)):
                        continue
                    raw = _raw("Mirror", (a, b, axis))
                    if raw is not None:
                        _emit("Mirror", (a, b, axis), "position", raw, bbox)

    return out


def filter_candidates(
    candidates: list[Candidate],
    tolerances: dict[str, float] | None = None,
    *,
    per_subtype: dict[str, float] | None = None,
) -> list[Candidate]:
    """Keep candidates whose margin is within the family (or per-subtype) tolerance.

    ``per_subtype`` overrides the family tolerance for specific sub-type labels
    (e.g. ``{"Equal.radius": 0.03}``); it takes precedence over ``tolerances``.
    """
    tol = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    per_subtype = per_subtype or {}
    kept: list[Candidate] = []
    for c in candidates:
        limit = per_subtype.get(c.subtype, tol.get(c.family, 0.0))
        if c.margin <= limit:
            kept.append(c)
    return kept


def augment_sketch(
    sketch: datalib.Sketch,
    context: SketchContext | None = None,
    *,
    tolerances: dict[str, float] | None = None,
    per_subtype: dict[str, float] | None = None,
    types: set[str] | None = None,
    id_prefix: str = "rule",
) -> datalib.Sketch:
    """Return a deep copy of *sketch* with detected constraints added.

    Existing constraints are preserved; a detected candidate that duplicates an
    existing ``(type, canonical-refs)`` is skipped.  New constraints get unique
    ids ``f"{id_prefix}_{type}_{i}"``.
    """
    if context is None:
        context = SketchContext()

    candidates = filter_candidates(
        detect_candidates(sketch, context, types=types),
        tolerances,
        per_subtype=per_subtype,
    )

    augmented = copy.deepcopy(sketch)

    existing: set[tuple[str, tuple[str, ...]]] = set()
    for constr in augmented.constraints.values():
        type_name = getattr(constr.type, "name", None)
        if type_name is None:
            continue
        refs = tuple(
            p.value
            for p in constr.parameters
            if isinstance(p, datalib.LocalReferenceParameter)
        )
        existing.add((type_name, _canonical_refs(type_name, refs)))

    for i, c in enumerate(candidates):
        key = (c.type_name, c.refs)
        if key in existing:
            continue
        existing.add(key)
        handler = context.get_constraint_handler(c.type_name)
        cid = f"{id_prefix}_{c.type_name}_{i}"
        constraint = handler.create_constraint(c.type_name, c.refs)._replace(identifier=cid)
        augmented.constraints[cid] = constraint

    return augmented
