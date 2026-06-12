"""cleanup_sketch.py - Post-processing cleanup for FreeCAD sketch script files.

This script applies a series of geometric cleanup operations to FreeCAD sketch scripts
produced by convert_fusion360_to_freecad.py:

  1. Duplicate primitive removal  – exact geometric duplicates are dropped; when a
     normal line and a construction line overlap the construction one is removed.
  2. Micro-gap merging            – endpoint pairs closer than EPS are snapped to the
     same coordinate and linked with a Coincident constraint.
  3. Degenerate primitive removal – zero-length lines, zero-radius circles/arcs.
  4. PointOnObject inference      – when a point (or endpoint subnode) lies strictly
     in the INTERIOR of a curve (not at its endpoints), a PointOnObject constraint
     is added. Endpoints already joined by Coincident are skipped.
  5. Redundant constraint rewriting – delegates to
     `src.sketch_handlers.sketch_utils.cleanse_sketch_constraints` operating on the
     SketchGraphs datalib.Sketch representation.

Design note on the output pipeline
-----------------------------------
`SketchFreecadConverter.to_freecad_commands` goes through a full FreeCAD round-trip
(datalib → FreeCAD object → dump_freecad_commands).  FreeCAD's constraint solver
may silently drop constraints it considers redundant (e.g. a Coincident that is
implied by a simultaneously active Perpendicular with point sub-references).  To
avoid this, we write output a different way:

  * Geometry lines are preserved verbatim from the original script (no round-trip).
  * Only the constraint lines are regenerated from the cleaned datalib.Sketch via
    `SketchFreecadConverter.to_freecad_commands`, but the constraints are spliced
    into the original script skeleton rather than using the header from
    `dump_freecad_commands`.
"""

import argparse
import glob
import itertools
import math
import os
import random
import string
import re
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import (
    Arc, Circle, Line, Point,
    ConstraintType, Constraint,
    LocalReferenceParameter,
)
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import cleanse_sketch_constraints, get_entity_point

ensure_freecad_imported()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPS = 1e-4   # tolerance for micro-gap merging and degenerate detection
EPS_DUP = 1e-9  # tolerance for exact duplicate detection


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------

class UnionFind(dict):
    def find(self, x):
        if x not in self:
            self[x] = x
        root = x
        while self[root] != root:
            root = self[root]
        while self[x] != root:
            self[x], x = root, self[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self[rx] = ry

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _new_constraint_id(sketch: datalib.Sketch) -> str:
    """Generate a unique constraint identifier that does not clash with existing ones.
    Uses the global random state which can be seeded for reproducibility.
    """
    existing = set(sketch.constraints.keys())
    while True:
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        cid = f"cleanup_{suffix}"
        if cid not in existing:
            return cid


def _make_coincident(cid: str, ref_a: str, ref_b: str) -> Constraint:
    """Build a Coincident constraint between two references (e.g. '3.start', '5.end')."""
    return Constraint(
        identifier=cid,
        constraint_type=ConstraintType.Coincident,
        parameters=[
            LocalReferenceParameter(parameterId='localFirst', value=ref_a),
            LocalReferenceParameter(parameterId='localSecond', value=ref_b),
        ],
    )


def _make_point_on_object(cid: str, point_ref: str, curve_ref: str) -> Constraint:
    """Build a PointOnObject (Coincident with edge body) constraint.

    In SketchGraphs format PointOnObject is represented as Coincident where one
    reference has no subnode (pointing to the curve body itself).
    """
    return Constraint(
        identifier=cid,
        constraint_type=ConstraintType.Coincident,
        parameters=[
            LocalReferenceParameter(parameterId='localFirst', value=point_ref),
            LocalReferenceParameter(parameterId='localSecond', value=curve_ref),
        ],
    )


def _all_point_refs(sketch: datalib.Sketch) -> list[tuple[str, Optional[str]]]:
    """Return all (entity_id, subnode) pairs that represent geometric points."""
    result = []
    for eid, ent in sketch.entities.items():
        if isinstance(ent, Point):
            result.append((eid, None))
        elif isinstance(ent, Line):
            result.append((eid, 'start'))
            result.append((eid, 'end'))
        elif isinstance(ent, (Circle, Arc)):
            result.append((eid, 'center'))
        if isinstance(ent, Arc):
            result.append((eid, 'start'))
            result.append((eid, 'end'))
    return result


def _ref_str(eid: str, subnode: Optional[str]) -> str:
    return f"{eid}.{subnode}" if subnode else eid


def _parse_ref(ref: str) -> tuple[str, Optional[str]]:
    parts = ref.split('.')
    if len(parts) == 2 and parts[1] in ('start', 'end', 'center'):
        return parts[0], parts[1]
    return parts[0], None


def _get_coord(sketch: datalib.Sketch, eid: str, subnode: Optional[str]) -> Optional[np.ndarray]:
    ent = sketch.entities.get(eid)
    if ent is None:
        return None
    pt = get_entity_point(ent, subnode)
    if pt is None:
        return None
    return np.array(pt, dtype=float)


def _remove_entity(sketch: datalib.Sketch, eid: str) -> None:
    """Remove an entity and all constraints that reference it."""
    sketch.entities.pop(eid, None)
    to_del = [cid for cid, c in sketch.constraints.items()
              if any(isinstance(p, LocalReferenceParameter) and p.value.split('.')[0] == eid
                     for p in c.parameters)]
    for cid in to_del:
        del sketch.constraints[cid]


def _existing_coincidence_set(sketch: datalib.Sketch) -> set[frozenset]:
    """Collect all existing Coincident constraint pairs as frozensets of ref strings."""
    existing: set[frozenset] = set()
    for c in sketch.constraints.values():
        if c.constraint_type == ConstraintType.Coincident:
            refs = [p.value for p in c.parameters if isinstance(p, LocalReferenceParameter)]
            if len(refs) == 2:
                existing.add(frozenset(refs))
    return existing


def _existing_poo_set(sketch: datalib.Sketch) -> set[frozenset]:
    """Collect PointOnObject-style pairs from the sketch.

    In the datalib representation, FreeCAD's PointOnObject(pt, curve) is stored
    as a Coincident constraint where one ref is a point sub-ref ('entity.subnode')
    and the other is a bare entity ID (no dot), representing the curve body.
    """
    poo: set[frozenset] = set()
    for c in sketch.constraints.values():
        if c.constraint_type != ConstraintType.Coincident:
            continue
        refs = [p.value for p in c.parameters if isinstance(p, LocalReferenceParameter)]
        if len(refs) != 2:
            continue
        # PoO pattern: one ref has a '.' (point subnode), other is a bare entity ID
        has_dot = ['.' in r for r in refs]
        if sum(has_dot) == 1:  # exactly one dotted ref → PoO-style
            poo.add(frozenset(refs))
    return poo


def _build_coincident_clusters(
    existing: set[frozenset],
    entities: Optional[dict] = None,
) -> UnionFind:
    """Build a Union-Find parent map over Coincident constraint pairs.

    Covers two cases:
    - Point-Point (both refs contain '.'): always unioned.
    - Point-entity pairs where one ref has '.' and the other is a bare ID for a
      Point entity (no dot): also unioned, since FreeCAD stores
      PointOnObject(IndependentPoint, curve) as Coincident(barePointID, curveID),
      and Coincident(IndependentPoint, lineEndpoint) as Coincident(bareID, 'L.end').

    Bare curve entity IDs (Line/Circle/Arc bodies) are NOT unioned to avoid
    merging unrelated geometric clusters.
    """
    from sketchgraphs.data import Point as _Point

    def _is_point_entity(ref: str, entities: dict) -> bool:
        """True if ref is a bare entity ID referring to a Point primitive."""
        if entities is None or '.' in ref:
            return False
        ent = entities.get(ref)
        return isinstance(ent, _Point)

    uf = UnionFind()
    for pair in existing:
        refs = list(pair)
        if len(refs) != 2:
            continue
        r0, r1 = refs
        dot0, dot1 = '.' in r0, '.' in r1
        if dot0 and dot1:
            uf.union(r0, r1)
        elif dot0 and not dot1 and _is_point_entity(r1, entities):
            uf.union(r0, r1)
        elif dot1 and not dot0 and _is_point_entity(r0, entities):
            uf.union(r0, r1)
    return uf


def _is_linked_to_curve(pt_ref: str, curve_eid: str,
                         existing: set[frozenset],
                         coincident_uf: UnionFind) -> bool:
    target_root = coincident_uf.find(pt_ref)
    for pair in existing:
        if len(pair) != 2:
            continue
        refs = list(pair)
        for i, r in enumerate(refs):
            if coincident_uf.find(r) != target_root:
                continue
            other_eid = refs[1 - i].split('.')[0]
            if other_eid == curve_eid:
                return True
    return False


# ---------------------------------------------------------------------------
# Step 1 – Duplicate primitive removal
# ---------------------------------------------------------------------------

def _line_canonical(line: Line) -> tuple:
    sp = tuple(np.round(line.start_point, 9))
    ep = tuple(np.round(line.end_point, 9))
    return (min(sp, ep), max(sp, ep))


def _circle_canonical(circle: Circle) -> tuple:
    return (round(circle.xCenter, 9), round(circle.yCenter, 9), round(circle.radius, 9))


def _arc_canonical(arc: Arc) -> tuple:
    sp = tuple(np.round(arc.start_point, 9))
    ep = tuple(np.round(arc.end_point, 9))
    cp = (round(arc.xCenter, 9), round(arc.yCenter, 9))
    r = round(arc.radius, 9)
    # Include the arc midpoint so that complementary arcs (same endpoints,
    # center and radius but tracing the opposite half of the circle) are NOT
    # considered duplicates.  Without this, e.g. two semicircular arcs that
    # together form a full circle would be incorrectly deduplicated.
    mp = tuple(np.round(arc.mid_point, 9))
    return (min(sp, ep), max(sp, ep), cp, r, mp)


def _point_canonical(point: Point) -> tuple:
    return (round(point.x, 9), round(point.y, 9))


def remove_duplicate_primitives(sketch: datalib.Sketch) -> datalib.Sketch:
    """Remove geometrically identical primitives.

    Rules
    -----
    * Same Python type (Line/Circle/Arc/Point).
    * Direction reversal (start↔end swap) counts as duplicate for Lines/Arcs.
    * When a construction and normal geometry are identical, keep the normal one.
    """
    seen: dict[tuple, str] = {}
    to_remove: list[str] = []

    for eid, ent in list(sketch.entities.items()):
        if isinstance(ent, Line):
            key = ('Line', _line_canonical(ent))
        elif isinstance(ent, Circle):
            key = ('Circle', _circle_canonical(ent))
        elif isinstance(ent, Arc):
            key = ('Arc', _arc_canonical(ent))
        elif isinstance(ent, Point):
            key = ('Point', _point_canonical(ent))
        else:
            continue

        if key not in seen:
            seen[key] = eid
        else:
            prev_eid = seen[key]
            prev_ent = sketch.entities[prev_eid]
            if prev_ent.isConstruction and not ent.isConstruction:
                to_remove.append(prev_eid)
                seen[key] = eid
            else:
                to_remove.append(eid)

    for eid in to_remove:
        _remove_entity(sketch, eid)

    return sketch


def remove_redundant_points(sketch: datalib.Sketch, eps: float = EPS) -> datalib.Sketch:
    """Remove independent Point primitives that coincide with a curve endpoint.

    An independent ``Part.Point`` that sits exactly on the start or end of a Line
    or Arc carries no independent geometric information once the pipeline has added
    the corresponding Coincident constraints.  Keeping such points clutters the
    output and prevents FreeCAD from being fully constrained.

    A Point is removed when its coordinate is within *eps* of **any** endpoint of
    any non-Point entity (Line, Arc) in the sketch.
    """
    # Collect all curve-endpoint coordinates
    curve_endpoints: list[np.ndarray] = []
    for ent in sketch.entities.values():
        if isinstance(ent, Line):
            curve_endpoints.append(ent.start_point)
            curve_endpoints.append(ent.end_point)
        elif isinstance(ent, Arc):
            curve_endpoints.append(ent.start_point)
            curve_endpoints.append(ent.end_point)

    if not curve_endpoints:
        return sketch

    to_remove: list[str] = []
    for eid, ent in list(sketch.entities.items()):
        if not isinstance(ent, Point):
            continue
        pt = np.array([ent.x, ent.y])
        if any(np.linalg.norm(pt - ep) <= eps for ep in curve_endpoints):
            to_remove.append(eid)

    for eid in to_remove:
        _remove_entity(sketch, eid)

    return sketch


# ---------------------------------------------------------------------------
# Step 2 – Micro-gap merging (snap + Coincident)
# ---------------------------------------------------------------------------

def merge_micro_gaps(sketch: datalib.Sketch, eps: float = EPS) -> datalib.Sketch:
    """Snap endpoint pairs closer than *eps* to the same coordinate and add Coincident."""
    pts = _all_point_refs(sketch)

    coords: list[tuple[np.ndarray, str, Optional[str]]] = []
    for eid, sub in pts:
        coord = _get_coord(sketch, eid, sub)
        if coord is not None:
            coords.append((coord, eid, sub))

    if not coords:
        return sketch

    n = len(coords)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in itertools.combinations(range(n), 2):
        dist = np.linalg.norm(coords[i][0] - coords[j][0])
        if dist < eps:
            union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    existing_coincidences = _existing_coincidence_set(sketch)
    coin_uf = _build_coincident_clusters(existing_coincidences)

    for members in clusters.values():
        if len(members) < 2:
            continue

        centroid = np.mean([coords[i][0] for i in members], axis=0)
        for i in members:
            _, eid, sub = coords[i]
            ent = sketch.entities.get(eid)
            if ent is None:
                continue
            _set_entity_point(ent, sub, centroid)

        ref_strs = [_ref_str(coords[i][1], coords[i][2]) for i in members]
        for ra, rb in itertools.combinations(ref_strs, 2):
            pair = frozenset([ra, rb])
            if pair in existing_coincidences:
                continue
            if ra.split('.')[0] == rb.split('.')[0]:
                continue
            if coin_uf.connected(ra, rb):
                continue
            cid = _new_constraint_id(sketch)
            sketch.constraints[cid] = _make_coincident(cid, ra, rb)
            existing_coincidences.add(pair)
            coin_uf.union(ra, rb)

    return sketch


def _set_entity_point(ent, subnode: Optional[str], coord: np.ndarray) -> None:
    """Update the geometric point inside *ent* identified by *subnode* to *coord*."""
    x, y = float(coord[0]), float(coord[1])

    if isinstance(ent, Point):
        ent.x = x
        ent.y = y

    elif isinstance(ent, Line):
        if subnode == 'start':
            old_ep = ent.end_point.copy()
            _update_line_from_endpoints(ent, np.array([x, y]), old_ep)
        elif subnode == 'end':
            old_sp = ent.start_point.copy()
            _update_line_from_endpoints(ent, old_sp, np.array([x, y]))

    elif isinstance(ent, (Circle, Arc)):
        if subnode == 'center':
            ent.xCenter = x
            ent.yCenter = y
        elif isinstance(ent, Arc):
            if subnode == 'start':
                angle = math.atan2(y - ent.yCenter, x - ent.xCenter)
                base_angle = math.atan2(ent.yDir, ent.xDir)
                ent.startParam = angle - base_angle
                if ent.clockwise:
                    ent.startParam = -ent.startParam
            elif subnode == 'end':
                angle = math.atan2(y - ent.yCenter, x - ent.xCenter)
                base_angle = math.atan2(ent.yDir, ent.xDir)
                ent.endParam = angle - base_angle
                if ent.clockwise:
                    ent.endParam = -ent.endParam


def _update_line_from_endpoints(line: Line, sp: np.ndarray, ep: np.ndarray) -> None:
    vec = ep - sp
    length = np.linalg.norm(vec)
    if length > 0:
        line.dirX, line.dirY = float(vec[0] / length), float(vec[1] / length)
    line.pntX = float((sp[0] + ep[0]) / 2)
    line.pntY = float((sp[1] + ep[1]) / 2)
    line.startParam = float(-length / 2)
    line.endParam = float(length / 2)


# ---------------------------------------------------------------------------
# Step 3 – Degenerate primitive removal
# ---------------------------------------------------------------------------

def remove_degenerate_primitives(sketch: datalib.Sketch, eps: float = EPS) -> datalib.Sketch:
    """Remove zero-length lines, zero-radius circles/arcs, and near-zero-length arcs."""
    to_remove = []

    for eid, ent in sketch.entities.items():
        if isinstance(ent, Line):
            length = np.linalg.norm(ent.end_point - ent.start_point)
            if length <= eps:
                to_remove.append(eid)

        elif isinstance(ent, Circle):
            if ent.radius <= eps:
                to_remove.append(eid)

        elif isinstance(ent, Arc):
            if ent.radius <= eps:
                to_remove.append(eid)
                continue
            chord = np.linalg.norm(ent.end_point - ent.start_point)
            if chord <= eps:
                to_remove.append(eid)

    for eid in to_remove:
        _remove_entity(sketch, eid)

    return sketch


# ---------------------------------------------------------------------------
# Step 4 – PointOnObject inference
# ---------------------------------------------------------------------------

def _dist_point_to_line(px: float, py: float, line: Line) -> float:
    dx = px - line.pntX
    dy = py - line.pntY
    norm = math.hypot(line.dirX, line.dirY)
    if norm == 0:
        return math.hypot(dx, dy)
    return abs(dx * line.dirY - dy * line.dirX) / norm


def _dist_point_along_line_param(px: float, py: float, line: Line) -> float:
    dx = px - line.pntX
    dy = py - line.pntY
    norm2 = line.dirX ** 2 + line.dirY ** 2
    if norm2 == 0:
        return 0.0
    return (dx * line.dirX + dy * line.dirY) / norm2


def _point_on_line_segment(px: float, py: float, line: Line, eps: float) -> bool:
    if _dist_point_to_line(px, py, line) > eps:
        return False
    t = _dist_point_along_line_param(px, py, line)
    return line.startParam - eps <= t <= line.endParam + eps


def _dist_point_to_circle(px: float, py: float, circ) -> float:
    d = math.hypot(px - circ.xCenter, py - circ.yCenter)
    return abs(d - circ.radius)


def _angle_on_arc(px: float, py: float, arc: Arc) -> bool:
    angle = math.atan2(py - arc.yCenter, px - arc.xCenter)
    base = math.atan2(arc.yDir, arc.xDir)
    t = angle - base
    if arc.clockwise:
        t = -t
    sp = arc.startParam
    ep = arc.endParam
    span = ep - sp
    if span < 0:
        span += 2 * math.pi
    t_rel = (t - sp) % (2 * math.pi)
    return t_rel <= span + 1e-9


def _point_on_arc_curve(px: float, py: float, arc: Arc, eps: float) -> bool:
    if _dist_point_to_circle(px, py, arc) > eps:
        return False
    return _angle_on_arc(px, py, arc)


def _curve_endpoints(ent) -> list[np.ndarray]:
    """Return a list of geometric endpoint coordinates for a curve entity."""
    if isinstance(ent, Line):
        return [ent.start_point, ent.end_point]
    elif isinstance(ent, Arc):
        return [ent.start_point, ent.end_point]
    elif isinstance(ent, Circle):
        return []  # circles have no endpoints
    return []


def _point_is_at_curve_endpoint(px: float, py: float, curve_ent,
                                 eps: float) -> bool:
    """True if (px, py) coincides with any geometric endpoint of *curve_ent*."""
    for ep in _curve_endpoints(curve_ent):
        if math.hypot(px - ep[0], py - ep[1]) <= eps:
            return True
    return False


def add_point_on_object_constraints(sketch: datalib.Sketch, eps: float = EPS) -> datalib.Sketch:
    """Add PointOnObject (Coincident point-to-edge) constraints where a point
    geometrically lies STRICTLY in the interior of a curve.

    Rules
    -----
    * If the point is at a curve's endpoint (within *eps*), skip it – the
      relationship should be handled by a Coincident constraint instead.
    * If any existing Coincident already links the point to the curve (by any
      subnode: start, end, center, or body), skip it.
    * For each (point, curve) pair only ONE PointOnObject direction is added
      (point → curve body), not the reciprocal (curve → point).
    """
    existing = _existing_coincidence_set(sketch)
    # Build transitive Coincident clusters (Union-Find).
    # Pass entities so bare Point IDs (e.g. '9') are treated as point nodes.
    coincident_parent = _build_coincident_clusters(existing, entities=sketch.entities)
    # all_links includes both Coincident and PointOnObject-style pairs so that
    # transitive reachability via existing PoO constraints is detected.
    # e.g. Coincident(9.start, 2.end) + PointOnObject(9.start, curve7)
    #      → 2.end is already reachable from curve7 → skip PoO(2.end, curve7).
    all_links: set[frozenset] = existing | _existing_poo_set(sketch)

    # All point-like refs with their coords
    point_refs: list[tuple[str, Optional[str], np.ndarray]] = []
    for eid, ent in sketch.entities.items():
        if isinstance(ent, Point):
            point_refs.append((eid, None, np.array([ent.x, ent.y])))
        elif isinstance(ent, Line):
            point_refs.append((eid, 'start', ent.start_point))
            point_refs.append((eid, 'end', ent.end_point))
        elif isinstance(ent, Arc):
            point_refs.append((eid, 'start', ent.start_point))
            point_refs.append((eid, 'end', ent.end_point))

    curves = {eid: ent for eid, ent in sketch.entities.items()
              if isinstance(ent, (Line, Circle, Arc))}

    new_constraints: list[Constraint] = []
    added_pairs: set[frozenset] = set()  # track (pt_ref, curve_eid) to prevent duplicates

    for pt_eid, pt_sub, coord in point_refs:
        px, py = float(coord[0]), float(coord[1])

        for curve_eid, curve_ent in curves.items():
            if curve_eid == pt_eid:
                continue

            # Skip if point coincides with one of the curve's geometric endpoints.
            # Such a relationship should be expressed as Coincident, not PointOnObject.
            if _point_is_at_curve_endpoint(px, py, curve_ent, eps):
                continue

            pt_ref = _ref_str(pt_eid, pt_sub)

            # Skip if pt_ref (or any transitively Coincident point) is already
            # linked to this curve via any existing Coincident / PointOnObject.
            if _is_linked_to_curve(pt_ref, curve_eid, all_links, coincident_parent):
                continue

            # Prevent duplicate (pt_ref, curve_eid) pair
            pair_key = frozenset([pt_ref, curve_eid])
            if pair_key in added_pairs:
                continue

            # Check geometric proximity (strictly interior)
            on_curve = False
            if isinstance(curve_ent, Line):
                on_curve = _point_on_line_segment(px, py, curve_ent, eps)
            elif isinstance(curve_ent, Circle):
                on_curve = _dist_point_to_circle(px, py, curve_ent) <= eps
            elif isinstance(curve_ent, Arc):
                on_curve = _point_on_arc_curve(px, py, curve_ent, eps)

            if not on_curve:
                continue

            cid = _new_constraint_id(sketch)
            pair = frozenset([pt_ref, curve_eid])
            existing.add(pair)
            all_links.add(pair)   # keep all_links in sync for subsequent iterations
            added_pairs.add(pair_key)
            new_constraints.append(_make_point_on_object(cid, pt_ref, curve_eid))

    for c in new_constraints:
        sketch.constraints[c.identifier] = c

    return sketch


# ---------------------------------------------------------------------------
# Step 5 – Redundant constraint rewriting (delegate to sketch_utils)
# ---------------------------------------------------------------------------

def rewrite_redundant_constraints(sketch: datalib.Sketch) -> datalib.Sketch:
    """Apply `cleanse_sketch_constraints` from sketch_utils."""
    return cleanse_sketch_constraints(sketch, resolve_redundancy=True, resolve_synonym=True)


# ---------------------------------------------------------------------------
# Top-level cleanup pipeline
# ---------------------------------------------------------------------------

def cleanup_sketch(
    sketch: datalib.Sketch,
    eps: float = EPS,
) -> datalib.Sketch:
    """Apply all 5 cleanup passes to *sketch* in the correct order.

    Returns
    -------
    sketch : datalib.Sketch
        Cleaned sketch after all passes.
    """
    sketch = remove_duplicate_primitives(sketch)
    sketch = remove_degenerate_primitives(sketch, eps=eps)
    sketch = remove_redundant_points(sketch, eps=eps)
    sketch = merge_micro_gaps(sketch, eps=eps)
    sketch = add_point_on_object_constraints(sketch, eps=eps)
    sketch = rewrite_redundant_constraints(sketch)
    return sketch


# ---------------------------------------------------------------------------
# Script generation helpers
# ---------------------------------------------------------------------------

_SECTION_HEADERS = [
    "# --- Add Primitives ---",
    "# --- Add Independent Points ---",
    "# --- Add Implicit Coincident Constraints ---",
    "# --- Add Relational Constraints ---",
    "# --- Add Dimensional Constraints ---",
]

def _extract_sketch_name(script_content: str) -> Optional[str]:
    """Extract the sketch object name from a FreeCAD script string."""
    m = re.search(r"newObject\('Sketcher::SketchObject',\s*'([^']+)'\)", script_content)
    return m.group(1) if m else None


def _extract_preamble(script_content: str) -> list[str]:
    """Extract script lines up to and including the fc_sketch setup block,
    preserving the original sketch name and import statements."""
    lines = script_content.splitlines()
    result = []
    for line in lines:
        result.append(line)
        # Stop after MapMode line (end of sketch setup)
        if "MapMode" in line:
            break
    return result


def _parse_original_fcid_to_uuid(script_content: str) -> dict[str, str]:
    """Parse the fcid_to_uuid dictionary from a FreeCAD script string."""
    m = re.search(r'fcid_to_uuid\s*=\s*(\{[^}]+\})', script_content)
    if not m:
        return {}
    import ast
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return {}


# Subnode name → FreeCAD vertex index
_SUBNODE_TO_FC_VTX = {'start': 1, 'end': 2, 'center': 3}


def _coincident_lines_from_prim_map(constraints: dict, primitive_freecad_mapping: dict) -> list[str]:
    """Generate addConstraint lines for Point-Point Coincident constraints
    directly from datalib constraints, bypassing the FreeCAD round-trip.

    FreeCAD sometimes silently drops Coincident(p,q) when it deems the constraint
    implied by another active constraint (e.g. Perpendicular with point sub-refs).
    This function makes sure those lines appear regardless.
    """
    lines = []
    for c in constraints.values():
        if c.constraint_type != ConstraintType.Coincident:
            continue
        refs = [p.value for p in c.parameters if isinstance(p, LocalReferenceParameter)]
        if len(refs) != 2:
            continue
        # Only handle Point-Point (both refs have a '.')
        # PointOnObject (one ref = bare entity ID) is handled by to_freecad_commands
        if not all('.' in r for r in refs):
            continue

        mapped = []
        ok = True
        for ref in refs:
            if ref in primitive_freecad_mapping:
                gid, _ = primitive_freecad_mapping[ref]
                sub = ref.split('.')[1]
                vtx = _SUBNODE_TO_FC_VTX.get(sub, 0)
                mapped.append((gid, vtx))
            else:
                ok = False
                break
        if not ok or len(mapped) != 2:
            continue
        (g1, v1), (g2, v2) = mapped
        lines.append(
            f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {g1}, {v1}, {g2}, {v2}))"
        )
    return lines


def _parse_coincident_key(line: str) -> frozenset | None:
    """Coincident制約行から (gid, vtx) ペアの frozenset を返す。
    パース失敗時は None。"""
    m = re.search(
        r"Constraint\('Coincident',\s*(-?\d+),\s*(\d+),\s*(-?\d+),\s*(\d+)\)",
        line,
    )
    if not m:
        return None
    g1, v1, g2, v2 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return frozenset({(g1, v1), (g2, v2)})


def _generate_cleaned_commands(
    sketch: datalib.Sketch,
    converter: SketchFreecadConverter,
    original_fcid_to_uuid: dict[str, str],
) -> tuple[list[str], str, list[str]]:
    """Run the FreeCAD round-trip once and return:
      - geometry_lines  : addGeometry command strings
      - fcid_to_uuid_line: reconstructed 'fcid_to_uuid = {...}' line (or '')
      - constraint_lines : addConstraint command strings (hybrid: FC + direct)

    The geometry comes from ''dump_freecad_commands'' which reflects the actual
    cleaned sketch (removed duplicates/degenerates are gone).  The prim_map is
    used to remap the original UUID dict to the new sequential GeoIDs.
    """
    from src.sketch_handlers.freecad_utils import ensure_freecad_imported as _efi, dump_freecad_commands
    FreeCAD, _, _ = _efi()
    doc = None
    prim_map: dict = {}
    try:
        doc, fc_sketch, prim_map, _, _ = converter.to_freecad(sketch, skip_invalid=True)
        all_cmds = dump_freecad_commands(fc_sketch)
    except Exception as e:
        print(f"  [WARN] to_freecad/dump failed: {e}")
        return [], '', []
    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                pass

    # --- Geometry lines ---
    geometry_lines = [l for l in all_cmds if l.startswith("fc_sketch.addGeometry")]

    # --- fcid_to_uuid の再構築 ---
    new_fcid_to_uuid: dict[str, str] = {}
    for orig_id in sketch.entities:
        candidates = {v for k, v in prim_map.items() if k.split('.')[0] == orig_id}
        if candidates and orig_id in original_fcid_to_uuid:
            new_gid = next(iter(candidates))[0]
            new_fcid_to_uuid[str(new_gid)] = original_fcid_to_uuid[orig_id]

    if new_fcid_to_uuid:
        sorted_items = sorted(new_fcid_to_uuid.items(), key=lambda x: int(x[0]))
        inner = ", ".join(f'"{k}": "{v}"' for k, v in sorted_items)
        fcid_to_uuid_line = f"fcid_to_uuid = {{{inner}}}"
    else:
        fcid_to_uuid_line = ""

    # --- Constraint lines ---
    fc_constraint_lines = [l for l in all_cmds if l.startswith("fc_sketch.addConstraint")]

    def _direct_coincident_lines(constraints: dict) -> list[str]:
        lines = []
        for c in constraints.values():
            if c.constraint_type != ConstraintType.Coincident:
                continue
            refs = [p.value for p in c.parameters if isinstance(p, LocalReferenceParameter)]
            if len(refs) != 2 or not all('.' in r for r in refs):
                continue
            mapped, ok = [], True
            for ref in refs:
                if ref not in prim_map:
                    ok = False; break
                gid, _ = prim_map[ref]
                vtx = _SUBNODE_TO_FC_VTX.get(ref.split('.')[1], 0)
                mapped.append((gid, vtx))
            if not ok:
                continue
            (g1, v1), (g2, v2) = mapped
            lines.append(
                f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident',"
                f" {g1}, {v1}, {g2}, {v2}))"
            )
        return lines

    direct_lines = _direct_coincident_lines(sketch.constraints)

    # duplication resolution
    final_uf = UnionFind()
    final_lines: list[str] = []
    for line in fc_constraint_lines + direct_lines:
        key = _parse_coincident_key(line)
        if key is not None:
            nodes = list(key)
            if final_uf.connected(nodes[0], nodes[1]):
                continue
            final_uf.union(nodes[0], nodes[1])
        final_lines.append(line)

    return geometry_lines, fcid_to_uuid_line, final_lines


def _build_output_script(
    original_content: str,
    sketch: datalib.Sketch,
    converter: SketchFreecadConverter,
) -> str:
    """Build the output FreeCAD script preserving original metadata.

    Strategy:
      * Preamble (imports + doc/sketch setup)  ← original file (sketch name preserved)
      * Geometry (addGeometry)                 ← from FreeCAD round-trip (reflects cleanup)
      * fcid_to_uuid                           ← remapped for surviving entities
      * Constraints                            ← hybrid (FC dump + direct Coincident supplement)
      * Footer (doc.recompute)                 ← standard
    """
    preamble = _extract_preamble(original_content)
    original_fcid_to_uuid = _parse_original_fcid_to_uuid(original_content)

    geometry_lines, fcid_to_uuid_line, constraint_lines = _generate_cleaned_commands(
        sketch, converter, original_fcid_to_uuid,
    )

    # Categorise constraints for pretty section headers
    implicit_coincidents = []
    relational = []
    dimensional = []
    dimensional_types = {'Distance', 'DistanceX', 'DistanceY', 'Angle', 'Radius', 'Diameter'}

    for line in constraint_lines:
        m = re.search(r"Constraint\('([^']+)'", line)
        if not m:
            relational.append(line)
            continue
        ctype = m.group(1)
        if ctype == 'Coincident':
            implicit_coincidents.append(line)
        elif ctype in dimensional_types:
            dimensional.append(line)
        else:
            relational.append(line)

    out_lines: list[str] = []
    out_lines.extend(preamble)
    out_lines.append("")

    if geometry_lines:
        out_lines.append("# --- Add Primitives ---")
        out_lines.extend(geometry_lines)
    if fcid_to_uuid_line:
        out_lines.append(fcid_to_uuid_line)
    out_lines.append("")

    if implicit_coincidents:
        out_lines.append("# --- Add Implicit Coincident Constraints ---")
        out_lines.extend(implicit_coincidents)

    if relational:
        out_lines.append("# --- Add Relational Constraints ---")
        out_lines.extend(relational)

    if dimensional:
        out_lines.append("# --- Add Dimensional Constraints ---")
        out_lines.extend(dimensional)

    out_lines.append("")
    out_lines.append("doc.recompute()")
    out_lines.append("FreeCAD.ActiveDocument.recompute()")

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# FCStd export helper
# ---------------------------------------------------------------------------

def _save_cleaned_fcstd(
    sketch: datalib.Sketch,
    converter: 'SketchFreecadConverter',
    out_fcstd_path: str,
) -> None:
    """Save *sketch* as a FreeCAD .FCStd file at *out_fcstd_path*.

    Runs ``converter.to_freecad()`` on the **cleaned** datalib.Sketch so the
    saved document reflects all cleanup changes (duplicate removal, micro-gap
    merging, constraint rewriting, etc.).  On success the document is closed
    immediately to avoid memory leaks.

    Parameters
    ----------
    sketch : datalib.Sketch
        Cleaned sketch to export.
    converter : SketchFreecadConverter
        Converter instance (already initialised with a SketchContext).
    out_fcstd_path : str
        Absolute path where the .FCStd file should be written.
    """
    from src.sketch_handlers.freecad_utils import ensure_freecad_imported as _efi
    FreeCAD, _, _ = _efi()

    doc = None
    try:
        doc, _fc_sketch, _prim_map, _, _ = converter.to_freecad(sketch, skip_invalid=True)
        doc.saveAs(out_fcstd_path)
    except Exception as e:
        print(f"  [WARN] FCStd save failed for {os.path.basename(out_fcstd_path)}: {e}")
    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

def main(args):
    from src.sketch_handlers.context import SketchContext

    FreeCAD, _, _ = ensure_freecad_imported()

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False, raise_on_error=False)

    in_dir = args.data_dir
    out_dir = args.out_dir if args.out_dir else in_dir

    if out_dir != in_dir:
        os.makedirs(out_dir, exist_ok=True)

    py_files = sorted(glob.glob(os.path.join(in_dir, "*.py")))
    if not py_files:
        print(f"No .py sketch files found in {in_dir}")
        return

    n_ok = n_err = 0
    for py_path in tqdm(py_files):
        try:
            with open(py_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Parse FreeCAD script → datalib.Sketch
            sketch = converter.from_freecad_commands(py_path)

            # Apply cleanup
            sketch = cleanup_sketch(sketch, eps=args.eps)

            # Build output preserving original metadata
            output_script = _build_output_script(
                original_content, sketch, converter,
            )

            # Write new .py file
            basename = os.path.basename(py_path)
            out_py = os.path.join(out_dir, basename)
            with open(out_py, 'w', encoding='utf-8') as f:
                f.write(output_script)

            # Save .FCStd reflecting the cleaned sketch
            out_fcstd = out_py.replace('.py', '.FCStd')
            _save_cleaned_fcstd(sketch, converter, out_fcstd)

            n_ok += 1

        except Exception as e:
            import traceback
            print(f"[ERROR] {os.path.basename(py_path)}: {e}")
            if args.verbose:
                traceback.print_exc()
            n_err += 1

    print(f"Done. {n_ok} sketches cleaned, {n_err} errors.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clean up FreeCAD sketch scripts: remove duplicates, merge micro-gaps, "
                    "add missing constraints, and rewrite redundant constraints."
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing input .py FreeCAD sketch scripts.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory. Defaults to data_dir (in-place).')
    parser.add_argument('--eps', type=float, default=EPS,
                        help=f'Tolerance for micro-gap merging (default {EPS}).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full tracebacks for errors.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
