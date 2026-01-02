"""Convert SketchGraphs Sketch objects to FreeCAD sketches.

This module provides `convert_sketch_to_freecad()` which takes a SketchGraphs Sketch
object and converts it directly to a FreeCAD sketch.

This approach supports dimension constraints (Distance, Diameter, Radius, Length, Angle)
which are lost when converting Sketch → Sequence.
"""
from __future__ import annotations

import math
import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt

# Add FreeCAD Libs (conditionally)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part
import Sketcher

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, Arc, Circle, Line, Point, Sketch


# =============================================================================
# Unit Conversion
# =============================================================================

def _meter_to_millimeter(value: float) -> float:
    """Convert meters to millimeters (SketchGraphs uses meters, FreeCAD uses mm)."""
    return value * 1000.0


# =============================================================================
# Primitive Conversion (Sketch.entities → FreeCAD Geometry)
# =============================================================================

def _add_primitive(
    entity_idx: int,
    entity: datalib.Entity,
    fc_sketch: "Sketcher.SketchObject",
    entity_to_geoid: dict[int, int],
) -> str | None:
    """Add a primitive entity to the FreeCAD sketch.

    Parameters
    ----------
    entity_idx : int
        The entity index in the SketchGraphs Sketch.
    entity : datalib.Entity
        The entity object (Line, Arc, Circle, Point).
    fc_sketch : Sketcher.SketchObject
        The FreeCAD sketch object.
    entity_to_geoid : dict[int, int]
        Mapping from entity_idx to FreeCAD GeoId (updated in-place).

    Returns
    -------
    str | None
        Command string for debugging, or None if skipped.
    """
    cmd = None
    geo_count_before = len(fc_sketch.Geometry)

    if isinstance(entity, Line):
        # Line uses start_point and end_point (numpy arrays [x, y])
        x1 = _meter_to_millimeter(entity.start_point[0])
        y1 = _meter_to_millimeter(entity.start_point[1])
        x2 = _meter_to_millimeter(entity.end_point[0])
        y2 = _meter_to_millimeter(entity.end_point[1])
        is_construction = entity.isConstruction

        # Validate line is not degenerate (both points equal)
        if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
            raise ValueError(f"Degenerate Line: both points are equal ({x1:.3f}, {y1:.3f}) (entity_idx={entity_idx})")

        start = FreeCAD.Vector(x1, y1, 0)
        end = FreeCAD.Vector(x2, y2, 0)
        fc_sketch.addGeometry(Part.LineSegment(start, end), is_construction)
        cmd = f"fc_sketch.addGeometry(Part.LineSegment(Vector({x1}, {y1}, 0), Vector({x2}, {y2}, 0)), {is_construction})"

    elif isinstance(entity, Circle):
        cx = _meter_to_millimeter(entity.xCenter)
        cy = _meter_to_millimeter(entity.yCenter)
        r = _meter_to_millimeter(entity.radius)
        is_construction = entity.isConstruction

        # Validate radius
        if r <= 0:
            raise ValueError(f"Invalid Circle radius: {r} (entity_idx={entity_idx})")

        center = FreeCAD.Vector(cx, cy, 0)
        fc_sketch.addGeometry(Part.Circle(center, FreeCAD.Vector(0, 0, 1), r), is_construction)
        cmd = f"fc_sketch.addGeometry(Part.Circle(Vector({cx}, {cy}, 0), Vector(0, 0, 1), {r}), {is_construction})"

    elif isinstance(entity, Arc):
        import math
        cx = _meter_to_millimeter(entity.xCenter)
        cy = _meter_to_millimeter(entity.yCenter)
        r = _meter_to_millimeter(entity.radius)
        start_param = entity.startParam
        end_param = entity.endParam
        is_construction = entity.isConstruction
        clockwise = getattr(entity, 'clockwise', False)

        # Validate arc parameters
        if r <= 0:
            raise ValueError(f"Invalid Arc radius: {r} (entity_idx={entity_idx})")
        if abs(start_param - end_param) < 1e-6:
            raise ValueError(f"Degenerate Arc: start_angle == end_angle ({start_param}) (entity_idx={entity_idx})")

        # Calculate arc points (matching Sequence converter logic)
        # rx, ry define the reference direction for angle=0
        rx = getattr(entity, 'xDir', 1.0)
        ry = getattr(entity, 'yDir', 0.0)

        def get_pt(angle_offset):
            base_angle = math.atan2(ry, rx)
            if clockwise:
                angle_offset *= -1
            theta = base_angle + angle_offset
            return FreeCAD.Vector(cx + r*math.cos(theta), cy + r*math.sin(theta), 0)

        p_start = get_pt(start_param)
        p_end = get_pt(end_param)

        # Midpoint calculation: use modulo-based method for wrap-around handling
        # This correctly handles negative angles by normalizing to [0, 2π] range
        sp = start_param % (2*math.pi)
        ep = end_param % (2*math.pi)
        if sp > ep:
            ep += 2*math.pi
        mid_param = (sp + ep) / 2.0
        p_mid = get_pt(mid_param)

        # For clockwise arcs, swap start and end (FreeCAD expects CCW)
        if clockwise:
            p_start, p_end = p_end, p_start

        fc_sketch.addGeometry(Part.ArcOfCircle(p_start, p_mid, p_end), is_construction)
        cmd = f"fc_sketch.addGeometry(Part.ArcOfCircle({p_start}, {p_mid}, {p_end}), {is_construction})"

    elif isinstance(entity, Point):
        x = _meter_to_millimeter(entity.x)
        y = _meter_to_millimeter(entity.y)
        is_construction = entity.isConstruction
        fc_sketch.addGeometry(Part.Point(FreeCAD.Vector(x, y, 0)), is_construction)
        cmd = f"fc_sketch.addGeometry(Part.Point(Vector({x}, {y}, 0)), {is_construction})"

    else:
        # Skip External and other unsupported entity types
        cmd = f"# Skipped entity type: {type(entity).__name__}"

    # Map entity_idx to GeoId
    geo_count_after = len(fc_sketch.Geometry)
    if geo_count_after > geo_count_before:
        geo_id = geo_count_after - 1
        entity_to_geoid[entity_idx] = geo_id

    return cmd


# =============================================================================
# Reference Resolution
# =============================================================================

def _resolve_ref(
    ref_str: str,
    entity_to_geoid: dict[int, int],
    entities: dict[str, datalib.Entity],
) -> tuple[int, int] | None:
    """Resolve a constraint reference string to (GeoId, PointPos).

    Parameters
    ----------
    ref_str : str
        Reference string like '7' or '7.start'
    entity_to_geoid : dict[int, int]
        Mapping from entity_idx to FreeCAD GeoId.
    entities : dict[str, Entity]
        The sketch entities dict.

    Returns
    -------
    tuple[int, int] | None
        (GeoId, PointPos) where PointPos is:
        - 0 = Edge
        - 1 = Start point
        - 2 = End point
        - 3 = Center point
        Returns None if reference cannot be resolved.
    """
    if '.' in ref_str:
        parts = ref_str.split('.')
        if len(parts) != 2:
            return None
        try:
            entity_idx = int(parts[0])
        except ValueError:
            return None
        subnode_name = parts[1]
        if entity_idx not in entity_to_geoid:
            return None
        geo_id = entity_to_geoid[entity_idx]

        # Get entity to check for clockwise arc
        entity = entities.get(str(entity_idx))
        is_clockwise = isinstance(entity, Arc) and getattr(entity, 'clockwise', False)

        if subnode_name == 'start':
            return (geo_id, 2 if is_clockwise else 1)
        elif subnode_name == 'end':
            return (geo_id, 1 if is_clockwise else 2)
        elif subnode_name == 'center':
            return (geo_id, 3)
        else:
            return None
    else:
        try:
            entity_idx = int(ref_str)
        except ValueError:
            return None
        if entity_idx not in entity_to_geoid:
            return None
        geo_id = entity_to_geoid[entity_idx]

        # Check if it's a Point entity
        entity = entities.get(str(entity_idx))
        if isinstance(entity, Point):
            return (geo_id, 1)  # Point vertex
        return (geo_id, 0)  # Edge reference


def _get_refs_from_constraint(
    constraint: datalib.Constraint,
    entity_to_geoid: dict[int, int],
    entities: dict[str, datalib.Entity],
) -> list[tuple[int, int]]:
    """Extract resolved references from a constraint.

    Returns list of (GeoId, PointPos) tuples.
    """
    refs = []
    for param in constraint.parameters:
        if not hasattr(param, 'value'):
            continue
        value = param.value
        if not isinstance(value, str):
            continue
        # Skip enum values like 'MINIMUM', 'LEFT', 'RIGHT'
        if '.' in value:
            parts = value.split('.')
            if not parts[0].isdigit():
                continue
            if len(parts) == 2 and parts[1] not in ('start', 'end', 'center'):
                continue
        elif not value.isdigit():
            continue

        resolved = _resolve_ref(value, entity_to_geoid, entities)
        if resolved is not None:
            refs.append(resolved)

    return refs



def _get_quantity_param(constraint: datalib.Constraint) -> float | None:
    """Extract quantity parameter (distance, angle, etc.) from constraint.

    SketchGraphs QuantityParameter has two attributes:
    - .value: often 0.0 (not reliable for dimension constraints)
    - .expression: contains the actual value like '0.0127 m' or '45.0 deg'

    We parse .expression to get the correct value.
    """
    for param in constraint.parameters:
        if type(param).__name__ == 'QuantityParameter':
            # Try .expression first (contains actual value like '0.0127 m')
            if hasattr(param, 'expression') and param.expression:
                expr = str(param.expression).strip()
                # Parse value and unit (e.g., '0.0127 m', '45.0 deg')
                parts = expr.split()
                if parts:
                    try:
                        value = float(parts[0])
                        unit = parts[1] if len(parts) > 1 else ''
                        # Convert based on unit
                        if unit in ('m', 'METER'):
                            return _meter_to_millimeter(value)
                        elif unit in ('mm', 'MILLIMETER'):
                            return value
                        elif unit in ('deg', 'DEGREE'):
                            import math
                            return math.radians(value)
                        elif unit in ('rad', 'RADIAN'):
                            return value
                        else:
                            # Unknown unit, assume meters for length
                            return _meter_to_millimeter(value)
                    except (ValueError, IndexError):
                        pass

            # Fallback to .value (may be string or float)
            value = param.value
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    continue
            if value != 0.0:  # Skip if 0.0 as it's likely default
                return _meter_to_millimeter(value)
    return None


def _resolve_tangent(
    constraints: dict[str, datalib.Constraint],
    entity_to_geoid: dict[int, int],
    entities: dict[str, datalib.Entity],
) -> tuple[set[str], dict[str, list[tuple[int, int]]]]:
    """Pre-process constraints to resolve Tangent+Coincident pairs.

    FreeCAD handles Tangent(edge, edge) + Coincident(point, point) better as
    a single Tangent(edge, point, edge, point). This function identifies such
    pairs and returns resolved reference info for Tangent constraints.

    Returns:
        - Set of Coincident IDs to skip
        - Dict mapping Tangent constraint IDs to resolved refs list [(geo1, pt1), (geo2, pt2)]
    """
    skip_coincident_ids: set[str] = set()
    resolved_tangent_refs: dict[str, list[tuple[int, int]]] = {}

    # Build Coincident map: (geo1_id, geo2_id) -> (refs_parsed, coincident_id)
    # refs_parsed is [(geo1, pt1), (geo2, pt2)] in constraint order
    # Now handles cases where one ref has subnode and one doesn't (e.g., Circle + Line.end)
    coincident_map: dict[tuple[int, int], tuple[list[tuple[int, int]], str]] = {}
    for cid, c in constraints.items():
        if c.constraint_type.name == 'Coincident':
            refs_parsed = []  # List of (geo_id, pt_id)
            for p in c.parameters:
                if type(p).__name__ == 'LocalReferenceParameter':
                    ref_str = str(p.value)
                    if '.' in ref_str:
                        # Has subnode (e.g., '1.end')
                        base, subnode = ref_str.split('.', 1)
                        try:
                            base_idx = int(base)
                            if base_idx in entity_to_geoid:
                                geo_id = entity_to_geoid[base_idx]
                                # Check if Arc with clockwise=True - swap start/end
                                entity = entities.get(str(base_idx))
                                is_clockwise = (type(entity).__name__ == 'Arc' and
                                              getattr(entity, 'clockwise', False))
                                subnode_lower = subnode.lower()
                                if subnode_lower == 'start':
                                    pt_id = 2 if is_clockwise else 1
                                elif subnode_lower == 'end':
                                    pt_id = 1 if is_clockwise else 2
                                elif subnode_lower == 'center':
                                    pt_id = 3
                                else:
                                    pt_id = 0
                                refs_parsed.append((geo_id, pt_id))
                        except ValueError:
                            pass
                    else:
                        # No subnode (e.g., '4' for Circle) - treat as edge reference
                        try:
                            base_idx = int(ref_str)
                            if base_idx in entity_to_geoid:
                                geo_id = entity_to_geoid[base_idx]
                                refs_parsed.append((geo_id, 0))  # pt_id=0 for edge
                        except ValueError:
                            pass

            # Need at least one ref with pt_id > 0 to be useful for TangentViaPoint
            if len(refs_parsed) == 2:
                geo1, pt1 = refs_parsed[0]
                geo2, pt2 = refs_parsed[1]
                # Only useful if at least one has a point (not both edges)
                if pt1 > 0 or pt2 > 0:
                    # Use normalized key for lookup, but store full refs_parsed
                    key = (min(geo1, geo2), max(geo1, geo2))
                    # Prioritize point-point over point-edge (like Sequence converter)
                    if key in coincident_map:
                        existing_refs, _ = coincident_map[key]
                        existing_pt1 = existing_refs[0][1]
                        existing_pt2 = existing_refs[1][1]
                        # Only replace if current is point-point and existing is not
                        if pt1 > 0 and pt2 > 0 and not (existing_pt1 > 0 and existing_pt2 > 0):
                            coincident_map[key] = (refs_parsed, cid)
                    else:
                        coincident_map[key] = (refs_parsed, cid)

    # Find Tangent constraints that have matching Coincident
    for tid, t in constraints.items():
        if t.constraint_type.name == 'Tangent':
            geo_ids = []
            for p in t.parameters:
                if type(p).__name__ == 'LocalReferenceParameter':
                    ref_str = str(p.value)
                    base = ref_str.split('.')[0] if '.' in ref_str else ref_str
                    try:
                        base_idx = int(base)
                        if base_idx in entity_to_geoid:
                            geo_ids.append(entity_to_geoid[base_idx])
                    except ValueError:
                        pass

            if len(geo_ids) == 2:
                key = (min(geo_ids[0], geo_ids[1]), max(geo_ids[0], geo_ids[1]))
                if key in coincident_map:
                    refs_parsed, coincident_id = coincident_map[key]
                    # Sort refs_parsed so smaller geo_id comes first (like Sequence converter)
                    if refs_parsed[0][0] > refs_parsed[1][0]:
                        refs_parsed = [refs_parsed[1], refs_parsed[0]]
                    resolved_tangent_refs[tid] = refs_parsed
                    skip_coincident_ids.add(coincident_id)

    return skip_coincident_ids, resolved_tangent_refs


def _resolve_redundant_coincident(
    constraints: dict[str, datalib.Constraint],
    entity_to_geoid: dict[int, int],
    entities: dict,
) -> set[str]:
    """Detect and return IDs of redundant Coincident constraints to skip.

    When same (geo1, geo2) pair has both:
    - PointOnObject (one ref has pt=0, other has pt>0)
    - Point-Point (both refs have pt>0)

    The PointOnObject is redundant and should be skipped.
    This mirrors resolve_redundant_coincident from convert_sequence_to_freecad.py.

    Returns:
        Set of Coincident constraint IDs to skip
    """
    skip_coincident_ids: set[str] = set()

    # Parse all Coincident constraints: (geo1, geo2) -> [(pt1, pt2, cid), ...]
    coincident_map: dict[tuple[int, int], list[tuple[int, int, str]]] = {}

    for cid, c in constraints.items():
        if c.constraint_type.name == 'Coincident':
            refs_parsed = []
            for p in c.parameters:
                if type(p).__name__ == 'LocalReferenceParameter':
                    ref_str = str(p.value)
                    if '.' in ref_str:
                        base, subnode = ref_str.split('.', 1)
                        try:
                            base_idx = int(base)
                            if base_idx in entity_to_geoid:
                                geo_id = entity_to_geoid[base_idx]
                                pt_id = {'start': 1, 'end': 2, 'center': 3}.get(subnode.lower(), 0)
                                refs_parsed.append((geo_id, pt_id))
                        except ValueError:
                            pass
                    else:
                        try:
                            base_idx = int(ref_str)
                            if base_idx in entity_to_geoid:
                                geo_id = entity_to_geoid[base_idx]
                                # Point entity (no subnode) should use pt_id=1, not 0
                                # Note: entities keys are str, base_idx is int
                                entity = entities.get(str(base_idx)) or entities.get(base_idx)
                                if entity is not None and type(entity).__name__ == 'Point':
                                    refs_parsed.append((geo_id, 1))
                                else:
                                    refs_parsed.append((geo_id, 0))
                        except ValueError:
                            pass

            if len(refs_parsed) == 2:
                geo1, pt1 = refs_parsed[0]
                geo2, pt2 = refs_parsed[1]
                # Normalize key ordering
                if geo1 > geo2:
                    geo1, pt1, geo2, pt2 = geo2, pt2, geo1, pt1
                key = (geo1, geo2)
                if key not in coincident_map:
                    coincident_map[key] = []
                coincident_map[key].append((pt1, pt2, cid))

    # For each geo pair with multiple Coincident, keep Point-Point, skip PointOnObject
    for key, entries in coincident_map.items():
        if len(entries) <= 1:
            continue

        # Find the best entry (Point-Point preferred)
        best_entry = None
        for pt1, pt2, cid in entries:
            if pt1 > 0 and pt2 > 0:
                best_entry = (pt1, pt2, cid)
                break

        # If no Point-Point found, keep ALL PointOnObject constraints (matching sequence behavior)
        if best_entry is None:
            continue  # Don't skip any

        # Skip all entries except the best (Point-Point)
        for pt1, pt2, cid in entries:
            if cid != best_entry[2]:
                skip_coincident_ids.add(cid)

    return skip_coincident_ids


# =============================================================================
# Constraint Conversion
# =============================================================================

def _add_constraint(
    constraint: datalib.Constraint,
    constraint_id: str,
    fc_sketch: "Sketcher.SketchObject",
    entity_to_geoid: dict[int, int],
    entities: dict[str, datalib.Entity],
    resolved_tangent_refs: dict[str, list[tuple[int, int]]] | None = None,
) -> str | None:
    """Add a constraint to the FreeCAD sketch.

    Returns command string for debugging.

    resolved_tangent_refs: dict mapping constraint IDs to resolved refs [(geo1, pt1), (geo2, pt2)]
    """
    ctype = constraint.constraint_type.name
    refs = _get_refs_from_constraint(constraint, entity_to_geoid, entities)
    cmd = None

    r1 = refs[0] if len(refs) > 0 else None
    r2 = refs[1] if len(refs) > 1 else None
    r3 = refs[2] if len(refs) > 2 else None

    try:
        # ===== Geometric Constraints =====
        if ctype == 'Coincident':
            if r1 and r2:
                if r1[0] == -1 or r2[0] == -1:
                    cmd = f"# Skipped External Coincident"
                elif r1[0] == r2[0]:
                    # Same geometry - point on own edge is tautological
                    cmd = f"# Skipped self-referencing Coincident (geo {r1[0]})"
                elif r1[1] == 0 and r2[1] == 0:
                    cmd = f"# Skipped Edge-Edge Coincident"
                elif r1[1] == 0:  # r1 is Edge, r2 is Point
                    fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', r2[0], r2[1], r1[0]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', {r2[0]}, {r2[1]}, {r1[0]}))"
                elif r2[1] == 0:  # r2 is Edge, r1 is Point
                    fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', r1[0], r1[1], r2[0]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', {r1[0]}, {r1[1]}, {r2[0]}))"
                else:  # Both are Points
                    fc_sketch.addConstraint(Sketcher.Constraint('Coincident', r1[0], r1[1], r2[0], r2[1]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"

        elif ctype == 'Horizontal':
            if len(refs) == 1 and r1:
                fc_sketch.addConstraint(Sketcher.Constraint('Horizontal', r1[0]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Horizontal', {r1[0]}))"
            elif len(refs) == 2 and r1 and r2:
                fc_sketch.addConstraint(Sketcher.Constraint('Horizontal', r1[0], r1[1], r2[0], r2[1]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Horizontal', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"

        elif ctype == 'Vertical':
            if len(refs) == 1 and r1:
                fc_sketch.addConstraint(Sketcher.Constraint('Vertical', r1[0]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Vertical', {r1[0]}))"
            elif len(refs) == 2 and r1 and r2:
                fc_sketch.addConstraint(Sketcher.Constraint('Vertical', r1[0], r1[1], r2[0], r2[1]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Vertical', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"

        elif ctype == 'Parallel':
            if r1 and r2:
                fc_sketch.addConstraint(Sketcher.Constraint('Parallel', r1[0], r2[0]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Parallel', {r1[0]}, {r2[0]}))"

        elif ctype == 'Perpendicular':
            if r1 and r2:
                fc_sketch.addConstraint(Sketcher.Constraint('Perpendicular', r1[0], r2[0]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Perpendicular', {r1[0]}, {r2[0]}))"

        elif ctype == 'Tangent':
            if r1 and r2:
                # Check if _resolve_tangent has pre-computed point info
                if resolved_tangent_refs and constraint_id in resolved_tangent_refs:
                    refs_list = resolved_tangent_refs[constraint_id]
                    geo1_id, pt1 = refs_list[0]
                    geo2_id, pt2 = refs_list[1]
                    # Use appropriate constraint type based on point info
                    if pt1 > 0 and pt2 > 0:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', geo1_id, pt1, geo2_id, pt2))
                        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {geo1_id}, {pt1}, {geo2_id}, {pt2}))"
                    elif pt1 > 0 and pt2 == 0:
                        # Point-to-Edge: use TangentViaPoint
                        fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', geo1_id, geo2_id, geo1_id, pt1))
                        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {geo1_id}, {geo2_id}, {geo1_id}, {pt1}))"
                    elif pt1 == 0 and pt2 > 0:
                        # Edge-to-Point: use TangentViaPoint
                        fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', geo1_id, geo2_id, geo2_id, pt2))
                        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {geo1_id}, {geo2_id}, {geo2_id}, {pt2}))"
                    else:
                        # Edge-to-Edge (shouldn't happen from resolved refs, but fallback)
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', geo1_id, geo2_id))
                        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {geo1_id}, {geo2_id}))"
                elif r1[1] > 0 and r2[1] > 0:
                    # Point-to-Point Tangent
                    fc_sketch.addConstraint(Sketcher.Constraint('Tangent', r1[0], r1[1], r2[0], r2[1]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"
                elif r1[1] > 0 and r2[1] == 0:
                    # Point-to-Edge: use TangentViaPoint (e.g., Line.end tangent to Circle)
                    fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', r1[0], r2[0], r1[0], r1[1]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {r1[0]}, {r2[0]}, {r1[0]}, {r1[1]}))"
                elif r1[1] == 0 and r2[1] > 0:
                    # Edge-to-Point: use TangentViaPoint (e.g., Circle tangent to Line.end)
                    fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', r1[0], r2[0], r2[0], r2[1]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {r1[0]}, {r2[0]}, {r2[0]}, {r2[1]}))"
                else:
                    # Edge-edge Tangent without point info
                    fc_sketch.addConstraint(Sketcher.Constraint('Tangent', r1[0], r2[0]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {r1[0]}, {r2[0]}))"

        elif ctype == 'Equal':
            if r1 and r2:
                fc_sketch.addConstraint(Sketcher.Constraint('Equal', r1[0], r2[0]))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Equal', {r1[0]}, {r2[0]}))"

        elif ctype == 'Concentric':
            if r1 and r2:
                fc_sketch.addConstraint(Sketcher.Constraint('Coincident', r1[0], 3, r2[0], 3))
                cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {r1[0]}, 3, {r2[0]}, 3))  # Concentric"

        elif ctype == 'Midpoint':
            if r1 and r2:
                if r3:
                    # Three-point: symmetric constraint
                    # SketchGraphs Midpoint(r1, r2, r3) logic is not always (Mid, P1, P2)
                    # We must determine which reference is the semantic midpoint (symmetry center).

                    def get_coords_3d(ref):
                        gid, pid = ref
                        if gid < 0: return None
                        try:
                            # fc_sketch.Geometry is accessible here
                            if gid >= len(fc_sketch.Geometry): return None
                            geo = fc_sketch.Geometry[gid]
                        except:
                            return None

                        # Handle Geometry Types
                        if isinstance(geo, Part.Point):
                            return (geo.X, geo.Y, 0.0)
                        elif isinstance(geo, (Part.LineSegment, Part.Line)):
                            # pid=1:Start, pid=2:End, pid=3:Center, pid=0:Edge(Use Center)
                            if pid == 2: # End
                                v = geo.EndPoint
                                return (v.x, v.y, v.z)
                            else: # Start (1) or Center (3) or Edge (0)
                                if pid == 3 or pid == 0:
                                    v = (geo.StartPoint + geo.EndPoint) * 0.5
                                else:
                                    v = geo.StartPoint
                                return (v.x, v.y, v.z)
                        elif isinstance(geo, (Part.Circle, Part.ArcOfCircle)):
                            if pid == 1 and hasattr(geo, 'StartPoint'): v = geo.StartPoint
                            elif pid == 2 and hasattr(geo, 'EndPoint'): v = geo.EndPoint
                            else: v = geo.Center
                            return (v.x, v.y, v.z)
                        return None

                    # Check permutations
                    candidates = [(r1, r2, r3), (r2, r1, r3), (r3, r1, r2)]
                    best_perm = None
                    min_err = 1e9

                    for mid, p1, p2 in candidates:
                        cm = get_coords_3d(mid)
                        c1 = get_coords_3d(p1)
                        c2 = get_coords_3d(p2)

                        if cm and c1 and c2:
                            mid_calc_x = (c1[0] + c2[0]) * 0.5
                            mid_calc_y = (c1[1] + c2[1]) * 0.5
                            # Dist
                            err = ((cm[0]-mid_calc_x)**2 + (cm[1]-mid_calc_y)**2)**0.5
                            if err < min_err:
                                min_err = err
                                best_perm = (mid, p1, p2)

                    # Selection
                    mid_ref, p1_ref, p2_ref = None, None, None

                    # Threshold for geometric match (0.1mm)
                    if best_perm and min_err < 0.1:
                        mid_ref, p1_ref, p2_ref = best_perm
                    else:
                        # Fallback Heuristic: Point Entity Preference
                        pt_indices = []
                        geos = [fc_sketch.Geometry[r[0]] if r[0]>=0 and r[0]<len(fc_sketch.Geometry) else None for r in [r1, r2, r3]]
                        for i, g in enumerate(geos):
                            if isinstance(g, Part.Point):
                                pt_indices.append(i)

                        if len(pt_indices) == 1:
                            idx = pt_indices[0]
                            if idx == 0: mid_ref, p1_ref, p2_ref = r1, r2, r3
                            elif idx == 1: mid_ref, p1_ref, p2_ref = r2, r1, r3
                            else: mid_ref, p1_ref, p2_ref = r3, r1, r2
                        else:
                            # Default fallback
                            mid_ref, p1_ref, p2_ref = r1, r2, r3

                    # Add Constraint
                    # For FreeCAD Symmetric:
                    # - pos_line=0: Line Symmetry (Reflection).
                    # - pos_line>0: Point Symmetry (Midpoint).
                    geo_mid = fc_sketch.Geometry[mid_ref[0]] if mid_ref[0] >= 0 else None
                    if isinstance(geo_mid, Part.Point):
                        # Point entity as symmetry center. Supported.
                        pass

                    fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', p1_ref[0], p1_ref[1], p2_ref[0], p2_ref[1], mid_ref[0], mid_ref[1]))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {p1_ref[0]}, {p1_ref[1]}, {p2_ref[0]}, {p2_ref[1]}, {mid_ref[0]}, {mid_ref[1]}))"
                else:
                    # Two-ref: point on curve midpoint
                    if r1[1] == 0:
                        c_ref, p_ref = r1, r2
                    else:
                        c_ref, p_ref = r2, r1

                    geo_type = fc_sketch.Geometry[c_ref[0]]
                    if isinstance(geo_type, Part.ArcOfCircle):
                        # For Arc: Midpoint along arc curve cannot be expressed in FreeCAD
                        # without additional geometry or dimension constraints - skip
                        cmd = f"# Skipped Midpoint on Arc (not expressible in FreeCAD)"
                    else:
                        # For Line: Symmetric constraint gives linear midpoint
                        fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', c_ref[0], 1, c_ref[0], 2, p_ref[0], p_ref[1]))
                        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {c_ref[0]}, 1, {c_ref[0]}, 2, {p_ref[0]}, {p_ref[1]}))  # Midpoint"

        # ===== Dimension Constraints =====
        elif ctype == 'Distance':
            # Distance should be calculated from geometry, not from expression
            # Skip if either reference is External (geo_id == -1)
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                # Helper to get point coordinates from geometry
                def get_point(geo_id, pt_id):
                    geo = fc_sketch.Geometry[geo_id]
                    if pt_id == 1:  # Start
                        if hasattr(geo, 'StartPoint'):
                            return geo.StartPoint
                        elif hasattr(geo, 'X'):  # Point geometry
                            return FreeCAD.Vector(geo.X, geo.Y, 0)
                    elif pt_id == 2:  # End
                        if hasattr(geo, 'EndPoint'):
                            return geo.EndPoint
                    elif pt_id == 3:  # Center
                        if hasattr(geo, 'Center'):
                            return geo.Center
                    return None

                # Helper to compute perpendicular distance from point to line
                def point_to_line_distance(pt, line_geo):
                    if not hasattr(line_geo, 'StartPoint') or not hasattr(line_geo, 'EndPoint'):
                        return None
                    p1 = line_geo.StartPoint
                    p2 = line_geo.EndPoint
                    d = p2 - p1
                    d_len = d.Length
                    if d_len < 1e-9:
                        return (pt - p1).Length
                    t = (pt - p1).dot(d) / (d_len * d_len)
                    closest = p1 + d * t
                    return (pt - closest).Length

                # IMPORTANT: Check Edge-to-Edge case FIRST and skip for non-LineSegments
                # This matches old converter behavior and prevents geometry distortion
                if r1[1] == 0 and r2[1] == 0:
                    geo1 = fc_sketch.Geometry[r1[0]]
                    geo2 = fc_sketch.Geometry[r2[0]]
                    if not isinstance(geo1, Part.LineSegment) or not isinstance(geo2, Part.LineSegment):
                        cmd = f"# Skipped Distance: Edge-to-Edge only supported for LineSegments"
                    elif hasattr(geo1, 'StartPoint'):
                        p1 = geo1.StartPoint
                        dist = point_to_line_distance(p1, geo2)
                        if dist is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], 1, r2[0], 0, dist))
                            cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, 1, {r2[0]}, 0, {dist}))"
                        else:
                            cmd = f"# Skipped Distance: Could not compute line-to-line distance"
                    else:
                        cmd = f"# Skipped Distance: geo1 has no StartPoint"
                elif r1[1] > 0 and r2[1] > 0:
                    # Point-to-Point distance
                    p1 = get_point(r1[0], r1[1])
                    p2 = get_point(r2[0], r2[1])
                    if p1 and p2:
                        dist = (p1 - p2).Length
                        fc_sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], r1[1], r2[0], r2[1], dist))
                        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}, {dist}))"
                    else:
                        cmd = f"# Skipped Distance: Could not get points"
                elif r1[1] > 0:
                    # Point to Edge distance
                    p1 = get_point(r1[0], r1[1])
                    geo2 = fc_sketch.Geometry[r2[0]]
                    if p1:
                        dist = point_to_line_distance(p1, geo2)
                        if dist is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], r1[1], r2[0], 0, dist))
                            cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, {r1[1]}, {r2[0]}, 0, {dist}))"
                        else:
                            cmd = f"# Skipped Distance: Could not compute point-to-line distance"
                    else:
                        cmd = f"# Skipped Distance: Could not get point"
                elif r2[1] > 0:
                    # Edge to Point distance
                    p2 = get_point(r2[0], r2[1])
                    geo1 = fc_sketch.Geometry[r1[0]]
                    if p2:
                        dist = point_to_line_distance(p2, geo1)
                        if dist is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', r2[0], r2[1], r1[0], 0, dist))
                            cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {r2[0]}, {r2[1]}, {r1[0]}, 0, {dist}))"
                        else:
                            cmd = f"# Skipped Distance: Could not compute point-to-line distance"
                    else:
                        cmd = f"# Skipped Distance: Could not get point"
                else:
                    cmd = f"# Skipped Distance: Unexpected reference types"
            else:
                cmd = f"# Skipped Distance: missing refs"

        elif ctype == 'Diameter':
            # Diameter should be calculated from geometry, not from expression
            # (expression in SketchGraphs Diameter constraint is not the actual diameter value)
            if r1:
                geo = fc_sketch.Geometry[r1[0]]
                if hasattr(geo, 'Radius'):
                    diameter = geo.Radius * 2
                    fc_sketch.addConstraint(Sketcher.Constraint('Diameter', r1[0], diameter))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Diameter', {r1[0]}, {diameter}))"
                else:
                    cmd = f"# Skipped Diameter: geometry has no Radius attribute"
            else:
                cmd = f"# Skipped Diameter: missing reference"

        elif ctype == 'Radius':
            # Radius should be calculated from geometry, not from expression
            if r1:
                geo = fc_sketch.Geometry[r1[0]]
                if hasattr(geo, 'Radius'):
                    radius = geo.Radius
                    fc_sketch.addConstraint(Sketcher.Constraint('Radius', r1[0], radius))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Radius', {r1[0]}, {radius}))"
                else:
                    cmd = f"# Skipped Radius: geometry has no Radius attribute"
            else:
                cmd = f"# Skipped Radius: missing reference"

        elif ctype == 'Length':
            # Length should be calculated from geometry (line length)
            if r1:
                geo = fc_sketch.Geometry[r1[0]]
                if hasattr(geo, 'StartPoint') and hasattr(geo, 'EndPoint'):
                    length = (geo.EndPoint - geo.StartPoint).Length
                    fc_sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], length))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, {length}))  # Length"
                else:
                    cmd = f"# Skipped Length: geometry has no StartPoint/EndPoint"
            else:
                cmd = f"# Skipped Length: missing reference"

        elif ctype == 'Angle':
            # Angle should be calculated from geometry (angle between two lines)
            if r1 and r2:
                import math
                geo1 = fc_sketch.Geometry[r1[0]]
                geo2 = fc_sketch.Geometry[r2[0]]
                if hasattr(geo1, 'StartPoint') and hasattr(geo1, 'EndPoint') and \
                   hasattr(geo2, 'StartPoint') and hasattr(geo2, 'EndPoint'):
                    # Calculate direction vectors
                    d1 = geo1.EndPoint - geo1.StartPoint
                    d2 = geo2.EndPoint - geo2.StartPoint
                    # Calculate angle between vectors
                    dot = d1.x * d2.x + d1.y * d2.y
                    cross = d1.x * d2.y - d1.y * d2.x
                    angle = math.atan2(cross, dot)
                    fc_sketch.addConstraint(Sketcher.Constraint('Angle', r1[0], r2[0], angle))
                    cmd = f"fc_sketch.addConstraint(Sketcher.Constraint('Angle', {r1[0]}, {r2[0]}, {angle}))"
                else:
                    cmd = f"# Skipped Angle: geometry has no StartPoint/EndPoint"
            else:
                cmd = f"# Skipped Angle: missing references"

        else:
            cmd = f"# Skipped constraint type: {ctype}"

    except Exception as e:
        cmd = f"# Error in {ctype}: {e}"

    return cmd


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_sketch_to_freecad(
    sketch: Sketch,
    doc_name: str = "SketchDoc",
) -> tuple[FreeCAD.Document, "Sketcher.SketchObject", list[str]]:
    """Convert a SketchGraphs Sketch object directly to FreeCAD sketch.

    Parameters
    ----------
    sketch : Sketch
        SketchGraphs Sketch object.
    doc_name : str
        Name for the FreeCAD document.

    Returns
    -------
    tuple[FreeCAD.Document, Sketcher.SketchObject]
        The FreeCAD document and sketch object, or (None, None) if failed.
    """
    doc = FreeCAD.newDocument(doc_name)
    body = doc.addObject('PartDesign::Body', 'Body')
    fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    xy_plane = body.Origin.OriginFeatures[3]
    fc_sketch.AttachmentSupport = xy_plane
    fc_sketch.MapMode = 'FlatFace'

    history_commands = []
    history_commands.append("import FreeCAD")
    history_commands.append("import Part")
    history_commands.append("import Sketcher")
    history_commands.append("from FreeCAD import Vector")
    history_commands.append(f"doc = FreeCAD.newDocument('{doc_name}')")
    history_commands.append("body = doc.addObject('PartDesign::Body', 'Body')")
    history_commands.append("fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')")
    history_commands.append("xy_plane = body.Origin.OriginFeatures[3]")
    history_commands.append("fc_sketch.AttachmentSupport = xy_plane")
    history_commands.append("fc_sketch.MapMode = 'FlatFace'")

    entity_to_geoid: dict[int, int] = {}

    # Step 1: Add primitives
    for entity_id_str, entity in sketch.entities.items():
        entity_idx = int(entity_id_str)
        cmd = _add_primitive(entity_idx, entity, fc_sketch, entity_to_geoid)
        history_commands.append(cmd)

    # Step 1.5: Pre-process Tangent+Coincident pairs
    # Sequence converter uses resolve_tangent to convert Tangent refs and disable paired Coincident
    skip_coincident_ids, resolved_tangent_refs = _resolve_tangent(sketch.constraints, entity_to_geoid, sketch.entities)

    # Step 1.6: Detect redundant Coincident constraints (PointOnObject vs Point-Point for same geo pair)
    # Sequence converter uses this to skip point-edge when point-point exists for same geo pair
    skip_redundant_ids = _resolve_redundant_coincident(sketch.constraints, entity_to_geoid, sketch.entities)
    skip_coincident_ids.update(skip_redundant_ids)

    # Step 2: Add constraints
    for constraint_id, constraint in sketch.constraints.items():
        # Skip Coincident constraints that are handled by Tangent resolution
        if constraint_id in skip_coincident_ids:
            history_commands.append(f"# Skipped Coincident (handled by Tangent): {constraint_id}")
            continue
        cmd = _add_constraint(constraint, constraint_id, fc_sketch, entity_to_geoid, sketch.entities, resolved_tangent_refs)
        history_commands.append(cmd)

    doc.recompute()
    history_commands.append("doc.recompute()")

    return doc, fc_sketch, history_commands


def draw_svg(
    doc: FreeCAD.Document,
    sketch: "Sketcher.SketchObject",
    output_svg_path: str,
    debug: bool = False,
):
    """
    Generate SVG from sketch geometry.
    - Construction lines are rendered as dashed lines.
    - Regular geometry is rendered as solid lines.
    """
    import math

    # Collect geometry data and compute bounding box
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
    svg_elements = []

    for i, geo in enumerate(sketch.Geometry):
        is_construction = sketch.getConstruction(i)

        # Extract geometry coordinates
        try:
            # Check for Circle/Arc FIRST (they also have StartPoint/EndPoint)
            if hasattr(geo, 'Center') and hasattr(geo, 'Radius'):
                # Circle or Arc
                c = geo.Center
                r = geo.Radius
                x_min = min(x_min, c.x - r)
                y_min = min(y_min, c.y - r)
                x_max = max(x_max, c.x + r)
                y_max = max(y_max, c.y + r)

                # Distinguish full circle from arc by geometry type or angle span
                geo_type_name = type(geo).__name__
                is_full_circle = 'Circle' in geo_type_name and 'Arc' not in geo_type_name

                if not is_full_circle and hasattr(geo, 'FirstParameter') and hasattr(geo, 'LastParameter'):
                    # Check if it's nearly a full circle (angle span ≈ 2π)
                    angle_span = abs(geo.LastParameter - geo.FirstParameter)
                    if angle_span >= 2 * math.pi - 0.01:
                        is_full_circle = True

                if is_full_circle:
                    svg_elements.append(('circle', c.x, c.y, r, is_construction))
                else:
                    # Calculate base angle from StartPoint (most reliable)
                    p_start = geo.StartPoint
                    base_angle = math.atan2(p_start.y - c.y, p_start.x - c.x)

                    # Arc span
                    arc_span = geo.LastParameter - geo.FirstParameter

                    # Direction.z tells us CCW (+1) or CW (-1)
                    is_ccw = geo.Axis.z > 0

                    # For CW arcs, we go backwards from start
                    if is_ccw:
                        start_angle = base_angle
                        end_angle = base_angle + arc_span
                    else:
                        start_angle = base_angle
                        end_angle = base_angle - arc_span

                    svg_elements.append(('arc', c.x, c.y, r, start_angle, end_angle, is_ccw, is_construction))

            elif hasattr(geo, 'StartPoint') and hasattr(geo, 'EndPoint'):
                # Line segment (check after Circle/Arc)
                p1 = geo.StartPoint
                p2 = geo.EndPoint
                x_min = min(x_min, p1.x, p2.x)
                y_min = min(y_min, p1.y, p2.y)
                x_max = max(x_max, p1.x, p2.x)
                y_max = max(y_max, p1.y, p2.y)
                svg_elements.append(('line', p1.x, p1.y, p2.x, p2.y, is_construction))

            elif hasattr(geo, 'X') and hasattr(geo, 'Y'):
                # Point
                x_min = min(x_min, geo.X)
                y_min = min(y_min, geo.Y)
                x_max = max(x_max, geo.X)
                y_max = max(y_max, geo.Y)
                svg_elements.append(('point', geo.X, geo.Y, is_construction))

        except Exception as e:
            if debug:
                print(f"Skipping geometry {i}: {e}")

    if x_min >= float('inf'):
        print("SVG Export failed: No valid geometry")
        return

    # Calculate viewBox and stroke width
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    vb_x = x_min - margin
    vb_y = -(y_max + margin)  # SVG Y is inverted
    vb_w = (x_max - x_min) + 2 * margin
    vb_h = (y_max - y_min) + 2 * margin

    diag = (vb_w**2 + vb_h**2) ** 0.5
    stroke_width = max(diag * 0.001, 0.5)
    point_radius = stroke_width * 1.5

    # Build SVG content
    svg_lines = [
        '<?xml version="1.0"?>',
        f'<svg viewBox="{vb_x:.4f} {vb_y:.4f} {vb_w:.4f} {vb_h:.4f}" xmlns="http://www.w3.org/2000/svg">',
        f'<g transform="scale(1,-1)">',  # Flip Y axis
    ]

    for elem in svg_elements:
        if elem[0] == 'line':
            _, x1, y1, x2, y2, is_const = elem
            color = "#0088FF" if is_const else "#FF0000"
            dash = f'stroke-dasharray="{stroke_width*3:.2f},{stroke_width*2:.2f}"' if is_const else ''
            svg_lines.append(
                f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" {dash} stroke-linecap="round"/>'
            )

        elif elem[0] == 'circle':
            _, cx, cy, r, is_const = elem
            color = "#0088FF" if is_const else "#FF0000"
            dash = f'stroke-dasharray="{stroke_width*3:.2f},{stroke_width*2:.2f}"' if is_const else ''
            svg_lines.append(
                f'<circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" fill="none" {dash}/>'
            )

        elif elem[0] == 'arc':
            _, cx, cy, r, start_angle, end_angle, is_ccw, is_const = elem

            # Calculate start and end points
            x1 = cx + r * math.cos(start_angle)
            y1 = cy + r * math.sin(start_angle)
            x2 = cx + r * math.cos(end_angle)
            y2 = cy + r * math.sin(end_angle)

            # Arc span determines large_arc flag
            arc_span = abs(end_angle - start_angle)
            large_arc = 1 if arc_span > math.pi else 0

            # Sweep: CCW arcs need sweep=1 (after Y-flip appears CCW), CW arcs need sweep=0
            sweep = 1 if is_ccw else 0

            color = "#0088FF" if is_const else "#FF0000"
            dash = f'stroke-dasharray="{stroke_width*3:.2f},{stroke_width*2:.2f}"' if is_const else ''
            svg_lines.append(
                f'<path d="M {x1:.4f},{y1:.4f} A {r:.4f},{r:.4f} 0 {large_arc},{sweep} {x2:.4f},{y2:.4f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" fill="none" {dash} stroke-linecap="round"/>'
            )

        elif elem[0] == 'point':
            _, x, y, is_const = elem
            color = "#0088FF" if is_const else "#FF0000"
            svg_lines.append(
                f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{point_radius:.4f}" fill="{color}"/>'
            )

    svg_lines.append('</g>')
    svg_lines.append('</svg>')

    content = '\n'.join(svg_lines)

    with open(output_svg_path, 'w') as f:
        f.write(content)

    if debug:
        display(SVG(data=content))
        print(f"SVG exported to {output_svg_path}")


if __name__ == '__main__':
    seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')

    output_dir = "sg_filtered_unique"
    os.makedirs(output_dir, exist_ok=True)

    error_log = os.path.join(output_dir, "error_log.txt")
    if os.path.isfile(error_log):
        os.remove(error_log)

    for i, seq in enumerate(tqdm(seq_data['sequences'])):

        if 1200 <= i:
            break

        # if i != 314:
        #     continue

        # if os.path.isfile(f"{output_dir}/{i}.svg"):
        #     print(f"{output_dir}/{i}.svg already exists. Skipping...")
        #     continue

        sketch = datalib.sketch_from_sequence(seq)
        fig = datalib.render_sketch(sketch)
        fig.savefig(f"{output_dir}/{i}_gt.svg")
        plt.close(fig)

        try:
            doc, fc_sketch, history_commands = convert_sketch_to_freecad(sketch)

            # dump paths
            output_py_path=f"{output_dir}/{i}.py"
            output_svg_path=f"{output_dir}/{i}.svg"

            # Dump History to File
            with open(output_py_path, "w") as f:
                f.write("\n".join(history_commands))

            # Dump SVG
            draw_svg(doc, fc_sketch, output_svg_path)

        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            with open(error_log, "a") as f:
                f.write(f"Error processing sequence {i}: {e}\n\n")

        finally:
            FreeCAD.closeDocument(doc.Name)
