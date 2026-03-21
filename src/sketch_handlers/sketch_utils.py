"""This module contains some utilities for processing the sketch data for
img2cad.
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

import sketchgraphs.data as datalib
from sketchgraphs.data import EntityType, BooleanParameter, ConstraintParameter, EnumParameter, QuantityParameter
from sketchgraphs.data import Arc, Circle, Line, Point
from sketchgraphs.data._constraint import LocalReferenceParameter


# Positional parameters
POS_PARAMS = {
    EntityType.Arc: ('xCenter', 'yCenter'),
    EntityType.Circle: ('xCenter', 'yCenter'),
    EntityType.Line: ('pntX', 'pntY'),
    EntityType.Point: ('x', 'y')
}


# Scale parameters
SCALE_PARAMS = {
    EntityType.Arc: ('radius',),
    EntityType.Circle: ('radius',),
    EntityType.Line: ('startParam', 'endParam'),
    EntityType.Point: ()
}


from src.sketch_handlers.registry import PRIMITIVE_HANDLERS

def _get_entity_bbox(ent):
    """Compute the bounding box for the given entity.

    Parameters
    ----------
    ent : Entity
        The entity of interest for bbox computation

    Returns
    -------
    np.array
        Bounding box of the form `[[x0, y0], [x1, y1]]`
    """
    handler_cls = PRIMITIVE_HANDLERS.get(type(ent).__name__)
    if not handler_cls:
        return None

    # Instantiate handler (lightweight) and compute bbox
    return handler_cls().get_bbox(ent)


def get_sketch_bbox(sketch):
    """Compute the bounding box for the sketch.

    The bounding box is only computed based on entities currently supported
    in `_get_entity_bbox` (`Arc`, `Circle`, `Line`, `Point`).
    TODO: raise error if sketch has unsupported entities

    Parameters
    ----------
    sketch : Sketch
        The sketch of interest for bbox computation

    Returns
    -------
    np.array
        Bounding box of the form `[[x0, y0], [x1, y1]]`
    """
    # Get entity bboxes
    bboxes = [_get_entity_bbox(ent) for ent in sketch.entities.values()]
    # Remove any None bboxes
    bboxes = np.array([bbox for bbox in bboxes if bbox is not None])
    if bboxes.size == 0:
        # Return origin as bounding box for empty sketch
        return np.array([[0., 0.], [0., 0.]])
    # Compute overall bbox
    x0, y0 = np.min(bboxes[:,0,:], axis=0)
    x1, y1 = np.max(bboxes[:,1,:], axis=0)
    return np.array([[x0, y0], [x1, y1]])


def center_sketch(sketch):
    """Center the sketch's bounding box over the origin.

    The entity parameters of the given sketch are modified in place.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be centered

    Returns
    -------
    None
    """
    # Get bounding box
    (x0, y0), (x1, y1) = get_sketch_bbox(sketch)
    # Compute current offset from origin
    x_offset = np.mean([x0, x1])
    y_offset = np.mean([y0, y1])
    # Modify the sketch's entities
    for ent in sketch.entities.values():
        pos_params = POS_PARAMS.get(ent.type)
        if pos_params is None:
            # Skip unsupported entities
            continue
        for param_id in pos_params:
            this_offset = x_offset if 'x' in param_id.lower() else y_offset
            curr_val = getattr(ent, param_id)
            setattr(ent, param_id, curr_val - this_offset)


def rescale_sketch(sketch: datalib.Sketch, dst_length: float = 1.0) -> float:
    """Rescale the sketch such that the long axis of the bounding box is dst_length.

    The entity parameters of the given sketch are modified in place. This
    function should only be called on sketches that are centered; an error is
    raised it the given sketch is not.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be rescaled
    dst_length: float
        The target length of the long axis of the bounding box.

    Returns
    -------
    float
        The normalizing scale factor. If the sketch is zero-dim,
        -1 is returned instead.
    """

    # Get bounding box
    (x0, y0), (x1, y1) = get_sketch_bbox(sketch)
    # Ensure sketch is already centered
    if not np.isclose(x0, -x1) or not np.isclose(y0, -y1):
        raise ValueError("sketch must be centered before rescaling")
    # Calculate scale factor
    w = x1 - x0
    h = y1 - y0
    factor = max(w, h) / dst_length
    if factor == 0:
        return -1
    # Modify the sketch's entities
    for ent in sketch.entities.values():
        pos_params = POS_PARAMS.get(ent.type)
        scale_params = SCALE_PARAMS.get(ent.type)
        if pos_params is None:
            # Skip unsupported entities
            continue
        params = pos_params + scale_params  # rescale both types of params
        for param_id in params:
            curr_val = getattr(ent, param_id)
            setattr(ent, param_id, curr_val/factor)
    return factor


def normalize_sketch(sketch: datalib.Sketch, dst_length: float = 1.0) -> float:
    """Helper function that both centers and rescales the given sketch in place.
    """
    center_sketch(sketch)
    scale_factor = rescale_sketch(sketch, dst_length)
    return scale_factor


################################################################


def get_refs(constr: datalib.Constraint) -> list[tuple[int, str | None]]:
    refs = [p.value.split(".") for p in constr.parameters if isinstance(p, LocalReferenceParameter)]  # e.g., 1.start
    refs = [(int(x[0]), x[1] if (len(x) == 2 and len(x[1]) > 0) else None) for x in refs]
    assert all(y in (None, 'start', 'end', 'center') for _, y in refs)
    return sorted(refs, key=lambda x: (x[0], {'start': 1, 'end': 2, 'center': 3}.get(x[1], 0)))


def _resolve_tangent(sketch: datalib.Sketch) -> datalib.Sketch:
    """Resolve Tangent(edge, edge) + Coincident(point, point) pairs to Tangent(edge, point, edge, point).
    FreeCAD judges the former as redundant.
    """
    # Store all Point-Point Coincidents and Edge-Edge Tangents
    point_point_coincident_map = defaultdict(list)
    edge_edge_tangent_map = {}
    for cid, c in sketch.constraints.items():
        if c.constraint_type.name == 'Coincident':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                if pt1 and pt2:  # Point-Point
                    point_point_coincident_map[(geo1, geo2)].append((cid, refs))
        if c.constraint_type.name == 'Tangent':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                if pt1 is None and pt2 is None:  # Edge-Edge
                    edge_edge_tangent_map[(geo1, geo2)] = (cid, refs)

    # Overwrite Tangent(edge, edge) with Tangent(edge, point, edge, point) and drop Coincident(point, point)
    common_keys = point_point_coincident_map.keys() & edge_edge_tangent_map.keys()
    for key in common_keys:
        tangent_cid, ((geo1, _), (geo2, _)) = edge_edge_tangent_map[key]
        for coincident_cid, ((_geo1, pt1), (_geo2, pt2)) in point_point_coincident_map[key]:
            assert geo1 == _geo1 and geo2 == _geo2
            # new params
            new_param = []
            for p in sketch.constraints[tangent_cid].parameters:
                if isinstance(p, LocalReferenceParameter) and (p.value == str(geo1) or p.value.startswith(f"{geo1}.")):
                    p.value = f"{geo1}.{pt1}"
                elif isinstance(p, LocalReferenceParameter) and (p.value == str(geo2) or p.value.startswith(f"{geo2}.")):
                    p.value = f"{geo2}.{pt2}"
                new_param.append(p)
            sketch.constraints[tangent_cid]._replace(parameters=new_param)
            sketch.constraints.pop(coincident_cid)

    return sketch


def _resolve_redundant_coincident(sketch: datalib.Sketch) -> datalib.Sketch:
    """When same (geo1, geo2) pair has both types of Coincident:
        - Point-on-Object (one ref has pt=0, other has pt>0)
        - Point-Point (both refs have pt>0)

    The Point-on-Object is redundant and should be skipped.
    """
    # Parse all Coincident constraints
    coincident_map = defaultdict(list)
    for cid, c in sketch.constraints.items():
        if c.constraint_type.name == 'Coincident':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                coincident_map[(geo1, geo2)].append((cid, (pt1, pt2)))

    # Remove redundant Point-on-Object
    for key, cid_pts_list in coincident_map.items():
        firsts = {pt1 for (cid, (pt1, pt2)) in cid_pts_list if pt1 != 0}
        seconds = {pt2 for (cid, (pt1, pt2)) in cid_pts_list if pt2 != 0}

        redundant_point_on_object = []
        for cid, (pt1, pt2) in cid_pts_list:
            if pt2 == 0 and pt1 in firsts:
                redundant_point_on_object.append(cid)
            if pt1 == 0 and pt2 in seconds:
                redundant_point_on_object.append(cid)

        for cid in redundant_point_on_object:
            del sketch.constraints[cid]

    return sketch


def _resolve_midpoint_mirror_ref_order(sketch: datalib.Sketch) -> datalib.Sketch:
    """Midpoint and Mirror have the following reference ordering: [EntityA, EntityB, SymmetyPoint/Axis]
    However, SketchGraphs format doesn't guarantee this ordering, so we recover it by checking the actual geometry.
    """

    def sort_triplet_refs(
        refs: tuple[tuple[int, str | None], tuple[int, str | None], tuple[int, str | None]],
        entities: dict[str, datalib.Entity],
    ) -> tuple[tuple[int, str | None], tuple[int, str | None], tuple[int, str | None]]:

        (geo1, pt1), (geo2, pt2), (geo3, pt3) = refs
        coord1 = np.array(get_entity_point(entities[str(geo1)], pt1))
        coord2 = np.array(get_entity_point(entities[str(geo2)], pt2))
        coord3 = np.array(get_entity_point(entities[str(geo3)], pt3))

        if (coord1 is not None) and (coord2 is not None) and (coord3 is not None):
            coords = [coord1, coord2, coord3]
            for i in range(3):
                a, b = coords[(i+1) % 3], coords[(i+2) % 3]
                if np.allclose(2*coords[i], a+b):
                    return (refs[(i+1) % 3], refs[(i+2) % 3], refs[i])
                # else:
                #     print(f"{i}: {coords[i]=}, {a=}, {b=}")
            raise ValueError(f"Three points are not equidistant collinear: {coord1=}, {coord2=}, {coord3=}")
        elif coord1 is None:
            assert coord2 is not None and coord3 is not None
            return (refs[1], refs[2], refs[0])
        elif coord2 is None:
            assert coord1 is not None and coord3 is not None
            return (refs[0], refs[2], refs[1])
        elif coord3 is None:
            assert coord1 is not None and coord2 is not None
            return (refs[0], refs[1], refs[2])
        else:
            raise ValueError(f"At least two references must be points: got {refs=}")

    def sort_line_point_refs(
        refs: tuple[tuple[int, str | None], tuple[int, str | None]],
        entities: dict[str, datalib.Entity],
    ) -> tuple[tuple[int, str | None], tuple[int, str | None]]:

        (geo1, pt1), (geo2, pt2) = local_refs
        if isinstance(sketch.entities[str(geo1)], datalib.Line) and pt1 is None:
            assert pt2 or isinstance(sketch.entities[str(geo2)], datalib.Point)
            return (refs[0], refs[1])
        elif isinstance(sketch.entities[str(geo2)], datalib.Line) and pt2 is None:
            assert pt1 or isinstance(sketch.entities[str(geo1)], datalib.Point)
            return (refs[1], refs[0])
        elif isinstance(sketch.entities[str(geo1)], datalib.Arc) or isinstance(sketch.entities[str(geo2)], datalib.Arc):
            raise ValueError("Arc midpoint is not supported for FreeCAD.")
        else:
            raise ValueError(f"Refs must consist of a Line and a Point: got {refs=}")


    for cid, constr in sketch.constraints.items():
        if constr.constraint_type.name not in ('Midpoint', 'Mirror'):
            continue

        # gather ref params
        local_refs = [p.value.split(".") for p in constr.parameters if isinstance(p, LocalReferenceParameter)]
        local_refs = tuple((int(x[0]), str(x[1]) if (len(x) == 2 and x[1] != "") else None) for x in local_refs)

        # sort the refs
        if len(local_refs) == 3:
            # [PointA, PointB, PointSym]
            local_refs_sorted = sort_triplet_refs(local_refs, sketch.entities)
        elif len(local_refs) == 2:
            # [Line, Point]
            local_refs_sorted = sort_line_point_refs(local_refs, sketch.entities)

        local_refs_sorted = [f"{ref[0]}.{ref[1]}" if ref[1] is not None else str(ref[0]) for ref in local_refs_sorted]

        # set the new order
        ref_params = []
        other_params = []
        for p in constr.parameters:
            if isinstance(p, LocalReferenceParameter):
                ref_params.append(p)
            else:
                other_params = []
        new_params = sorted(ref_params, key=lambda p: local_refs_sorted.index(p.value)) + other_params
        sketch.constraints[cid] = constr._replace(parameters=new_params)

    return sketch


def cleanse_sketch_constraints(sketch: datalib.Sketch, resolve_redundancy: bool = True) -> datalib.Sketch:
    """Remove or fix degenerate constraints from the sketch in-place.

    Currently handles:
    - Duplicate references: deduplicate LocalReferenceParameters that share
      the same value (e.g., Coincident(A, A, B, B) → Coincident(A, B)).
    - Self-referential: remove constraints where all LocalReferenceParameters
      point to the same entity/subnode after dedup (e.g., Coincident(A, A), Tangent(A, 1, A, 1) → removed).
    - Sort Midpoint and Mirror references so that the last entity is the symmetry point or axis.

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch to cleanse. Modified in-place.

    Returns
    -------
    sketch : datalib.Sketch
        The sketch with degenerate constraints removed.
    """
    keys_to_remove = []

    for cid, constr in sketch.constraints.items():
        # Deduplicate LocalReferenceParameters by value
        local_refs = []
        seen_refs = set()
        deduped_params = []
        for p in constr.parameters:
            if isinstance(p, LocalReferenceParameter):
                local_refs.append(p.value)
                if p.value in seen_refs:
                    continue  # skip duplicate
                seen_refs.add(p.value)
            deduped_params.append(p)

        # Skip constraints with 0 or 1 local ref (nothing to check)
        if len(local_refs) <= 1:
            continue

        # After dedup: if only 1 unique local ref remains, it's self-referential
        if len(seen_refs) <= 1:
            keys_to_remove.append(cid)
            continue

        # If dedup changed anything, update the constraint
        if len(deduped_params) < len(constr.parameters):
            sketch.constraints[cid] = constr._replace(parameters=deduped_params)

    for cid in keys_to_remove:
        del sketch.constraints[cid]

    # resolve midpoint/mirror reference ordering
    sketch = _resolve_midpoint_mirror_ref_order(sketch)

    if resolve_redundancy:
        sketch = _resolve_tangent(sketch)
        sketch = _resolve_redundant_coincident(sketch)

    return sketch


################################################################


@dataclass
class SketchProperties:
    direction: str = ""
    aligned: bool = True


def get_properties(constr: datalib.Constraint) -> SketchProperties:
    props = SketchProperties()

    for param in constr.parameters:
        # direction
        if isinstance(param, EnumParameter) and param.parameterId == 'direction':
            props.direction = param.value
        # aligned
        elif isinstance(param, BooleanParameter) and param.parameterId == 'aligned':
            props.aligned = param.value

    if props.direction not in ("", "MINIMUM", "HORIZONTAL", "VERTICAL"):
        raise ValueError(f"ERROR: Invalid direction for Distance constraint {constr}")

    return props


def get_entity_point(entity: datalib.Entity, subnode: str | None = None) -> tuple[float, float] | None:
    """Get the point coordinates for an entity and optional subnode."""
    if isinstance(entity, datalib.Point):
        return (entity.x, entity.y)
    elif isinstance(entity, datalib.Line):
        if subnode == 'start':
            return entity.start_point.tolist()
        elif subnode == 'end':
            return entity.end_point.tolist()
    elif isinstance(entity, datalib.Circle):
        if subnode == 'center':
            return entity.center_point.tolist()
    elif isinstance(entity, datalib.Arc):
        if subnode == 'start':
            return entity.start_point.tolist()
        elif subnode == 'end':
            return entity.end_point.tolist()
        elif subnode == 'center':
            return entity.center_point.tolist()
    return None


def complete_constraint_values(sketch: datalib.Sketch, expr_watermark: str = "VALUE_ASSIGNED:") -> datalib.Sketch:
    """
    Fill in the void dimensional constraint values by the primitive position.

    Dimensional constraints in the original SketchGraphs dataset contain contraint values in param.expression.
    Unfortunately, this is a plain text and param.value is left empty.
    This function assigns a value to param.value based on the primitive positions.

    NOTE: After the value is assigned, param.expression is updated in the format of "<expr_watermark>: <value>".
    NOTE: The value match between the calculated param.value and existing param.expression is not verified in this function,
          which is instead checked by `tests/test_compute_constraint_values.py`.
    """

    def compute_dist_points(p1: tuple[float, float], p2: tuple[float, float], direction: str) -> float:
        """Calculate Euclidean distance between two points."""
        if direction == "HORIZONTAL":
            return p2[0] - p1[0]
        elif direction == "VERTICAL":
            return p2[1] - p1[1]
        elif direction == "MINIMUM":
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        else:
            raise ValueError(f"ERROR: Invalid direction for Distance constraint {direction}")

    def compute_dist_point_line(p: tuple[float, float], line: datalib.Line) -> float:
        dx = p[0] - line.pntX
        dy = p[1] - line.pntY
        area = abs(dx * line.dirY - dy * line.dirX)
        norm = math.sqrt(line.dirX**2 + line.dirY**2)
        return area / norm if norm > 0 else 0.0

    def compute_dist_point_circle(p: tuple[float, float], circle: datalib.Circle | datalib.Arc) -> float:
        d_centers = math.sqrt((p[0] - circle.xCenter)**2 + (p[1] - circle.yCenter)**2)
        return abs(d_centers - circle.radius)  # For Point-Circle, FreeCAD expects non-negative values.

    def compute_dist_line_circle(line: datalib.Line, circle: datalib.Circle | datalib.Arc) -> float:
        d_center_to_line = compute_dist_point_line((circle.xCenter, circle.yCenter), line)
        return d_center_to_line - circle.radius

    def compute_dist_circle_circle(c1: datalib.Circle | datalib.Arc, c2: datalib.Circle | datalib.Arc) -> float:
        d_centers = math.sqrt((c1.xCenter - c2.xCenter)**2 + (c1.yCenter - c2.yCenter)**2)
        if d_centers < abs(c1.radius - c2.radius):
            # whole circle is included by the other (positive)
            return abs(c1.radius - c2.radius) - d_centers
        else:
            # circles are disjoint (positive) or partially overlapping (negative)
            return d_centers - (c1.radius + c2.radius)

    def compute_dist_line_line(l1: datalib.Line, l2: datalib.Line) -> float:
        # Check parallelism: |v1 x v2| should be close to 0
        cross_prod = abs(l1.dirX * l2.dirY - l1.dirY * l2.dirX)
        norm1 = math.sqrt(l1.dirX**2 + l1.dirY**2)
        norm2 = math.sqrt(l2.dirX**2 + l2.dirY**2)
        if cross_prod / (norm1 * norm2) > 1e-3:
            raise ValueError(f"Lines are not parallel: {l1}, {l2}")

        # Calculate distance from l1.start to l2
        p1_start = get_entity_point(l1, 'start')
        assert p1_start is not None
        return compute_dist_point_line(p1_start, l2)

    def compute_line_length(line: datalib.Line, direction: str) -> float:
        """Calculate the length of a line entity."""
        start_point = get_entity_point(line, 'start')
        end_point = get_entity_point(line, 'end')
        assert start_point is not None and end_point is not None
        return compute_dist_points(start_point, end_point, direction)

    def compute_angle_between_lines(line1: datalib.Line, line2: datalib.Line) -> float:
        """Calculate the directed angle from line1 to line2 based on parameters."""
        a1 = math.atan2(line1.dirY, line1.dirX)
        a2 = math.atan2(line2.dirY, line2.dirX)
        diff = a2 - a1
        return diff

    def assign_value(
        constr: datalib.Constraint,
        value: float,
        param_id: Literal['length', 'angle'],
        expr_watermark: str,
    ) -> datalib.Constraint:
        new_params = []
        value_assigned_times = 0
        for param in constr.parameters:
            if isinstance(param, QuantityParameter):
                assert param_id == param.parameterId, f"ERROR: parameterId mismatch {param_id=}, {param.parameterId=}"
                new_params.append(QuantityParameter(parameterId=param.parameterId, value=value, expression=expr_watermark+str(value)))
                value_assigned_times += 1
            else:
                new_params.append(param)

        if value_assigned_times == 0:
            # No value assigned in the original constraint, which sometimes happens
            new_params.append(QuantityParameter(parameterId=param_id, value=value, expression=expr_watermark+str(value)))
            value_assigned_times += 1

        if value_assigned_times != 1:
            raise ValueError(f"ERROR: value assigned {value_assigned_times} times for constraint {constr}")

        return constr._replace(parameters=new_params)

    def assign_boolean(
        constr: datalib.Constraint,
        flag: bool,
        name: Literal['aligned', 'clockwise'],
    ) -> datalib.Constraint:
        new_params = []
        boolean_assigned_times = 0
        for param in constr.parameters:
            if isinstance(param, BooleanParameter) and param.parameterId == name:
                new_params.append(BooleanParameter(parameterId=name, value=flag))
                boolean_assigned_times += 1
            else:
                new_params.append(param)

        if boolean_assigned_times == 0:
            # No value assigned in the original constraint, which sometimes happens
            new_params.append(BooleanParameter(parameterId=name, value=flag))
            boolean_assigned_times += 1

        if boolean_assigned_times != 1:
            raise ValueError(f"ERROR: value assigned {boolean_assigned_times} times for constraint {constr}")

        return constr._replace(parameters=new_params)


    for constr_id, constr in sketch.constraints.items():
        # Extract entity references
        entity_refs = []
        for param in constr.parameters:
            if isinstance(param, LocalReferenceParameter):
                entity_refs.append(param.value)  # (e.g.) 5, 3.center, ...

        # properties
        props = get_properties(constr)

        # Process each constraint type
        if constr.type == datalib.ConstraintType.Angle:
            # NOTE: Since the original SketchGraphs angle data is unclear in how it was computed, we instead use angles calculated on the fly.
            #       Call src.sketch_handlers.sketch_utils.complete_constraint_values() first to make that change.
            #       - Value is already set in radians (used as-is by FreeCAD)
            #       - Aligned is always True
            #       - Clockwise is neglected
            assert len(entity_refs) == 2, f"ERROR: Invalid references for Angle constraint {entity_refs}"
            ent1_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            ent2_id = entity_refs[1].split('.')[0] if '.' in entity_refs[1] else entity_refs[1]

            ent1 = sketch.entities.get(ent1_id)
            ent2 = sketch.entities.get(ent2_id)

            if isinstance(ent1, datalib.Line) and isinstance(ent2, datalib.Line):
                angle = compute_angle_between_lines(ent1, ent2)
                new_constr = assign_value(constr, angle, 'angle', expr_watermark)
                new_constr = assign_boolean(new_constr, True, 'aligned')
                sketch.constraints[constr_id] = new_constr
            else:
                raise ValueError(f"ERROR: Invalid references for Angle constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Diameter:
            assert len(entity_refs) == 1, f"ERROR: Invalid references for Diameter constraint {entity_refs}"
            ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            entity = sketch.entities.get(ent_id)

            if isinstance(entity, (datalib.Circle, datalib.Arc)):
                diameter = entity.radius * 2
                sketch.constraints[constr_id] = assign_value(constr, diameter, 'length', expr_watermark)
            else:
                raise ValueError(f"ERROR: Invalid references for Diameter constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Distance:

            if len(entity_refs) == 1:
                # Distance constraint on a single entity (line length)
                ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                entity = sketch.entities.get(ent_id)

                if isinstance(entity, datalib.Line):
                    length = compute_line_length(entity, props.direction)
                    sketch.constraints[constr_id] = assign_value(constr, length, 'length', expr_watermark)
                else:
                    raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")

            elif len(entity_refs) == 2:
                # Distance between two entities
                ent1_ref = entity_refs[0]
                ent2_ref = entity_refs[1]

                # Parse entity IDs and subnodes
                if '.' in ent1_ref:
                    ent1_id, sub1 = ent1_ref.split('.', 1)
                else:
                    ent1_id, sub1 = ent1_ref, None

                if '.' in ent2_ref:
                    ent2_id, sub2 = ent2_ref.split('.', 1)
                else:
                    ent2_id, sub2 = ent2_ref, None

                ent1 = sketch.entities.get(ent1_id)
                ent2 = sketch.entities.get(ent2_id)
                assert ent1 and ent2, f"ERROR: Invalid references for Distance constraint {entity_refs}"

                p1 = get_entity_point(ent1, sub1)
                p2 = get_entity_point(ent2, sub2)
                if p1 and p2:
                    distance = compute_dist_points(p1, p2, props.direction)
                elif p1:
                    assert props.direction == 'MINIMUM'
                    if isinstance(ent2, datalib.Line):
                        distance = compute_dist_point_line(p1, ent2)
                    elif isinstance(ent2, (datalib.Circle, datalib.Arc)):
                        distance = compute_dist_point_circle(p1, ent2)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")
                elif p2:
                    assert props.direction == 'MINIMUM'
                    if isinstance(ent1, datalib.Line):
                        distance = compute_dist_point_line(p2, ent1)
                    elif isinstance(ent1, (datalib.Circle, datalib.Arc)):
                        distance = compute_dist_point_circle(p2, ent1)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")
                else:
                    assert props.direction == 'MINIMUM'
                    if isinstance(ent1, datalib.Line) and isinstance(ent2, (datalib.Circle, datalib.Arc)):
                        distance = compute_dist_line_circle(ent1, ent2)
                    elif isinstance(ent1, (datalib.Circle, datalib.Arc)) and isinstance(ent2, datalib.Line):
                        distance = compute_dist_line_circle(ent2, ent1)
                    elif isinstance(ent1, (datalib.Circle, datalib.Arc)) and isinstance(ent2, (datalib.Circle, datalib.Arc)):
                        distance = compute_dist_circle_circle(ent1, ent2)
                    elif isinstance(ent1, datalib.Line) and isinstance(ent2, datalib.Line):
                        distance = compute_dist_line_line(ent1, ent2)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {ent1=}, {ent2=}")

                sketch.constraints[constr_id] = assign_value(constr, distance, 'length', expr_watermark)

            else:
                raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Length:
            assert len(entity_refs) == 1, f"ERROR: Invalid references for Length constraint {entity_refs}"
            ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            entity = sketch.entities.get(ent_id)

            if isinstance(entity, datalib.Line):
                length = compute_line_length(entity, props.direction)
                sketch.constraints[constr_id] = assign_value(constr, length, 'length', expr_watermark)
            elif isinstance(entity, datalib.Arc):
                # Arc length = radius * angle (in radians)
                angle_diff = abs(entity.endParam - entity.startParam)
                # Handle full circle case
                if angle_diff > math.pi * 2:
                    angle_diff = angle_diff % (math.pi * 2)
                length = entity.radius * angle_diff
                sketch.constraints[constr_id] = assign_value(constr, length, 'length', expr_watermark)
            else:
                raise ValueError(f"ERROR: Invalid references for Length constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Radius:
            assert len(entity_refs) >= 1, f"ERROR: Invalid references for Radius constraint {entity_refs}"
            ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            entity = sketch.entities.get(ent_id)

            if isinstance(entity, (datalib.Circle, datalib.Arc)):
                radius = entity.radius
                sketch.constraints[constr_id] = assign_value(constr, radius, 'length', expr_watermark)
            else:
                raise ValueError(f"ERROR: Invalid references for Radius constraint {entity_refs}")

    return sketch

################

def cleanse_sketch_primitives(sketch: datalib.Sketch) -> datalib.Sketch:
    """Force startParam < endParam for datalib.Line, which makes the direction vector definition aligned with FreeCAD.
    This is beneficial for Angle constraints.
    """
    for pid, p in sketch.entities.items():
        if isinstance(p, datalib.Line):
            if p.startParam > p.endParam:
                p.dirX, p.dirY = -p.dirX, -p.dirY
                p.startParam, p.endParam = -p.startParam, -p.endParam
            sketch.entities[pid] = p
    return sketch