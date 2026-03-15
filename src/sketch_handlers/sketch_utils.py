"""This module contains some utilities for processing the sketch data for
img2cad.
"""
import math

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


def rescale_sketch(sketch: datalib.Sketch) -> float:
    """Rescale the sketch such that the long axis of the bounding box is one.

    The entity parameters of the given sketch are modified in place. This
    function should only be called on sketches that are centered; an error is
    raised it the given sketch is not.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be rescaled

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
    factor = max(w, h)
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


def normalize_sketch(sketch: datalib.Sketch) -> float:
    """Helper function that both centers and rescales the given sketch in place.
    """
    center_sketch(sketch)
    scale_factor = rescale_sketch(sketch)
    return scale_factor


def cleanse_sketch_constraints(sketch: datalib.Sketch) -> int:
    """Remove or fix degenerate constraints from the sketch in-place.

    Currently handles:
    - Duplicate references: deduplicate LocalReferenceParameters that share
      the same value (e.g., Coincident(A, A, B, B) → Coincident(A, B)).
    - Self-referential: remove constraints where all LocalReferenceParameters
      point to the same entity/subnode after dedup (e.g., Coincident(A, A) → removed).

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch to cleanse. Modified in-place.

    Returns
    -------
    int
        Number of constraints modified or removed.
    """
    keys_to_remove = []
    n_modified = 0

    for cid, constr in sketch.constraints.items():
        local_refs = [
            p for p in constr.parameters
            if isinstance(p, LocalReferenceParameter)
        ]

        # Skip constraints with 0 or 1 local ref (nothing to check)
        if len(local_refs) <= 1:
            continue

        # Deduplicate LocalReferenceParameters by value
        seen_values = set()
        deduped_params = []
        for p in constr.parameters:
            if isinstance(p, LocalReferenceParameter):
                if p.value in seen_values:
                    continue  # skip duplicate
                seen_values.add(p.value)
            deduped_params.append(p)

        # After dedup: if only 1 unique local ref remains, it's self-referential
        if len(seen_values) <= 1:
            keys_to_remove.append(cid)
            continue

        # If dedup changed anything, update the constraint
        if len(deduped_params) < len(constr.parameters):
            sketch.constraints[cid] = constr._replace(parameters=deduped_params)
            n_modified += 1

    for cid in keys_to_remove:
        del sketch.constraints[cid]

    return len(keys_to_remove) + n_modified


def complete_constraint_values(sketch: datalib.Sketch) -> datalib.Sketch:
    """
    Fill in the void dimensional constraint values by the primitive position.
    """
    import math

    def get_entity_point(entity: datalib.Entity, subnode: str | None = None) -> tuple[float, float] | None:
        """Get the point coordinates for an entity and optional subnode."""
        if entity.type == datalib.EntityType.Point:
            return (entity.x, entity.y)
        elif entity.type == datalib.EntityType.Line:
            if subnode == 'start':
                return entity.start_point.tolist()
            elif subnode == 'end':
                return entity.end_point.tolist()
        elif entity.type == datalib.EntityType.Circle:
            if subnode == 'center':
                return entity.center_point.tolist()
        elif entity.type == datalib.EntityType.Arc:
            if subnode == 'start':
                return entity.start_point.tolist()
            elif subnode == 'end':
                return entity.end_point.tolist()
            elif subnode == 'center':
                return entity.center_point.tolist()
        return None

    def get_direction(constr: datalib.Constraint) -> str:
        direction = ""
        for param in constr.parameters:
            if isinstance(param, EnumParameter) and param.parameterId == 'direction':
                direction = param.value
                break
        if direction not in ("MINIMUM", "HORIZONTAL", "VERTICAL"):
            raise ValueError(f"ERROR: Invalid direction for Distance constraint {constr}")
        return direction

    def get_aligned(constr: datalib.Constraint) -> bool:
        for param in constr.parameters:
            if param.parameterId == 'aligned' and isinstance(param, BooleanParameter):
                return param.value
        return True  # Default to True

    def compute_dist_points(p1: tuple[float, float], p2: tuple[float, float], direction: str) -> float:
        """Calculate Euclidean distance between two points."""
        if direction == "HORIZONTAL":
            return abs(p2[0] - p1[0])
        elif direction == "VERTICAL":
            return abs(p2[1] - p1[1])
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
        return abs(d_centers - circle.radius)

    def compute_dist_line_circle(line: datalib.Line, circle: datalib.Circle | datalib.Arc) -> float:
        d_center_to_line = compute_dist_point_line((circle.xCenter, circle.yCenter), line)
        return abs(d_center_to_line - circle.radius)

    def compute_dist_circle_circle(c1: datalib.Circle | datalib.Arc, c2: datalib.Circle | datalib.Arc) -> float:
        d_centers = math.sqrt((c1.xCenter - c2.xCenter)**2 + (c1.yCenter - c2.yCenter)**2)
        if d_centers < max(c1.radius, c2.radius):
            return abs(abs(c1.radius - c2.radius) - d_centers)
        else:
            return abs(d_centers - (c1.radius + c2.radius))

    def compute_dist_line_line(l1: datalib.Line, l2: datalib.Line) -> float:
        # Check parallelism: |v1 x v2| should be close to 0
        cross_prod = abs(l1.dirX * l2.dirY - l1.dirY * l2.dirX)
        norm1 = math.sqrt(l1.dirX**2 + l1.dirY**2)
        norm2 = math.sqrt(l2.dirX**2 + l2.dirY**2)
        if cross_prod / (norm1 * norm2) > 1e-6:
            raise ValueError(f"Lines are not parallel: {l1}, {l2}")

        # Calculate distance from l1.start to l2
        p1_start = get_entity_point(l1, 'start')
        assert p1_start is not None
        return compute_dist_point_line(p1_start, l2)

    def compute_line_length(line: datalib.Line, direction: str) -> float:
        """Calculate the length of a line entity."""
        start_point = get_entity_point(line, 'start')
        end_point = get_entity_point(line, 'end')
        return compute_dist_points(start_point, end_point, direction)

    def compute_angle_between_lines(line1: datalib.Line, line2: datalib.Line, aligned: bool = True) -> float:
        """Calculate the directed angle [0, 2*pi) between two lines based on parameters."""
        a1 = math.atan2(line1.dirY, line1.dirX)
        a2 = math.atan2(line2.dirY, line2.dirX)
        diff = a2 - a1

        # Cross product for orientation
        cross = line1.dirX * line2.dirY - line1.dirY * line2.dirX
        cross_sign = 1 if cross >= 0 else -1

        if aligned:
            return (cross_sign * diff) % (2 * math.pi)
        else:
            return (math.pi - cross_sign * diff) % (2 * math.pi)

    def assign_value(constr: datalib.Constraint, value: float, param_id: str = 'length') -> datalib.Constraint:
        new_params = []
        value_assigned_times = 0
        for param in constr.parameters:
            if isinstance(param, QuantityParameter):
                assert param_id == param.parameterId, f"ERROR: parameterId mismatch {param_id=}, {param.parameterId=}"
                new_params.append(QuantityParameter(parameterId=param.parameterId, value=value, expression=str(value)))
                value_assigned_times += 1
            else:
                new_params.append(param)

        if value_assigned_times == 0:
            # No value assigned in the original constraint, which sometimes happens
            new_params.append(QuantityParameter(parameterId=param_id, value=value, expression=str(value)))
            value_assigned_times += 1

        if value_assigned_times != 1:
            raise ValueError(f"ERROR: value assigned {value_assigned_times} times for constraint {constr}")

        return constr._replace(parameters=new_params)


    for constr_id, constr in sketch.constraints.items():
        # Extract entity references
        entity_refs = []
        for param in constr.parameters:
            if isinstance(param, LocalReferenceParameter):
                entity_refs.append(param.value)  # (e.g.) 5, 3.center, ...

        # Process each constraint type
        if constr.type == datalib.ConstraintType.Angle:
            assert len(entity_refs) == 2, f"ERROR: Invalid references for Angle constraint {entity_refs}"
            ent1_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            ent2_id = entity_refs[1].split('.')[0] if '.' in entity_refs[1] else entity_refs[1]

            ent1 = sketch.entities.get(ent1_id)
            ent2 = sketch.entities.get(ent2_id)

            if ent1 and ent2 and ent1.type == ent2.type == datalib.EntityType.Line:
                aligned = get_aligned(constr)
                angle = compute_angle_between_lines(ent1, ent2, aligned=aligned)
                sketch.constraints[constr_id] = assign_value(constr, angle, 'angle')
            else:
                raise ValueError(f"ERROR: Invalid references for Angle constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Diameter:
            assert len(entity_refs) == 1, f"ERROR: Invalid references for Diameter constraint {entity_refs}"
            ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            entity = sketch.entities.get(ent_id)

            if entity and entity.type in (datalib.EntityType.Circle, datalib.EntityType.Arc):
                diameter = entity.radius * 2
                sketch.constraints[constr_id] = assign_value(constr, diameter, 'length')
            else:
                raise ValueError(f"ERROR: Invalid references for Diameter constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Distance:

            direction = get_direction(constr)

            if len(entity_refs) == 1:
                # Distance constraint on a single entity (line length)
                ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                entity = sketch.entities.get(ent_id)

                if entity and entity.type == datalib.EntityType.Line:
                    length = compute_line_length(entity, direction)
                    sketch.constraints[constr_id] = assign_value(constr, length, 'length')
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
                    distance = compute_dist_points(p1, p2, direction)
                elif p1:
                    assert direction == 'MINIMUM'
                    if ent2.type == datalib.EntityType.Line:
                        distance = compute_dist_point_line(p1, ent2)
                    elif ent2.type in (datalib.EntityType.Circle, datalib.EntityType.Arc):
                        distance = compute_dist_point_circle(p1, ent2)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")
                elif p2:
                    assert direction == 'MINIMUM'
                    if ent1.type == datalib.EntityType.Line:
                        distance = compute_dist_point_line(p2, ent1)
                    elif ent1.type in (datalib.EntityType.Circle, datalib.EntityType.Arc):
                        distance = compute_dist_point_circle(p2, ent1)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")
                else:
                    assert direction == 'MINIMUM'
                    if ent1.type == datalib.EntityType.Line and ent2.type in (datalib.EntityType.Circle, datalib.EntityType.Arc):
                        distance = compute_dist_line_circle(ent1, ent2)
                    elif ent1.type in (datalib.EntityType.Circle, datalib.EntityType.Arc) and ent2.type == datalib.EntityType.Line:
                        distance = compute_dist_line_circle(ent2, ent1)
                    elif ent1.type in (datalib.EntityType.Circle, datalib.EntityType.Arc) and ent2.type in (datalib.EntityType.Circle, datalib.EntityType.Arc):
                        distance = compute_dist_circle_circle(ent1, ent2)
                    elif ent1.type == datalib.EntityType.Line and ent2.type == datalib.EntityType.Line:
                        distance = compute_dist_line_line(ent1, ent2)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {ent1=}, {ent2=}")

                sketch.constraints[constr_id] = assign_value(constr, distance, 'length')

            else:
                raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Length:
            assert len(entity_refs) == 1, f"ERROR: Invalid references for Length constraint {entity_refs}"
            ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            entity = sketch.entities.get(ent_id)

            direction = get_direction(constr)

            if entity and entity.type == datalib.EntityType.Line:
                length = compute_line_length(entity, direction)
                sketch.constraints[constr_id] = assign_value(constr, length, 'length')
            else:
                raise ValueError(f"ERROR: Invalid references for Length constraint {entity_refs}")

        elif constr.type == datalib.ConstraintType.Radius:
            assert len(entity_refs) >= 1, f"ERROR: Invalid references for Radius constraint {entity_refs}"
            ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
            entity = sketch.entities.get(ent_id)

            if entity and entity.type in (datalib.EntityType.Circle, datalib.EntityType.Arc):
                radius = entity.radius
                sketch.constraints[constr_id] = assign_value(constr, radius, 'length')
            else:
                raise ValueError(f"ERROR: Invalid references for Radius constraint {entity_refs}")

    return sketch
