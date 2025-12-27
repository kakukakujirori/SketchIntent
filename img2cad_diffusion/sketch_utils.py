"""This module contains some utilities for processing the sketch data for
img2cad.
"""
import math

import numpy as np
import matplotlib.pyplot as plt

import sketchgraphs.data as datalib
from sketchgraphs.data import EntityType
from sketchgraphs.data import Arc, Circle, Line, Point

from .tokens import PrimitiveParamToken, NUM_PARAMS

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
    def get_max_circle_bbox(ent):
        """Get maximum bounding box for an arc or circle."""
        if type(ent) not in [Arc, Circle]:
            raise ValueError("Only applicable to Arc and Circle")
        return np.array([[ent.xCenter-ent.radius, ent.yCenter-ent.radius],
                         [ent.xCenter+ent.radius, ent.yCenter+ent.radius]])

    def get_relative_quadrant(point, center):
        """Determine relative quadrant for the given point (array-like [x, y]).
        """
        x = point[0] - center[0]
        y = point[1] - center[1]
        if x >= 0:
            if y >= 0:
                return 1
            else:
                return 4
        else:
            if y >= 0:
                return 2
            else:
                return 3

    if isinstance(ent, Arc):
        x_start, y_start = ent.start_point
        x_end, y_end = ent.end_point
        (x0, y0), (x1, y1) = get_max_circle_bbox(ent)
        # Get start and end relative quadrants
        start_quadrant = get_relative_quadrant(
            ent.start_point, ent.center_point)
        end_quadrant = get_relative_quadrant(ent.end_point, ent.center_point)
        if ent.clockwise:
            start_quadrant, end_quadrant = end_quadrant, start_quadrant
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        # Get included quadrants
        if start_quadrant < end_quadrant:
            quadrants = list(range(start_quadrant, end_quadrant+1))
        elif start_quadrant > end_quadrant:
            quadrants = list(range(start_quadrant, 5)) + list(
                range(1, end_quadrant+1))
        else:
            # Handle case when start & end in same quadrant
            start_ahead = False
            if start_quadrant in [1, 2]:
                if x_start <= x_end:
                    start_ahead = True
            if start_quadrant in [3, 4]:
                if x_start >= x_end:
                    start_ahead = True
            if start_ahead:
                quadrants = list(range(start_quadrant, 5)) + list(
                    range(1, end_quadrant+1)
                )
            else:
                quadrants = [start_quadrant]
        def has_ordered_quadrants(q1, q2):
            if not (q1 in quadrants and q2 in quadrants):
                return False
            idx1 = quadrants.index(q1)
            idx2 = quadrants.index(q2)
            if idx2 > idx1:
                return True
            return False
        # Replace each coordinate of max bbox as needed
        if not has_ordered_quadrants(4, 1):
            x1 = max(x_start, x_end)
        if not has_ordered_quadrants(1, 2):
            y1 = max(y_start, y_end)
        if not has_ordered_quadrants(2, 3):
            x0 = min(x_start, x_end)
        if not has_ordered_quadrants(3, 4):
            y0 = min(y_start, y_end)

        # Include center point in bbox for tokenization purposes
        # (center point is used in parameterize_entity and must be normalized)
        cx, cy = ent.center_point
        x0 = min(x0, cx)
        y0 = min(y0, cy)
        x1 = max(x1, cx)
        y1 = max(y1, cy)

        return np.array([[x0, y0], [x1, y1]])

    if isinstance(ent, Circle):
        return get_max_circle_bbox(ent)
    if isinstance(ent, Line):
        start_x, start_y = ent.start_point
        end_x, end_y = ent.end_point
        return np.array([[min(start_x, end_x), min(start_y, end_y)],
                         [max(start_x, end_x), max(start_y, end_y)]])
    if isinstance(ent, Point):
        return np.array([[ent.x, ent.y],
                         [ent.x, ent.y]])
    # Return None for unsupported entities
    return None


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


def _parameterize_arc(arc: datalib.Arc | None) -> tuple[np.ndarray | None, tuple[PrimitiveParamToken, ...]]:
    """Extract parameterization for Arc.

    Returns [center_point, start_point, end_point] to match SketchGraphs
    Sequence representation where Arc subnodes are ordered as
    [SN_Center, SN_Start, SN_End].

    If arc is None, returns (None, param_tokens) for offset derivation.
    """
    param_tokens = (
        PrimitiveParamToken.CenterX, PrimitiveParamToken.CenterY,
        PrimitiveParamToken.StartX, PrimitiveParamToken.StartY,
        PrimitiveParamToken.EndX, PrimitiveParamToken.EndY,
    )
    if arc is None:
        return None, param_tokens
    values = np.concatenate([arc.center_point, arc.start_point, arc.end_point])
    return values, param_tokens

def _parameterize_circle(circle: datalib.Circle | None) -> tuple[np.ndarray | None, tuple[PrimitiveParamToken, ...]]:
    """Extract parameterization for Circle.

    If circle is None, returns (None, param_tokens) for offset derivation.
    """
    param_tokens = (PrimitiveParamToken.CenterX, PrimitiveParamToken.CenterY, PrimitiveParamToken.Radius)
    if circle is None:
        return None, param_tokens
    values = np.append(circle.center_point, circle.radius)
    return values, param_tokens

def _parameterize_line(line: datalib.Line | None) -> tuple[np.ndarray | None, tuple[PrimitiveParamToken, ...]]:
    """Extract parameterization for Line.

    If line is None, returns (None, param_tokens) for offset derivation.
    """
    param_tokens = (PrimitiveParamToken.StartX, PrimitiveParamToken.StartY, PrimitiveParamToken.EndX, PrimitiveParamToken.EndY)
    if line is None:
        return None, param_tokens
    values = np.concatenate([line.start_point, line.end_point])
    return values, param_tokens

def _parameterize_point(point: datalib.Point | None) -> tuple[np.ndarray | None, tuple[PrimitiveParamToken, ...]]:
    """Extract parameterization for Point.

    If point is None, returns (None, param_tokens) for offset derivation.
    """
    param_tokens = (PrimitiveParamToken.StartX, PrimitiveParamToken.StartY)
    if point is None:
        return None, param_tokens
    values = np.array([point.x, point.y])
    return values, param_tokens


def parameterize_entity(ent: datalib.Entity) -> tuple[np.ndarray, tuple[PrimitiveParamToken, ...]]:
    """Extract parameterization for an entity instance.

    Parameters
    ----------
    ent : Entity instance (Arc, Circle, Line, Point)

    Returns
    -------
    tuple[np.ndarray, tuple[PrimitiveParamToken, ...]]
        (values, param_tokens)
    """
    # Mapping from entity type to parameterization function
    _PARAM_BY_TYPE = {
        Arc: _parameterize_arc,
        Circle: _parameterize_circle,
        Line: _parameterize_line,
        Point: _parameterize_point,
    }

    # Instance passed - return values + param_tokens
    param_fn = _PARAM_BY_TYPE.get(type(ent))
    if param_fn is None:
        raise ValueError(f"Unsupported entity type: {type(ent)}")
    return param_fn(ent)


def _arc_from_params(params, entity_id: str | None = None, clockwise: bool | None = None):
    """Instantiate an arc from the given parameterization.

    Parameters are expected as [center_point, start_point, end_point] to match
    SketchGraphs Sequence representation [SN_Center, SN_Start, SN_End].

    Parameters
    ----------
    params : list
        Arc parameters [cx, cy, sx, sy, ex, ey]
    entity_id : str, optional
        Entity ID
    clockwise : bool, optional
        If provided, use this value for clockwise. Otherwise infer from geometry.
    """
    center = params[:2]
    start = params[2:4]
    end = params[4:]

    # Calculate radius from center to start point
    radius = ((center[0] - start[0])**2 + (center[1] - start[1])**2)**.5
    if radius < 1e-9:
        radius = 1e-9

    # Compute angles from center to start and end
    v_start = [start[0] - center[0], start[1] - center[1]]
    v_end = [end[0] - center[0], end[1] - center[1]]
    start_angle = math.atan2(v_start[1], v_start[0])
    end_angle = math.atan2(v_end[1], v_end[0])

    # Determine clockwise if not provided
    if clockwise is None:
        # Use shorter path heuristic (default to CCW for 180° arcs)
        ccw_angle = end_angle - start_angle
        if ccw_angle <= 0:
            ccw_angle += 2 * math.pi
        clockwise = ccw_angle > math.pi

    # Compute startParam and endParam based on clockwise
    if clockwise:
        start_param = -start_angle
        end_param = -end_angle
        if start_param > end_param:
            start_param -= 2 * math.pi
    else:
        # CCW arc: ensure end_angle > start_angle
        start_param = start_angle
        end_param = end_angle
        # Handle wrap-around: if end < start, add 2π to end
        # This handles cases like start_angle=π, end_angle=-π/2
        # which should become start_param=π, end_param=3π/2
        if end_param <= start_param:
            end_param += 2 * math.pi

    return Arc(
        entityId=entity_id,
        xCenter=center[0],
        yCenter=center[1],
        radius=radius,
        clockwise=clockwise,
        xDir=1.0,
        yDir=0.0,
        startParam=start_param,
        endParam=end_param
    )

def _circle_from_params(params, entity_id=None):
    """Instantiate a circle from the given parameterization."""
    # Ensure radius is positive (noise can occasionally make it negative)
    radius = abs(params[2])
    if radius < 1e-9:
        radius = 1e-9  # Minimum radius to avoid degenerate circles
    return Circle(entity_id, xCenter=params[0], yCenter=params[1], radius=radius)

def _line_from_params(params, entity_id=None):
    """Instantiate a line from the given parameterization."""
    info = {
        'id': entity_id,
        'startPoint': params[:2],
        'endPoint': params[2:]
    }
    return Line.from_info(info)

def _point_from_params(params, entity_id=None):
    """Instantiate a point from the given parameterization"""
    return Point(entity_id, x=params[0], y=params[1])


def entity_from_params(params, entity_id: str | None = None, clockwise: bool | None = None):
    """Instantiate an entity from the given parameterization.

    The length of params uniquely determines the target entity type.

    Parameters
    ----------
    params : np.array
        The entity's parameterization
    entity_id : str, optional
        Optional string specifying the id of the entity to create.
    clockwise : bool, optional
        For Arc entities, the clockwise flag. Ignored for other entity types.

    Returns
    -------
    Entity
        The entity instance corresponding to the input parameterization
    """
    if len(params) == NUM_PARAMS[Arc]:
        return _arc_from_params(params, entity_id, clockwise=clockwise)
    elif len(params) == NUM_PARAMS[Circle]:
        return _circle_from_params(params, entity_id)
    elif len(params) == NUM_PARAMS[Line]:
        return _line_from_params(params, entity_id)
    elif len(params) == NUM_PARAMS[Point]:
        return _point_from_params(params, entity_id)
    else:
        raise ValueError(f"Unsupported number of parameters: {len(params)}")


# Minimum and maximum parameter values following normalization
MIN_VAL = -0.5
MAX_VAL = 0.5


def quantize_params(params: np.ndarray, n_bins):
    """Convert params in [MIN_VAL, MAX_VAL] to discrete values in [0, n_bins-1].

    Parameters
    ----------
    params : np.array
        The parameters to be quantized
    n_bins : int
        The number of bins

    Returns
    -------
    np.array
        The quantized parameter bins
    """
    min_val, max_val = MIN_VAL, MAX_VAL
    params = np.around(params, decimals=10)
    if (params < min_val).any() or (params > max_val).any():
        raise ValueError("Parameters must be in [%f, %f]. Got [%f, %f]."
        % (min_val, max_val, np.min(params), np.max(params)))
    params_quantized = (params - min_val) / (max_val - min_val) * n_bins
    params_quantized = params_quantized.astype('int32')
    # Handle max_val edge case
    params_quantized[params_quantized == n_bins] -= 1
    return params_quantized


def dequantize_params(params, n_bins):
    """Convert quantized parameters to floats in range [MIN_VAL, MAX_VAL].

    Parameters
    ----------
    params : array-like
        The parameters to be dequantized
    n_bins : int
        The number of bins

    Returns
    -------
    np.array
        The dequantized parameter values
    """
    if isinstance(params, list):
        params = np.array(params)
    min_val, max_val = MIN_VAL, MAX_VAL
    if ((params < 0).any() or (params > (n_bins-1)).any()
        or not np.issubdtype(params.dtype, np.integer)):
        raise ValueError("Invalid quantized params")
    params = params.astype('float32') + 0.5  # center of each bin
    params = params / n_bins * (max_val - min_val) + min_val
    return params
