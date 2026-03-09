"""This module contains some utilities for processing the sketch data for
img2cad.
"""
import math

import numpy as np

import sketchgraphs.data as datalib
from sketchgraphs.data import EntityType
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


def cleanse_sketch(sketch: datalib.Sketch) -> int:
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
