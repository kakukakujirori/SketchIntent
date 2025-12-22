import logging

import numpy as np
import numpy.random as npr
from numpy.linalg import cholesky
from scipy.stats import truncnorm

from img2cad_diffusion.data_utils import parameterize_entity, entity_from_params
from sketchgraphs.data import Sketch
from sketchgraphs.data._entity import Arc, Circle, Line, Point, Entity


def _trunc_normal_entity_noise(entity: Entity, std: float=0.2, max_diff: float=0.1):
    """
    Applies translation noise drawn from a trunctated normal distribution
    to the parameters of the provided entity.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the truncated normal noise distribution (default 0.2)
    max_diff : float, optional
        Maximum allowed magnitude of change in a parameter value (default 0.1)

    Returns
    ---------
    new_ent : EntityObject
    """
    assert std > 0.0 and max_diff > 0.0

    # Parameterize entity
    params = parameterize_entity(entity)  # TODO: check param val legality
    trial = 0
    while True:
        # Draw noise vector
        noise = truncnorm.rvs(a=-max_diff, b=max_diff,
                              scale=std/(2**trial), size=params.size)
        new_params = params + noise
        new_ent = entity_from_params(new_params, entity_id=entity.entityId)

        if isinstance(entity, Arc):  # prevent large arc alterations
            trial += 1
            if new_ent is None:  # try again when arc fails
                continue
            actual_new_params = parameterize_entity(new_ent)
            if (np.abs(actual_new_params - params) <= max_diff).all():
                if (np.abs(entity.center_point - new_ent.center_point) <= max_diff).all():
                    break
        else:
            break
    return new_ent


def noisify_sketch_ents(sketch: Sketch, std: float=0.2, max_diff: float=0.1):
    """Inject noise into the entity parameters of the given sketch.

    The given sketch is assumed to already be normalized. The sketch is modified
    in place.

    Parameters
    ----------
    sketch : Sketch
        The sketch whose entities are to be noisified
    std : float (see `_trunc_normal_entity_noise`)
    max_diff : float (see `_trunc_normal_entity_noise`)

    Returns
    -------
    None
    """
    for ent_key, ent in sketch.entities.items():
        new_ent = _trunc_normal_entity_noise(ent, std, max_diff)
        new_ent.isConstruction = ent.isConstruction
        sketch.entities[ent_key] = new_ent
    return sketch
