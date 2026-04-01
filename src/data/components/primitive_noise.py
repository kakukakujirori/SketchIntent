import copy
import numpy as np
from scipy.stats import truncnorm

import sketchgraphs.data as datalib


class PrimitiveNoise:
    def __init__(self, std: float = 0.15, max_diff: float = 0.15):
        """
        std : float
            Standard deviation of the truncated normal noise distribution.
        max_diff : float
            Maximum allowed magnitude of change in a parameter value.
        """
        self.std = std
        self.max_diff = max_diff
        assert std >= 0.0 and self.max_diff >= 0.0

    def _trunc_normal_entity_noise(self, entity: datalib.Entity):
        """Applies truncated normal noise to entity parameters.

        Parameters
        ----------
        entity : Entity
            The entity to apply noise to.

        Returns
        -------
        Entity
            New entity with noisy parameters.
        """
        assert self.std > 0.0

        params, _ = sketch_utils.parameterize_entity(entity)
        trial = 0
        while True:
            noise = truncnorm.rvs(a=-self.max_diff, b=self.max_diff, scale=self.std / (2 ** trial), size=params.size)
            new_params = params + noise

            new_ent = sketch_utils.entity_from_params(new_params, entity_id=entity.entityId)

            if isinstance(entity, datalib.Arc):
                trial += 1
                if new_ent is None:
                    continue
                actual_new_params, _ = sketch_utils.parameterize_entity(new_ent)
                if (np.abs(actual_new_params - params) <= self.max_diff).all():
                    if (np.abs(entity.center_point - new_ent.center_point) <= self.max_diff).all():
                        break
            else:
                break
        return new_ent

    def _noisify_sketch_ents(self, sketch: datalib.Sketch):
        """Inject noise into entity parameters of the given sketch (in-place)."""
        if self.std <= 0.0:
            return sketch

        for ent_key, ent in sketch.entities.items():
            new_ent = self._trunc_normal_entity_noise(ent)
            new_ent.isConstruction = ent.isConstruction
            sketch.entities[ent_key] = new_ent
        return sketch

    def apply_primitive_noise(self, sketch: datalib.Sketch) -> datalib.Sketch:
        """Apply noise to sketch entities, returning a new noisy copy."""
        if self.std <= 0.0:
            return sketch

        noise_sketch = copy.deepcopy(sketch)
        try:
            noise_sketch = self._noisify_sketch_ents(noise_sketch)
        except:
            noise_sketch = sketch
        return noise_sketch