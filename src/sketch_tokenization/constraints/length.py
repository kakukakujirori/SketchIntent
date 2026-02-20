from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class LengthHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Length

    @property
    def name(self) -> str:
        return "Length"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), (ConstraintToken.Length,)

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        val = self.get_constraint_value(constraint)

        # Length is mapped to FreeCAD's Distance on a single edge
        if len(refs) == 1:
            res1 = primitive_geoID.get(refs[0])
            if res1:
                return [('Distance', [res1[0], float(val)], None)]
        else:
            raise ValueError(f"Invalid number of refs for {self.name}: {len(refs)}")

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        # Length tokens don't come directly from a "Length" FC constraint type.
        # They come from "Distance" on a single entity (handled by DistanceHandler).
        raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1
