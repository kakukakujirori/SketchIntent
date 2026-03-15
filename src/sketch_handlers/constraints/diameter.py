from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class DiameterHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Diameter

    @property
    def name(self) -> str:
        return "Diameter"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        val = self.get_constraint_value(constraint)

        if not self.validate_refs(refs):
            raise ValueError(f"Invalid references for {self.name}: {refs}")

        res1 = primitive_geoID.get(refs[0])
        if res1 and val > 0:
            # Diameter(Circle/Arc, value)
            return [('Diameter', [res1[0], float(val)], None)]

        raise ValueError(f"Invalid references or value for {self.name}: {refs=}, {val=}")

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Diameter':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        r1 = geoID_to_primitive.get((g1, 0))  # Circle/Arc edge
        if not r1:
            raise ValueError(f"Invalid references for {self.name}: {r1}")

        val = fc_constraint.Value
        return self.create_constraint(self.name, (r1,), val)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1
