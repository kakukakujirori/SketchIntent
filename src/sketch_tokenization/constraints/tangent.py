from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class TangentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Tangent

    @property
    def name(self) -> str:
        return "Tangent"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of refs for {self.name}: {len(refs)}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs[0]}, {refs[1]}")

        g1, p1 = res1
        g2, p2 = res2

        val = self.get_constraint_value(constraint)
        datum = val if val is not None and val != 0.0 else None
        if p1 > 0 or p2 > 0:
            # Tangent with point specification: Tangent(e1, s1, e2, s2)
            if datum is not None:
                return [('Tangent', [g1, p1, g2, p2], datum)]
            return [('Tangent', [g1, p1, g2, p2], None)]
        else:
            # Edge-Edge tangent: Tangent(e1, e2)
            if datum is not None:
                return [('Tangent', [g1, g2], datum)]
            return [('Tangent', [g1, g2], None)]

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Tangent':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos

        if p1 > 0 or p2 > 0:
            # Point tangent
            r1 = geoID_to_primitive.get((g1, p1))
            r2 = geoID_to_primitive.get((g2, p2))
        else:
            # Edge tangent
            r1 = geoID_to_primitive.get((g1, 0))
            r2 = geoID_to_primitive.get((g2, 0))

        if not r1 or not r2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {g1}, {g2}")

        val = fc_constraint.Value
        return self.create_constraint(self.name, (r1, r2), val if val != 0.0 else None)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
