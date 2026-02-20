from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class SymmetricHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Symmetric

    @property
    def name(self) -> str:
        return "Symmetric"

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
        res3 = primitive_geoID.get(refs[2])
        if not res1 or not res2 or not res3:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs[0]}, {refs[1]}, {refs[2]}")

        g1, p1 = res1
        g2, p2 = res2
        g3, p3 = res3

        # Symmetric(e1, s1, e2, s2, e3, s3) — always 6 args with all subnodes
        # OR Symmetric(e1, s1, e2, s2, e3) — 5 args when axis is an edge
        if p3 == 0:
            # Axis of symmetry is a line (edge) — 5 args
            return [('Symmetric', [g1, p1, g2, p2, g3], None)]
        else:
            # Axis point — 6 args
            return [('Symmetric', [g1, p1, g2, p2, g3, p3], None)]

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Symmetric':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        g3, p3 = fc_constraint.Third, fc_constraint.ThirdPos

        r1 = geoID_to_primitive.get((g1, p1))
        r2 = geoID_to_primitive.get((g2, p2))
        r3 = geoID_to_primitive.get((g3, p3 if p3 > 0 else 0))

        if not r1 or not r2 or not r3:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {g1}, {g2}, {g3}")

        # Keep identifier as 'Symmetric' for roundtrip (routes back to SymmetricHandler.to_freecad)
        return self.create_constraint('Symmetric', (r1, r2, r3))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 3
