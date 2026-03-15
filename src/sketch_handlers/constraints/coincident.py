from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class CoincidentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Coincident

    @property
    def name(self) -> str:
        return "Coincident"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid references for {self.name}: {refs}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid references for {self.name}: {refs}")

        g1, p1 = res1
        g2, p2 = res2

        if p1 == 0 and p2 == 0:
            # Edge-Edge: not valid for Coincident
            raise ValueError(f"Edge-Edge is not valid for {self.name}: {refs} ({res1=}, {res2=})")
        elif p1 == 0:
            # PointOnObject: point (g2,p2) on edge g1
            return [('PointOnObject', [g2, p2, g1], None)]
        elif p2 == 0:
            # PointOnObject: point (g1,p1) on edge g2
            return [('PointOnObject', [g1, p1, g2], None)]
        else:
            # Point-Point
            return [('Coincident', [g1, p1, g2, p2], None)]

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        ctype = fc_constraint.Type
        if ctype not in ('Coincident', 'PointOnObject'):
            raise ValueError(f"Invalid constraint type for {self.name}: {ctype}")

        refs = ()
        if ctype == 'Coincident':
            g1, p1 = fc_constraint.First, fc_constraint.FirstPos
            g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
            r1 = geoID_to_primitive.get((g1, p1))
            r2 = geoID_to_primitive.get((g2, p2))
            if r1 and r2:
                refs = (r1, r2)
        elif ctype == 'PointOnObject':
            g1, p1 = fc_constraint.First, fc_constraint.FirstPos
            g2 = fc_constraint.Second
            r1 = geoID_to_primitive.get((g1, p1))  # Point
            r2 = geoID_to_primitive.get((g2, 0))   # Edge
            if r1 and r2:
                refs = (r1, r2)
        else:
            raise ValueError(f"Invalid constraint type for {self.name}: {ctype}")

        assert self.validate_refs(refs)

        return self.create_constraint('Coincident', tuple(refs))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
