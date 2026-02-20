from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class MirrorHandler(ConstraintHandler):
    """
    Mirror / Symmetric: Two points are symmetric about an axis.

    In SketchGraphs: Mirror with 3 refs:
        - ref0: point on entity1
        - ref1: point on entity2
        - ref2: axis (line edge or point)

    In FreeCAD: Symmetric(e1, s1, e2, s2, axis_g [, axis_p])
    """

    @property
    def token_id(self) -> int:
        return ConstraintToken.Mirror

    @property
    def name(self) -> str:
        return "Mirror"

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

        res0 = primitive_geoID.get(refs[0])
        res1 = primitive_geoID.get(refs[1])
        res2 = primitive_geoID.get(refs[2])
        if not res0 or not res1 or not res2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs[0]}, {refs[1]}, {refs[2]}")

        g0, p0 = res0
        g1, p1 = res1
        g2, p2 = res2

        # Determine which ref is the axis (edge with pid=0)
        if p2 == 0:
            # ref2 is axis (edge)
            return [('Symmetric', [g0, p0, g1, p1, g2], None)]
        elif p0 == 0:
            # ref0 is axis (edge)
            return [('Symmetric', [g1, p1, g2, p2, g0], None)]
        elif p1 == 0:
            # ref1 is axis (edge)
            return [('Symmetric', [g0, p0, g2, p2, g1], None)]
        else:
            # All refs are points — axis point case, put ref2 as axis
            return [('Symmetric', [g0, p0, g1, p1, g2, p2], None)]

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

        return self.create_constraint('Mirror', (r1, r2, r3))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 3
