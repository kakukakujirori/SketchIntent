from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class MirrorHandler(ConstraintHandler):
    """
    Mirror: Two entities are mirrored across an axis (line).

    In SketchGraphs: Mirror with 3 refs:
        - ref0: point on entity1
        - ref1: point on entity2
        - ref2: axis line (edge)

    In FreeCAD: Symmetric(e1, s1, e2, s2, axis_g)
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
        if len(refs) != 3:
            return []

        res0 = primitive_geoID.get(refs[0])
        res1 = primitive_geoID.get(refs[1])
        res2 = primitive_geoID.get(refs[2])
        if not res0 or not res1 or not res2:
            return []

        g0, p0 = res0
        g1, p1 = res1
        g2, p2 = res2

        # Determine which ref is the axis (edge with pid=0)
        if p2 == 0:
            # ref2 is axis
            return [('Symmetric', [g0, p0, g1, p1, g2], None)]
        elif p0 == 0:
            # ref0 is axis
            return [('Symmetric', [g1, p1, g2, p2, g0], None)]
        elif p1 == 0:
            # ref1 is axis
            return [('Symmetric', [g0, p0, g2, p2, g1], None)]

        return []

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        # Mirror comes in as Symmetric where it's NOT a midpoint
        # (i.e., First and Second are different entities)
        # This is handled by SymmetricHandler; Mirror is a SketchGraphs-level
        # distinction that we only create from specific patterns.
        raise NotImplementedError(f"{self.name}.from_freecad is not implemented")

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 3
