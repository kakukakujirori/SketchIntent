from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class HorizontalHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Horizontal

    @property
    def name(self) -> str:
        return "Horizontal"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)

        if len(refs) == 1:
            # Line constraint: Horizontal(e1)
            res1 = primitive_geoID.get(refs[0])
            assert res1 is not None, f"Invalid references for {self.name}: {refs=} ({res1=})"
            return [('Horizontal', [res1[0]], None)]
        elif len(refs) == 2:
            # Two-point constraint: Horizontal(e1, s1, e2, s2)
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            assert res1 and res2, f"Invalid references for {self.name}: {refs=} ({res1=}, {res2=})"
            g1, p1 = res1
            g2, p2 = res2
            return [('Horizontal', [g1, p1, g2, p2], None)]

        raise ValueError(f"Invalid references for {self.name}: {refs=}")

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Horizontal':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        refs = []
        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos

        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos

        if g2 >= 0 and p1 > 0 and p2 > 0:
            # Two-point form
            r1 = geoID_to_primitive.get((g1, p1))
            r2 = geoID_to_primitive.get((g2, p2))
            if r1 and r2:
                refs = [r1, r2]
        else:
            # Line form
            r1 = geoID_to_primitive.get((g1, 0))
            if r1:
                refs = [r1]

        assert self.validate_refs(tuple(refs))

        return self.create_constraint(self.name, tuple(refs))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2
