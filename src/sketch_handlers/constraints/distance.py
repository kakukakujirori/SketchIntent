from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class DistanceHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Distance

    @property
    def name(self) -> str:
        return "Distance"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        val = self.get_constraint_value(constraint)

        if len(refs) == 1:
            # Line length: Distance(e1, value)
            res1 = primitive_geoID.get(refs[0])
            if res1:
                return [('Distance', [res1[0], float(val)], None)]
        elif len(refs) == 2:
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            if res1 and res2:
                g1, p1 = res1
                g2, p2 = res2
                # Point-Point or Point-Line distance
                return [('Distance', [g1, p1, g2, p2, float(val)], None)]
        else:
            raise ValueError(f"Invalid number of references for {self.name}: {len(refs)}")

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != self.name:
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        refs = []
        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        r1 = geoID_to_primitive.get((g1, p1))
        if r1:
            refs.append(r1)

        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos
        if g2 >= 0:  # g2 == -1 means no second entity (line length)
            r2 = geoID_to_primitive.get((g2, p2))
            if r2:
                refs.append(r2)

        val = fc_constraint.Value

        # 1 ref = Length, 2 refs = Distance
        if len(refs) == 1:
            return self.create_constraint("Length", tuple(refs), val)
        elif len(refs) == 2:
            return self.create_constraint(self.name, tuple(refs), val)
        else:
            raise ValueError(f"Invalid number of references for {self.name}: {len(refs)}")

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2
