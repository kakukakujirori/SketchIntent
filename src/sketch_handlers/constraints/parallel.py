from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class ParallelHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Parallel

    @property
    def name(self) -> str:
        return "Parallel"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
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

        # Parallel(e1, e2) — edge-edge only
        return [('Parallel', [res1[0], res2[0]], None)]

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Parallel':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        g2 = fc_constraint.Second
        r1 = geoID_to_primitive.get((g1, 0))
        r2 = geoID_to_primitive.get((g2, 0))
        if not r1 or not r2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {g1}, {g2}")

        return self.create_constraint(self.name, (r1, r2))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
