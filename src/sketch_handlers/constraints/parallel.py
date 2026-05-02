from typing import Any, TYPE_CHECKING
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


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

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) < 2:
            return torch.zeros(1, dtype=state.dtype)
        eid1 = refs[0].split('.')[0]
        eid2 = refs[1].split('.')[0]
        # Direction vectors from line parameters
        s1 = state.get_point(eid1 + '.start')
        e1 = state.get_point(eid1 + '.end')
        s2 = state.get_point(eid2 + '.start')
        e2 = state.get_point(eid2 + '.end')
        d1 = e1 - s1
        d2 = e2 - s2
        # 2D cross product (scalar): zero when parallel
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        return cross.unsqueeze(0)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of refs for {self.name}: {len(refs)}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs[0]}, {refs[1]}")

        # Parallel(e1, e2) — edge-edge only
        return [('Parallel', [res1[0], res2[0]], None)]

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
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
