from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class EqualHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Equal

    @property
    def name(self) -> str:
        return "Equal"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj, entity_map: dict[str, int]):
        from .utils import resolve_ref
        try:
            import FreeCAD
            import Sketcher
        except ImportError:
            pass

        refs = []
        for p in constraint.parameters:
            if hasattr(p, 'value'):
                 refs.append(p.value)

        if len(refs) == 2:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            res2 = resolve_ref(refs[1], entity_map, sketch_obj)
            if res1 and res2:
                # Equal(Edge1, Edge2) - Line length or Circle radius
                sketch_obj.addConstraint(Sketcher.Constraint('Equal', res1[0], res2[0]))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Equal': return None

        refs = []
        g1 = fc_constraint.First
        r1 = make_ref_str(g1, 0, entity_map) # Edge
        if r1: refs.append(r1)

        g2 = fc_constraint.Second
        r2 = make_ref_str(g2, 0, entity_map) # Edge
        if r2: refs.append(r2)

        return self.create_constraint('Equal', tuple(refs))

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) >= 2
