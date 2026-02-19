from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class PerpendicularHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Perpendicular

    @property
    def name(self) -> str:
        return "Perpendicular"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

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
                # Perpendicular(Line1, Line2) or (Line1, Pt1, Line2, Pt2)
                # But here refs usually imply (Line, Line) for simplicity unless we have subnodes
                # If parsed as (gid, pid), we use it.
                if res1[1] > 0 or res2[1] > 0:
                     sketch_obj.addConstraint(Sketcher.Constraint('Perpendicular', res1[0], res1[1], res2[0], res2[1]))
                else:
                     sketch_obj.addConstraint(Sketcher.Constraint('Perpendicular', res1[0], res2[0]))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Perpendicular': return None

        refs = []
        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        r1 = make_ref_str(g1, p1, entity_map)
        if r1: refs.append(r1)

        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        r2 = make_ref_str(g2, p2, entity_map)
        if r2: refs.append(r2)

        return self.create_constraint('Perpendicular', tuple(refs))

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 2
