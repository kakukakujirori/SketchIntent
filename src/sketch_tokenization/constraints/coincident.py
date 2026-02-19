from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class CoincidentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Coincident

    @property
    def name(self) -> str:
        return "Coincident"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj, entity_map: dict[str, int]):
        from .utils import resolve_ref
        try:
            import FreeCAD
            import Sketcher
        except ImportError:
            pass # Should be in FreeCAD environment

        # Parse refs
        refs = []
        for p in constraint.parameters:
            if hasattr(p, 'value'):
                 refs.append(p.value)


        if len(refs) != 2:
            return

        res1 = resolve_ref(refs[0], entity_map, sketch_obj)
        res2 = resolve_ref(refs[1], entity_map, sketch_obj)

        if not res1 or not res2: return

        g1, p1 = res1
        g2, p2 = res2

        try:
            if p1 == 0 and p2 == 0:
                pass # Edge-Edge Coincident? usually not valid or redundant
            elif p1 == 0: # PointOnObject
                sketch_obj.addConstraint(Sketcher.Constraint('PointOnObject', g2, p2, g1))
            elif p2 == 0: # PointOnObject
                sketch_obj.addConstraint(Sketcher.Constraint('PointOnObject', g1, p1, g2))
            else: # Point-Point
                sketch_obj.addConstraint(Sketcher.Constraint('Coincident', g1, p1, g2, p2))
        except Exception as e:
            print(f"Error adding Coincident: {e}")

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str

        # Check type
        ctype = fc_constraint.Type
        if ctype not in ('Coincident', 'PointOnObject'):
            return None

        # Extract refs
        # FreeCAD Coincident: (g1, p1, g2, p2)
        # PointOnObject: (g_pt, p_pt, g_edge) -> In FC API it is (Geo1, Pt1, Geo2) but usually mapped to (Point, Edge)
        # Wait, PointOnObject(g1, p1, g2) -> g1=Point, p1=1, g2=Edge.

        refs = []

        if ctype == 'Coincident':
            g1, p1, g2, p2 = fc_constraint.First, fc_constraint.FirstPos, fc_constraint.Second, fc_constraint.SecondPos
            r1 = make_ref_str(g1, p1, entity_map)
            r2 = make_ref_str(g2, p2, entity_map)
            if r1 and r2:
                refs = [r1, r2]
        elif ctype == 'PointOnObject':
            g1, p1, g2 = fc_constraint.First, fc_constraint.FirstPos, fc_constraint.Second
            r1 = make_ref_str(g1, p1, entity_map) # Point
            r2 = make_ref_str(g2, 0, entity_map)  # Edge (p2 assumed 0)
            if r1 and r2:
                refs = [r1, r2]

        if len(refs) != 2:
            return None

        return self.create_constraint('Coincident', tuple(refs))


    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) >= 2
