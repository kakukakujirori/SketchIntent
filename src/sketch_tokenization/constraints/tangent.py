from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class TangentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Tangent

    @property
    def name(self) -> str:
        return "Tangent"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj, entity_map: dict[str, int]):
        from .utils import resolve_ref
        try:
            import FreeCAD  # ty:ignore[unresolved-import]
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
                # Tangent(Edge, Edge) or (Edge, Point) or (Point, Point)
                # FreeCAD API: (Geo1, Pt1, Geo2, Pt2)
                # Often Tangent(Curve1, Curve2) -> (g1, 0, g2, 0) is valid in API?
                # Actually, Tangent(Circle, Line) is (g1, g2).
                if res1[1] == 0 and res2[1] == 0:
                     sketch_obj.addConstraint(Sketcher.Constraint('Tangent', res1[0], res2[0]))
                else:
                     sketch_obj.addConstraint(Sketcher.Constraint('Tangent', res1[0], res1[1], res2[0], res2[1]))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type not in ('Tangent', 'TangentViaPoint'): return None

        refs = []
        # Tangent: (g1, p1, g2, p2)
        # TangentViaPoint: (g1, g2, g_pt, p_pt) -> g1 tangent to g2 at pt

        if fc_constraint.Type == 'Tangent':
            g1, p1 = fc_constraint.First, fc_constraint.FirstPos
            g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
            r1 = make_ref_str(g1, p1, entity_map)
            r2 = make_ref_str(g2, p2, entity_map)
            if r1: refs.append(r1)
            if r2: refs.append(r2)

        elif fc_constraint.Type == 'TangentViaPoint':
            # This is tricky. It implies g1 is tangent to g2 at point (g_pt, p_pt).
            # Usually mapped to Tangent(g1, g2) in SG?
            # Or Tangent(g1, pt)?
            # Let's map to Tangent(g1, g2) if possible, or include point?
            # SG Tangent usually takes 2 args.
            g1 = fc_constraint.First
            g2 = fc_constraint.Second

            # Use edges
            r1 = make_ref_str(g1, 0, entity_map)
            r2 = make_ref_str(g2, 0, entity_map)
            if r1: refs.append(r1)
            if r2: refs.append(r2)

        return self.create_constraint('Tangent', tuple(refs))

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 2
