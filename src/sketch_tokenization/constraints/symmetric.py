from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class SymmetricHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return -1 # Dummy ID, only used for string mapping from FC

    @property
    def name(self) -> str:
        return "Symmetric"

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

        # Symmetric(Pt1, Pt2, Axis/Center)
        if len(refs) == 3:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            res2 = resolve_ref(refs[1], entity_map, sketch_obj)
            res3 = resolve_ref(refs[2], entity_map, sketch_obj)

            if res1 and res2 and res3:
                # FC Symmetric: (g1, p1, g2, p2, g3, p3)
                sketch_obj.addConstraint(Sketcher.Constraint('Symmetric', res1[0], res1[1], res2[0], res2[1], res3[0], res3[1]))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Symmetric': return None

        refs = []
        # FreeCAD Symmetric: First=Pt1, Second=Pt2, Third=Axis/Center
        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        r1 = make_ref_str(g1, p1, entity_map)

        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        r2 = make_ref_str(g2, p2, entity_map)

        g3, p3 = fc_constraint.Third, fc_constraint.ThirdPos
        r3 = make_ref_str(g3, p3, entity_map)

        if r1 and r2 and r3:
            refs = [r1, r2, r3]

            # Dispatch:
            # If r3 is a Point -> Midpoint
            # If r3 is a Line/Axis -> Symmetric
            # How to know if r3 is Point or Line?
            # Reference string doesn't tell type directly.
            # But p3 (Pos) might? p3=1,2,3 -> Point on Line. p3=0 -> Line itself? No.
            # Point entity is (Geo, 1). Line entity is (Geo, 0) if selecting edge?
            # Or if r3 refers to a Point entity.
            # Just return 'Symmetric' for now or 'Midpoint' if we guess?
            # 'Symmetric' is safe for (Pt, Pt, Line).
            # 'Midpoint' is safe for (Pt, Pt, Pt).

            # Ideally we check entity type in map, but map is int->str.
            # We can't easily check entity type unless we have access to entities dict, which strictly from_freecad doesn't have passed in?
            # But checking fc_constraint.Third geometry type?
            if hasattr(g3, 'ShapeType'): # FC Geometry doesn't have ShapeType, Part does.
                # g3 is a Geometry object (e.g. Part.LineSegment, Part.Point)
                # Wait, fc_constraint.Third is index? No it's the Geometry object?
                # No, Sketcher constraints usually return INDEX.
                # Wait, fc_constraint.First returns index?
                # ERROR CHECK: fc_constraint.First property returns INTEGER index in newer FC?
                # Or returns the object?
                # In python API: `constr.First` is integer index.
                # `constr.FirstPos` is integer sub-element.
                # My `make_ref_str` uses `geo_id` (int).
                # So `g3` is an int. We can't check its type here easily without the sketch object or entity map providing type info.
                pass

            # Fallback: Use 'Mirror' for Symmetric with Axis (it corresponds to Mirror in SG).
            # OR we can assume if p3 == 1 (Vertex) -> Midpoint?
            # If p3 == 0 (Edge) -> Mirror?
            # This is a reasonable heuristic for Line vs Vertex.

            c_name = 'Mirror'
            if p3 != 0: # Targeting a vertex/point
                c_name = 'Midpoint'

            return self.create_constraint(c_name, tuple(refs))

        return None

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 3
