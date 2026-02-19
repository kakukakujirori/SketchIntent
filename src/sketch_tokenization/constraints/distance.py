from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

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

        # Check Direction if we were to support DistanceX/Y encoding
        # But here we just return Distance token
        return self.token_id, ref_tokens, np.array([val]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj: Any, entity_map: dict[str, int]) -> int:
        from .utils import resolve_ref
        from ..freecad_utils import ensure_freecad_imported
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        refs = constraint.get_references()
        val = None
        for p in constraint.parameters:
            if hasattr(p, 'expression'):
                val = p.value
                break

        # Distance/DistanceX/DistanceY behave similarly regarding refs
        # 1 ref: Line length (or P-P if handled via line, but usually Line)
        # 2 refs: Point-Point distance (or Point-Line)

        c_name = self.name # 'Distance', 'DistanceX', 'DistanceY'

        if len(refs) == 1:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            if res1:
                # Same as Length (Distance on 1 entity)
                if val is not None:
                    idx = sketch_obj.addConstraint(Sketcher.Constraint(c_name, res1[0], val))
                else:
                    idx = sketch_obj.addConstraint(Sketcher.Constraint(c_name, res1[0], 0))

        elif len(refs) == 2:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            res2 = resolve_ref(refs[1], entity_map, sketch_obj)

            if len(refs) >= 2 and val is not None:
             try:
                 val = float(val)
                 if len(refs) == 1: # Distance(Line, Val) -> Length? No, distance usually 2 points or point/line
                     pass
                 elif len(refs) == 2: # Distance(Pt1, Pt2, Val)
                      if res1 and res2:
                           idx = sketch_obj.addConstraint(Sketcher.Constraint('Distance', res1[0], res1[1], res2[0], res2[1], val))
                           if idx == -1:
                                idx = sketch_obj.addConstraint(Sketcher.Constraint('Distance', res1[0], res1[1], res2[0], res2[1]))
                                if idx != -1: sketch_obj.setDatum(idx, val)

             except Exception:
                 pass
            # This line was likely intended as the 'else' for 'val is not None' or a fallback
            # Re-adding it here, assuming it's the fallback for when val is None or conversion fails
            # and also for the case where the try-except block is skipped (e.g., val is None)
            elif res1 and res2: # Original else branch for val is None
                idx = sketch_obj.addConstraint(Sketcher.Constraint(c_name, res1[0], res1[1], res2[0], res2[1]))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str

        # Ensure type matches handler
        if fc_constraint.Type != self.name:
             return None

        refs = []

        g1 = fc_constraint.First
        r1 = make_ref_str(g1, fc_constraint.FirstPos, entity_map)
        if r1: refs.append(r1)

        g2 = fc_constraint.Second
        if g2 is not None: # Might be None for Line Length (Distance on 1 Line)
             r2 = make_ref_str(g2, fc_constraint.SecondPos, entity_map)
             if r2: refs.append(r2)

        val = fc_constraint.Value

        # Dispatch: 1 ref = Length, 2 refs = Distance
        if len(refs) == 1:
            return self.create_constraint("Length", tuple(refs), val)
        else:
            return self.create_constraint(self.name, tuple(refs), val)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) >= 1
