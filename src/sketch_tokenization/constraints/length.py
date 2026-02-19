from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class LengthHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Length

    @property
    def name(self) -> str:
        return "Length"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), (ConstraintToken.Length,)

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj, entity_map: dict[str, int]):
        from .utils import resolve_ref
        try:
            import FreeCAD
            import Sketcher
        except ImportError:
            pass

        refs = constraint.get_references()
        val = None
        for p in constraint.parameters:
            if hasattr(p, 'expression'):
                val = p.value
                break

        # Length is usually 1 ref (Line/Arc) -> Distance in FC
        if len(refs) == 1:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            if res1:
                # Distance on Line (Length) -> Distance(g1, 0)
                # If val: Distance(g1, 0, val)? Or Distance(g1, val)?
                # Distance(Point, Point) is 4 args + val = 5 args.
                # Distance(Line) is 1 arg + val = 2 args? `Distance(g1, val)`
                c_name = "Distance" # FreeCAD uses 'Distance' for Length
                idx = -1
                if len(refs) > 0 and val is not None:
                    try:
                        # Length(Edge, Value) -> Distance(Edge, Value)
                        val = float(val)
                        idx = sketch_obj.addConstraint(Sketcher.Constraint(c_name, res1[0], val))
                        if idx == -1:
                             # Fallback to no-val?
                             idx = sketch_obj.addConstraint(Sketcher.Constraint(c_name, res1[0]))
                    except Exception:
                        pass
                else:
                    idx = sketch_obj.addConstraint(Sketcher.Constraint(c_name, res1[0]))

                if idx == -1:
                     print(f"Warning: Failed to add Length/Distance constraint on {res1[0]}")

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        # Length tokens don't come directly from a "Length" FC constraint type.
        # They come from "Distance".
        # So this might not be called directly if we key by FC Type.
        return None

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 1
