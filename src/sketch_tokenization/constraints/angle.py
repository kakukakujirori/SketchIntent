from typing import Any
import math
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class AngleHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Angle

    @property
    def name(self) -> str:
        return "Angle"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj: Any, entity_map: dict[str, int]) -> int:
        from .utils import resolve_ref
        from ..freecad_utils import ensure_freecad_imported
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        val_rad = None
        for p in constraint.parameters:
            if p.type == datalib.ConstraintParameterType.Quantity:
                # Expect "X DEGREE"
                try:
                    val_str = p.expression.split(' ')[0]
                    val_deg = float(val_str)
                    val_rad = math.radians(val_deg)
                except (ValueError, IndexError):
                    pass

        refs = constraint.get_references()

        if len(refs) == 2:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            res2 = resolve_ref(refs[1], entity_map, sketch_obj)

            if res1 and res2:
                # Angle(Line1, Line2) -> (g1, g2)
                # But could also be (Line1, g2) where g2 is Axis?
                # For now assume Line-Line angle
                # FC Angle: (g1, g2) or (g1, p1, g2, p2)??
                # Usually Angle(Line1, Line2) is 2 args.
                if len(refs) >= 2 and val is not None:
                    try:
                        # Angle(Line1, Line2, Value)
                        # Or Angle(Line1, Point1, ...)? No usually Line/Line
                        if res1 and res2:
                            val = float(val)
                            print(f"DEBUG: Angle({res1[0]}, {res1[1]}, {res2[0]}, {res2[1]}, {val})")
                            idx = sketch_obj.addConstraint(Sketcher.Constraint('Angle', res1[0], res1[1], res2[0], res2[1], val))
                            if idx == -1:
                                 print(f"Warning: Failed to add Angle({res1[0]}, {res1[1]}, {res2[0]}, {res2[1]}, {val})")
                                 # Fallback to driving=False (no val)
                                 idx = sketch_obj.addConstraint(Sketcher.Constraint('Angle', res1[0], res1[1], res2[0], res2[1]))
                                 if idx != -1: sketch_obj.setDatum(idx, val)

                    except Exception as e:
                        print(f"DEBUG: Angle exception: {e}")
                        idx = sketch_obj.addConstraint(Sketcher.Constraint('Angle', res1[0], res2[0]))
                elif val is not None:
                    # Generic fallback
                    idx = sketch_obj.addConstraint(Sketcher.Constraint('Angle', res1[0], res2[0], val))
                else:
                    idx = sketch_obj.addConstraint(Sketcher.Constraint('Angle', res1[0], res2[0]))

                # if val is not None and idx > -1:
                #      sketch_obj.setDatum(idx, val)

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Angle': return None

        refs = []
        g1 = fc_constraint.First
        r1 = make_ref_str(g1, 0, entity_map)
        if r1: refs.append(r1)

        g2 = fc_constraint.Second
        r2 = make_ref_str(g2, 0, entity_map)
        if r2: refs.append(r2)

        val = fc_constraint.Value

        # val = fc_constraint.Value # Already extracted above

        return self.create_constraint(self.name, tuple(refs), val)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) >= 2
