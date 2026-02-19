from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class RadiusHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Radius

    @property
    def name(self) -> str:
        return "Radius"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj: Any, entity_map: dict[str, int]) -> int:
        print(f"DEBUG: Entering RadiusHandler.to_freecad for {constraint.identifier}")
        from .utils import resolve_ref
        from ..freecad_utils import ensure_freecad_imported
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        refs = constraint.get_references()
        val = None
        for p in constraint.parameters:
            if hasattr(p, 'expression'):
                val = p.value
                break

        print(f"DEBUG: RadiusHandler refs={refs}, val={val}")

        res1 = None
        if len(refs) > 0:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            print(f"DEBUG: RadiusHandler resolved res1={res1}")

        if val is not None:
            if res1:
                try:
                    # Standard Radius(Edge, Value)
                    print(f"DEBUG: Radius({res1[0]}, {val})")
                    print(f"DEBUG: Handler Sketcher: {Sketcher}")
                    idx = sketch_obj.addConstraint(Sketcher.Constraint('Radius', res1[0], val))
                    if idx > -1:
                         print(f"DEBUG: Radius added, idx={idx}. Constraints len={len(sketch_obj.Constraints)}")
                         try:
                             sketch_obj.setDatum(idx, val)
                             print(f"DEBUG: setDatum success")
                         except Exception as e:
                             print(f"DEBUG: setDatum failed: {e}")
                    else:
                         print(f"Warning: Failed to add Radius({res1[0]}, {val}) -> idx -1")
                         # Try fallback
                         idx = sketch_obj.addConstraint(Sketcher.Constraint('Radius', res1[0]))
                         if idx != -1: sketch_obj.setDatum(idx, val)
                except Exception as e:
                    print(f"DEBUG: Radius exception: {e}")
                    sketch_obj.addConstraint(Sketcher.Constraint('Radius', res1[0], 0))
            else:
                # res1 is None
                pass
        else:
            # val is None
            pass

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Radius': return None

        refs = []
        g1 = fc_constraint.First
        r1 = make_ref_str(g1, 0, entity_map) # Circle/Arc
        if r1: refs.append(r1)

        val = fc_constraint.Value

        return self.create_constraint(self.name, refs, val)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 1
