from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class DiameterHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Diameter

    @property
    def name(self) -> str:
        return "Diameter"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        val = self.get_constraint_value(constraint)
        return self.token_id, ref_tokens, np.array([val]), ()

    def to_freecad(self, constraint: datalib.Constraint, sketch_obj: Any, entity_map: dict[str, int]) -> int:
        print(f"DEBUG: Entering DiameterHandler.to_freecad for {constraint.identifier}")
        from .utils import resolve_ref
        from ..freecad_utils import ensure_freecad_imported
        FreeCAD, Part, Sketcher = ensure_freecad_imported()
        refs = constraint.get_references()
        val = None
        for p in constraint.parameters:
             if hasattr(p, 'expression'):
                 val = p.value
                 break

        if len(refs) == 1:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            if res1:
                if val is not None:
                    try:
                        val = float(val)
                        idx = sketch_obj.addConstraint(Sketcher.Constraint('Diameter', res1[0], val))
                    except:
                        idx = sketch_obj.addConstraint(Sketcher.Constraint('Diameter', res1[0], 0))
                else:
                    idx = sketch_obj.addConstraint(Sketcher.Constraint('Diameter', res1[0], 0))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Diameter': return None

        refs = []
        g1 = fc_constraint.First
        r1 = make_ref_str(g1, 0, entity_map)
        if r1: refs.append(r1)

        val = fc_constraint.Value

        return self.create_constraint(self.name, tuple(refs), val)

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 1
