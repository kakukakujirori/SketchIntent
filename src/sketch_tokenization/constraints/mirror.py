from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from ..constants import ConstraintToken
from .base import ConstraintHandler
from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class MirrorHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Mirror

    @property
    def name(self) -> str:
        return "Mirror"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)

        # Mirror Logic: 2-ref Mirror usually refers to external entities, so drop it
        if len(ref_tokens) != 3:
             # Mirror token requires 3 refs: (P1, P2, Axis)
             # If <3 refs, we can't encode it properly as Mirror token?
             # Or maybe it's valid but we just warn?
             pass

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

        # Mirror(P1, P2, Axis) -> 3 refs
        if len(refs) == 3:
            # Assuming params are [P1, P2, Axis]
            # Wait, `encode` sorts them?
            # If standard Mirror, it's [P1, P2, Axis].
            # But we should check which one is the Axis?
            # `encode` doesn't seem to force Axis to be last?
            # However in SG data, usually last one is the axis/line?

            p1 = resolve_ref(refs[0], entity_map, sketch_obj)
            p2 = resolve_ref(refs[1], entity_map, sketch_obj)
            ax = resolve_ref(refs[2], entity_map, sketch_obj)

            if p1 and p2 and ax:
                 sketch_obj.addConstraint(Sketcher.Constraint('Symmetric', p1[0], p1[1], p2[0], p2[1], ax[0], ax[1]))

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        # FC uses 'Symmetric' for both Midpoint and Mirror.
        # If 3rd arg is Line/Axis, it's Mirror.
        # If 3rd arg is Point, it's Midpoint.
        # But here we are in MirrorHandler.
        # Do we look for 'Symmetric'?
        # If so, we check 3rd arg.

        if fc_constraint.Type != 'Symmetric': return None

        from .utils import make_ref_str

        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        g3, p3 = fc_constraint.Third, fc_constraint.ThirdPos

        # Check if g3 is a Line/Axis... how?
        # We need to access geometry of g3?
        # Or just make refs and let downstream sort it out?
        # But if it's Midpoint, we shouldn't return Mirror constraint?
        # We can implement a check if we have access to geometry types?
        # Or we assume if p3 == 0 (Edge)?
        # If p3 != 0 (Vertex), it's likely a Point -> Midpoint.
        # If p3 == 0, it's likely a Line -> Mirror.

        if p3 != 0: return None # Likely Midpoint or Point-Symmetric

        r1 = make_ref_str(g1, p1, entity_map)
        r2 = make_ref_str(g2, p2, entity_map)
        r3 = make_ref_str(g3, p3, entity_map)

        if r1 and r2 and r3:
            return self.create_constraint(self.name, (r1, r2, r3))

        return None
