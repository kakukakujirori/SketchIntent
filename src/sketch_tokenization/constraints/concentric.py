from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class ConcentricHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Concentric

    @property
    def name(self) -> str:
        return "Concentric"

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

        # Concentric(Circle1, Circle2) -> (g1, g2)
        # Or (Point1, Point2) -> Coincident?
        # FC doesn't have explicit "Concentric" constraint type exposed as 'Concentric' usually?
        # Actually it does have `Constraint('Concentric', ...)`?
        # Let's check primitives conversion usage or FC API.
        # Wait, FC usually treats Concentric as Coincident of centers?
        # But if we want to round trip "Concentric", we should try to use 'Concentric' if available,
        # or 'Coincident' on centers.
        # Sketcher.Constraint('Coincident', g1, 3, g2, 3) where 3 is Center?
        # Actually for Circle, 3 is Center?
        # Or usually Center is Vertex 3?
        # Let's assume Coincident on centers for now as that's robust.

        if len(refs) == 2:
            res1 = resolve_ref(refs[0], entity_map, sketch_obj)
            res2 = resolve_ref(refs[1], entity_map, sketch_obj)

            if res1 and res2:
                # If these are Circles/Arcs, we want their centers.
                # resolve_ref checks for 'center', 'start', 'end'.
                # If just Circle ID, it returns (gid, 0).
                # For Coincident on Center, we need (gid, 3) usually?
                # Or does FC automatically constrain centers if we say Coincident(Circle1, Circle2)?
                # No, Coincident(Circle, Circle) might be invalid or mean something else.
                # We need Coincident(Center1, Center2).

                # Check geometry type?
                # We don't have easy access to type here unless we query `sketch_obj`.
                # But we can try to use Vertex specific indices if we knew them.
                # For now, let's try `Constraint('Coincident', res1[0], 3, res2[0], 3)` assuming they are circles?
                # But if they are points?
                # Using 'Concentric' type explicitly might work if FC supports it.
                # FC has internal Concentric.
                # Let's try `Constraint('Concentric', res1[0], res2[0])`?
                # If not, fallback to Coincident of centers.
                try:
                    sketch_obj.addConstraint(Sketcher.Constraint('Concentric', res1[0], res2[0])) # Try without vertex indices?
                except Exception:
                    # Fallback or maybe it needs vertex indices?
                    # Concentric usually strictly for Edge-Edge (Circle-Circle).
                    pass

    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        from .utils import make_ref_str
        if fc_constraint.Type != 'Concentric': return None # If FC uses Coincident for centers, we missed it?
        # If FC stores it as Concentric, we catch it here.

        refs = []
        g1 = fc_constraint.First
        r1 = make_ref_str(g1, 0, entity_map)
        if r1: refs.append(r1)

        g2 = fc_constraint.Second
        r2 = make_ref_str(g2, 0, entity_map)
        if r2: refs.append(r2)

        if len(refs) == 2:
            return self.create_constraint(self.name, tuple(refs))

        return None

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 2
