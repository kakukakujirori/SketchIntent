from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class MidpointHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Midpoint

    @property
    def name(self) -> str:
        return "Midpoint"

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

        # Midpoint(Mid, P1, P2) or (Mid, Line)
        # We need to identify which ref is Midpoint.
        # But `constraint.parameters` order?
        # `encode` sorts them so "Mid" comes first!
        # Datalib Constraint stores parameters in order.
        # Use simple heuristic: 3 refs -> Symmetric(P1, P2, Mid)
        # 2 refs -> Symmetric(Line_Start, Line_End, Mid) ??
        # Or (Point, Line) -> Point on Line + Symmetry?
        # FreeCAD 'Symmetric' takes (g1, p1, g2, p2, g3, p3) where 3 is center.

        if len(refs) == 3:
            # Assuming params are [Mid, P1, P2] based on `encode` logic?
            # Actually `encode` puts Mid first.
            # So refs[0] is Midpoint.
            rmid = resolve_ref(refs[0], entity_map, sketch_obj)
            p1 = resolve_ref(refs[1], entity_map, sketch_obj)
            p2 = resolve_ref(refs[2], entity_map, sketch_obj)

            if rmid and p1 and p2:
                 sketch_obj.addConstraint(Sketcher.Constraint('Symmetric', p1[0], p1[1], p2[0], p2[1], rmid[0], rmid[1]))

        elif len(refs) == 2:
            # (Mid, Line) or (Mid, P_other)?
            # Encode puts Mid first.
            # If 2nd ref is Line, we use Line's endpoints?
            rmid = resolve_ref(refs[0], entity_map, sketch_obj)
            rline = resolve_ref(refs[1], entity_map, sketch_obj)

            if rmid and rline and rline[1] == 0: # Line edge
                # Constrain Mid to be Symmetric between Start and End of Line
                # We need GID of line to get Start/End vertex IDs?
                # Actually we can use (gid, 1) and (gid, 2) for start/end of line.
                sketch_obj.addConstraint(Sketcher.Constraint('Symmetric', rline[0], 1, rline[0], 2, rmid[0], rmid[1]))
                # But wait, Midpoint implies Point is ON the line too?
                # Symmetric doesn't force OnObject.
                # Midpoint = PointOnObject + Symmetric?
                # SG Midpoint usually implies Coincident too?
                # Let's add PointOnObject as well if needed?
                # Actually usually Midpoint constraint implies it lies on the segment/line.
                sketch_obj.addConstraint(Sketcher.Constraint('PointOnObject', rmid[0], rmid[1], rline[0]))


    def from_freecad(self, fc_constraint, entity_map: dict[int, str]) -> datalib.Constraint:
        # We map 'Symmetric' to 'Symmetric' handler usually.
        # But if we want to detect Midpoint specifically from FC...
        # 'Symmetric' with 3rd arg as Point is Midpoint.
        # 'Symmetric' with 3rd arg as Line/Axis is Mirror.
        # This handler checks if it validates as Midpoint?
        # But `from_freecad` is called by type.
        # If we rely on Symmetric handler to handle all 'Symmetric', return None here.
        return None

    def _get_sorted_ref_tokens(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, ...]:
        refs = self._parse_references(constraint, primitive_tokenpos)
        refs = list(set(refs))

        sorted_refs = []
        # Special Midpoint Sorting
        if len(refs) == 2:
            # 2-ref: swap if first ref is a curve (Line/Arc/Circle), not a point/subnode
            ent0 = entities.get(refs[0][0])
            if ent0 and not isinstance(ent0, datalib.Point):
                sorted_refs = list(reversed(refs))
            else:
                sorted_refs = refs
        else:
            # 3+ refs: determine midpoint geometrically
            # For Midpoint(mid, p1, p2), mid.coords = (p1.coords + p2.coords) / 2

            def get_point_coords(entity_idx: str, subnode_name: str | None):
                """Get (x, y) coordinates for a reference."""
                ent = entities.get(entity_idx)
                if ent is None:
                    return None
                if isinstance(ent, datalib.Point):
                    return (ent.x, ent.y)
                elif isinstance(ent, datalib.Line):
                    if subnode_name == 'start':
                        sp = ent.start_point
                        return (sp[0], sp[1])
                    elif subnode_name == 'end':
                        ep = ent.end_point
                        return (ep[0], ep[1])
                    else:
                        # Edge reference - use midpoint of line
                        sp = ent.start_point
                        ep = ent.end_point
                        return ((sp[0] + ep[0]) / 2, (sp[1] + ep[1]) / 2)
                elif isinstance(ent, (datalib.Circle, datalib.Arc)):
                    if subnode_name in ('start', 'end'):
                         # For arc start/end, would need more complex calc but usually center is ref
                         pass
                    return (ent.xCenter, ent.yCenter)
                return None

            # Find which ref is the geometric midpoint
            best_mid_idx = 0
            best_err = float('inf')

            for mid_idx in range(len(refs)):
                mid_entity_idx, mid_subnode = refs[mid_idx]
                mid_coords = get_point_coords(mid_entity_idx, mid_subnode)
                if mid_coords is None:
                    continue

                # Get other refs
                other_refs = [refs[i] for i in range(len(refs)) if i != mid_idx]
                if len(other_refs) < 2:
                    continue

                # Check if mid is midpoint of first two others
                # NOTE: Only checking first two of others.
                # If there are >3 refs, this might be partial, but standard Midpoint is 3 refs.
                p1_coords = get_point_coords(other_refs[0][0], other_refs[0][1])
                p2_coords = get_point_coords(other_refs[1][0], other_refs[1][1])
                if p1_coords is None or p2_coords is None:
                    continue

                calc_mid_x = (p1_coords[0] + p2_coords[0]) / 2
                calc_mid_y = (p1_coords[1] + p2_coords[1]) / 2
                err = ((mid_coords[0] - calc_mid_x)**2 + (mid_coords[1] - calc_mid_y)**2)**0.5

                if err < best_err:
                    best_err = err
                    best_mid_idx = mid_idx

            # Reorder: midpoint first, then others sorted by token position
            mid_ref = refs[best_mid_idx]
            other_refs = [refs[i] for i in range(len(refs)) if i != best_mid_idx]

            def get_ref_pos(r):
                key = f"{r[0]}.{r[1]}" if r[1] else r[0]
                return primitive_tokenpos.get(key, 0)

            sorted_others = sorted(other_refs, key=get_ref_pos)
            sorted_refs = [mid_ref] + sorted_others

        ref_tokens = []
        for eid, sub in sorted_refs:
            key = f"{eid}.{sub}" if sub else eid
            ref_tokens.append(primitive_tokenpos[key])
        return tuple(ref_tokens)

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) >= 2
