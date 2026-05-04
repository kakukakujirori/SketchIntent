from typing import Any, TYPE_CHECKING
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


@CONSTRAINT_HANDLERS.register_module()
class CoincidentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Coincident

    @property
    def name(self) -> str:
        return "Coincident"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) < 2:
            return torch.zeros(2, dtype=state.dtype)

        ref1, ref2 = refs[0], refs[1]
        eid1 = ref1.split('.')[0]
        eid2 = ref2.split('.')[0]
        e1 = state.sketch.entities.get(eid1)
        e2 = state.sketch.entities.get(eid2)

        # Case: Both are edges (Collinear)
        # An "edge" or "object" ref is just the eid (e.g. '4'), not '4.start' or '4.center'.
        is_edge1 = (ref1 == eid1) and not isinstance(e1, datalib.Point)
        is_edge2 = (ref2 == eid2) and not isinstance(e2, datalib.Point)

        if is_edge1 and is_edge2:
            if isinstance(e1, datalib.Line) and isinstance(e2, datalib.Line):
                p1 = state.get_point(ref1) # anchor of line 1
                p2 = state.get_point(ref2) # anchor of line 2
                _, d1, _, _ = state.get_line_params(eid1)
                _, d2, _, _ = state.get_line_params(eid2)
                
                # 1. Parallel (1 DoF): cross product of dirs
                cross = d1[0] * d2[1] - d1[1] * d2[0]
                
                # 2. On the same line (1 DoF): distance from p1 to line 2
                n2 = torch.stack([-d2[1], d2[0]])
                dist = torch.dot(p1 - p2, n2) / (torch.linalg.norm(d2) + 1e-15)
                
                return torch.stack([cross, dist])

        # Case: One is an edge, the other is a point (PointOnObject)
        elif is_edge1 or is_edge2:
            pt_ref = ref2 if is_edge1 else ref1
            edge_ref = ref1 if is_edge1 else ref2
            edge_eid = edge_ref.split('.')[0]
            edge_ent = state.sketch.entities[edge_eid]
            pt = state.get_point(pt_ref)

            if isinstance(edge_ent, datalib.Line):
                s, d, _, _ = state.get_line_params(edge_eid)
                n = torch.stack([-d[1], d[0]])
                dist = torch.dot(pt - s, n) / (torch.linalg.norm(d) + 1e-15)
                return dist.unsqueeze(0)
            elif isinstance(edge_ent, (datalib.Circle, datalib.Arc)):
                if isinstance(edge_ent, datalib.Arc):
                    c, r, _, _ = state.get_arc_params(edge_eid)
                else:
                    c, r = state.get_circle_params(edge_eid)
                dist = torch.linalg.norm(pt - c) - r
                return dist.unsqueeze(0)

        # Default point-to-point case
        p1 = state.get_point(refs[0])
        p2 = state.get_point(refs[1])
        return p1 - p2  # (2,): zero when coincident

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid references for {self.name}: {refs}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid references for {self.name}: {refs}")

        g1, p1 = res1
        g2, p2 = res2

        if p1 == 0 and p2 == 0:
            prim1 = primitives[refs[0]]
            prim2 = primitives[refs[1]]
            # Pint-XXX ==> PointOnObject
            if isinstance(prim1, datalib.Point) and isinstance(prim2, datalib.Point):
                return [('Coincident', [g1, 1, g2, 1], None)]
            elif isinstance(prim1, datalib.Point):
                return [('PointOnObject', [g1, 1, g2], None)]
            elif isinstance(prim2, datalib.Point):
                return [('PointOnObject', [g2, 1, g1], None)]
            # Line-Line ==> Collinear, which can be achieved by Tangent in FreeCAD
            elif isinstance(prim1, datalib.Line) and isinstance(prim2, datalib.Line):
                return [('Tangent', [g1, 0, g2, 0], None)]
            # Circle-Circle ==> Concentric + Equal Radius
            elif isinstance(prim1, (datalib.Circle, datalib.Arc)) and isinstance(prim2, (datalib.Circle, datalib.Arc)):
                return [('Coincident', [g1, 3, g2, 3], None), ('Equal', [g1, g2], None)]
            else:
                raise ValueError(f"Invalid references for {self.name}: {refs}")
        elif p1 == 0:
            # PointOnObject: point (g2,p2) on edge g1
            return [('PointOnObject', [g2, p2, g1], None)]
        elif p2 == 0:
            # PointOnObject: point (g1,p1) on edge g2
            return [('PointOnObject', [g1, p1, g2], None)]
        else:
            # Point-Point
            return [('Coincident', [g1, p1, g2, p2], None)]

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        ctype = fc_constraint.Type
        if ctype not in ('Coincident', 'PointOnObject'):
            raise ValueError(f"Invalid constraint type for {self.name}: {ctype}")

        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        prim1 = fc_primitives[g1]
        prim2 = fc_primitives[g2]

        refs = ()
        if ctype == 'Coincident':
            r1 = geoID_to_primitive.get((g1, p1))
            r2 = geoID_to_primitive.get((g2, p2))
            assert r1 is not None and r2 is not None, f"{(g1, p1)=}, {(g2, p2)=}, {geoID_to_primitive=}"
            refs = (r1, r2)
        elif ctype == 'PointOnObject':
            r1 = geoID_to_primitive.get((g1, p1))  # Point
            r2 = geoID_to_primitive.get((g2, 0))   # Edge
            assert r1 is not None and r2 is not None, f"{(g1, p1)=}, {(g2, 0)=}, {geoID_to_primitive=}"
            refs = (r1, r2)
        else:
            raise ValueError(f"Invalid constraint type for {self.name}: {ctype}")

        assert self.validate_refs(refs)

        if isinstance(prim1, (Part.Circle, Part.ArcOfCircle)) and p1 == 3 and isinstance(prim2, (Part.Circle, Part.ArcOfCircle)) and p2 == 3:
            return self.create_constraint('Concentric', (r1, r2))
        else:
            return self.create_constraint('Coincident', (r1, r2))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
