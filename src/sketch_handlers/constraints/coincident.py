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
