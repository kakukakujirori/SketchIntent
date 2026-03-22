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
class TangentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Tangent

    @property
    def name(self) -> str:
        return "Tangent"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of refs for {self.name}: {refs=}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs=} ({primitive_geoID=})")

        g1, p1 = res1
        g2, p2 = res2

        try:
            props = self.get_constraint_props(constraint)
            datum = props["value"]
        except ValueError:
            datum = None

        if p1 > 0 or p2 > 0:
            # Tangent with point specification: Tangent(e1, s1, e2, s2)
            if datum is not None:
                return [('Tangent', [g1, p1, g2, p2], datum)]
            return [('Tangent', [g1, p1, g2, p2], None)]
        else:
            # Edge-Edge tangent: Tangent(e1, e2)
            if datum is not None:
                return [('Tangent', [g1, g2], datum)]
            return [('Tangent', [g1, g2], None)]

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        FreeCAD, Part, Sketcher = ensure_freecad_imported()

        if fc_constraint.Type != 'Tangent':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos

        r1 = geoID_to_primitive.get((g1, p1 if p1 > 0 else 0))
        r2 = geoID_to_primitive.get((g2, p2 if p2 > 0 else 0))

        if not r1 or not r2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {g1}, {g2}")

        val = fc_constraint.Value

        if p1 == 0 and p2 == 0 and isinstance(fc_primitives[g1], Part.LineSegment) and isinstance(fc_primitives[g2], Part.LineSegment):
            return self.create_constraint('Coincident', (r1, r2), None)  # SketchGraphs achieve Collinear by Coincident(Line, 0, Line, 0)
        else:
            return self.create_constraint(self.name, (r1, r2), val if val != 0.0 else None)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
