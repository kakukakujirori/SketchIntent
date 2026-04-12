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
class PerpendicularHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Perpendicular

    @property
    def name(self) -> str:
        return "Perpendicular"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of refs for {self.name}: {len(refs)}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs[0]}, {refs[1]}")

        g1, p1 = res1
        g2, p2 = res2

        try:
            props = self.get_constraint_props(constraint)
            datum = props["value"]
        except ValueError:
            datum = None

        if p1 > 0 and p2 > 0:
            # Perpendicular with point specification: Perpendicular(e1, s1, e2, s2)
            return [('Perpendicular', [g1, p1, g2, p2], datum)]
        elif p1 == 0 and p2 > 0:
            # Perpendicular with one point specification must decrate the point first
            return [('Perpendicular', [g2, p2, g1, p1], datum)]
        elif p2 == 0 and p1 > 0:
            # Perpendicular with one point specification must decrate the point first
            return [('Perpendicular', [g1, p1, g2, p2], datum)]
        elif p1 == 0 and p2 == 0:
            # Edge-Edge perpendicular: Perpendicular(e1, e2)
            return [('Perpendicular', [g1, g2], datum)]
        else:
            raise ValueError(f"Invalid point specification for {self.name}: {p1=}, {p2=}")

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Perpendicular':
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
        return self.create_constraint(self.name, (r1, r2), val if val != 0.0 else None)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
