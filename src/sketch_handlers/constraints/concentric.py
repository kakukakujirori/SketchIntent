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
class ConcentricHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Concentric

    @property
    def name(self) -> str:
        return "Concentric"

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

        # Concentric is Coincident on centers: Coincident(e1, 3, e2, 3)
        # FreeCAD represents concentric as a Coincident with center point subnodes
        return [('Coincident', [res1[0], 3, res2[0], 3], None)]

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        # Concentric comes in as Coincident with both FirstPos=3 and SecondPos=3
        if fc_constraint.Type != 'Coincident':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        p1 = fc_constraint.FirstPos
        p2 = fc_constraint.SecondPos

        # Only match center-center coincidents
        # NOTE: Otherwise fall through to Coincident handler, so don't raise an error here!
        if p1 != 3 or p2 != 3:
            return None

        g1 = fc_constraint.First
        g2 = fc_constraint.Second
        r1 = geoID_to_primitive.get((g1, 0))  # Edge reference
        r2 = geoID_to_primitive.get((g2, 0))  # Edge reference
        if not r1 or not r2:
            return None

        return self.create_constraint('Concentric', (r1, r2))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
