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
class RadiusHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Radius

    @property
    def name(self) -> str:
        return "Radius"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        props = self.get_constraint_props(constraint)
        return self.token_id, ref_tokens, np.array([props["value"]]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) < 1:
            return torch.zeros(1, dtype=state.dtype)
        target = self.get_constraint_props(constraint)["value"]
        eid = refs[0].split('.')[0]
        r = state.get_radius(eid)
        return (r - target).unsqueeze(0)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        props = self.get_constraint_props(constraint)

        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of refs for {self.name}: {len(refs)}")

        res1 = primitive_geoID.get(refs[0])
        if res1 and props["value"] > 0:
            # Radius(Circle/Arc, value)
            return [('Radius', [res1[0], props["value"]], None)]

        raise ValueError(f"Invalid references or value for {self.name}: {refs=}, {props=}")

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Radius':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        r1 = geoID_to_primitive.get((g1, 0))  # Circle/Arc edge
        if not r1:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {g1}")

        val = fc_constraint.Value
        return self.create_constraint(self.name, (r1,), val)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1
