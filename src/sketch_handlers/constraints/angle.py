from typing import Any
import math
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken

from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class AngleHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Angle

    @property
    def name(self) -> str:
        return "Angle"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        props = self.get_constraint_props(constraint)
        return self.token_id, ref_tokens, np.array([props["value"]]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        props = self.get_constraint_props(constraint)

        # NOTE: Since the original SketchGraphs angle data is unclear in how it was computed, we instead use angles calculated on the fly.
        #       Call src.sketch_handlers.sketch_utils.complete_constraint_values() first to make that change.
        #       - Value is already set in radians (used as-is by FreeCAD)
        #       - Aligned is always True
        #       - Clockwise is neglected
        val_rad = props["value"]
        if not props["aligned"]:
            raise ValueError(f"Aligned is false for {self.name}: {constraint=}. Run sketch_utils.complete_constraint_values() first.")

        if len(refs) == 2:
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            if res1 and res2:
                # Angle(Line1, Line2, value)
                return [('Angle', [res1[0], res2[0], val_rad], None)]
        elif len(refs) == 1:
            res1 = primitive_geoID.get(refs[0])
            if res1:
                # Angle(Line, value)
                return [('Angle', [res1[0], val_rad], None)]

        raise ValueError(f"Invalid references or value for {self.name}: {refs=}, {props=}")

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Angle':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        refs = []
        g1 = fc_constraint.First
        r1 = geoID_to_primitive.get((g1, 0))
        if r1:
            refs.append(r1)

        g2 = fc_constraint.Second
        if g2 >= 0:
            r2 = geoID_to_primitive.get((g2, 0))
            if r2:
                refs.append(r2)

        val = fc_constraint.Value  # radians from FC

        assert self.validate_refs(tuple(refs))

        return self.create_constraint(self.name, tuple(refs), val, aligned=True)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2
