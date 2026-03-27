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
class DistanceXHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.DistanceX

    @property
    def name(self) -> str:
        return "DistanceX"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        props = self.get_constraint_props(constraint)
        return self.token_id, ref_tokens, np.array([props["value"]]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint("Distance", refs, val, direction="HORIZONTAL")

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        props = self.get_constraint_props(constraint)
        if props["direction"] != "HORIZONTAL":
            raise ValueError(f"Invalid direction for {self.name}: {props['direction']}")

        if len(refs) == 2:
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            if res1 and res2:
                g1, p1 = res1
                g2, p2 = res2
                return [('DistanceX', [g1, p1, g2, p2, props["value"]], None)]
        elif len(refs) == 1:
            res1 = primitive_geoID.get(refs[0])
            if res1:
                g1, p1 = res1
                return [('DistanceX', [g1, p1, props["value"]], None)]

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        raise ValueError(f"Invalid references or value for {self.name}: {refs=} ({res1=}, {res2=}), {props=}")

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'DistanceX':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        refs = []
        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        r1 = geoID_to_primitive.get((g1, p1))
        if r1:
            refs.append(r1)

        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos
        if g2 >= 0:
            r2 = geoID_to_primitive.get((g2, p2))
            if r2:
                refs.append(r2)

        assert self.validate_refs(tuple(refs))

        val = fc_constraint.Value

        # 1 ref = Length, 2 refs = Distance
        if len(refs) == 1:
            return self.create_constraint("Length", tuple(refs), val, direction="HORIZONTAL")
        elif len(refs) == 2:
            return self.create_constraint("Distance", tuple(refs), val, direction="HORIZONTAL")

        raise ValueError(f"Invalid references or value for {self.name}: {refs=}, {val=}")

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2


@CONSTRAINT_HANDLERS.register_module()
class DistanceYHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.DistanceY

    @property
    def name(self) -> str:
        return "DistanceY"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        props = self.get_constraint_props(constraint)
        return self.token_id, ref_tokens, np.array([props["value"]]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint("Distance", refs, val, direction="VERTICAL")

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        props = self.get_constraint_props(constraint)
        if props["direction"] != "VERTICAL":
            raise ValueError(f"Invalid direction for {self.name}: {props['direction']}")

        if len(refs) == 2:
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            if res1 and res2:
                g1, p1 = res1
                g2, p2 = res2
                return [('DistanceY', [g1, p1, g2, p2, props["value"]], None)]
        elif len(refs) == 1:
            res1 = primitive_geoID.get(refs[0])
            if res1:
                g1, p1 = res1
                return [('DistanceY', [g1, p1, props["value"]], None)]

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        raise ValueError(f"Invalid references or value for {self.name}: {refs=} ({res1=}, {res2=}), {props=}")

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'DistanceY':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        refs = []
        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        r1 = geoID_to_primitive.get((g1, p1))
        if r1:
            refs.append(r1)

        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos
        if g2 >= 0:
            r2 = geoID_to_primitive.get((g2, p2))
            if r2:
                refs.append(r2)

        assert self.validate_refs(tuple(refs))

        val = fc_constraint.Value

        # 1 ref = Length, 2 refs = Distance
        if len(refs) == 1:
            return self.create_constraint("Length", tuple(refs), val, direction="VERTICAL")
        elif len(refs) == 2:
            return self.create_constraint("Distance", tuple(refs), val, direction="VERTICAL")

        raise ValueError(f"Invalid references or value for {self.name}: {refs=}, {val=}")

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2
