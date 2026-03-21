from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from sketchgraphs.data._constraint import DirectionValue
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class DistanceHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Distance

    @property
    def name(self) -> str:
        return "Distance"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        props = self.get_constraint_props(constraint)

        # Potentially changes to DistanceX or DistanceY
        token_id = self.token_id
        for param in constraint.parameters:
            if getattr(param, 'parameterId', None) == 'direction':
                dir_val = getattr(param, 'value', None)
                if dir_val in ('HORIZONTAL', DirectionValue.HORIZONTAL):
                    token_id = ConstraintToken.DistanceX
                elif dir_val in ('VERTICAL', DirectionValue.VERTICAL):
                    token_id = ConstraintToken.DistanceY

        return token_id, ref_tokens, np.array([props["value"]]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint(self.name, refs, val)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        props = self.get_constraint_props(constraint)

        type_name = ""
        if props["direction"] == "MINIMUM":
            type_name = "Distance"
        elif props["direction"] == "HORIZONTAL":
            type_name = "DistanceX"
        elif props["direction"] == "VERTICAL":
            type_name = "DistanceY"
        else:
            raise ValueError(f"Invalid direction for {self.name}: {props['direction']}")

        if len(refs) == 1:
            # Line length: Distance(e1, value)
            res1 = primitive_geoID.get(refs[0])
            assert res1
            return [(type_name, [res1[0], props["value"]], None)]
        elif len(refs) == 2:
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            assert res1 and res2
            g1, p1 = res1
            g2, p2 = res2

            # NOTE: Line-Line is invalid in FreeCAD. If parallel, translate it to Point-Line distance
            ent1 = primitives[refs[0].split(".")[0]]
            ent2 = primitives[refs[1].split(".")[0]]
            if p1 == p2 == 0 and isinstance(ent1, datalib.Line) and isinstance(ent2, datalib.Line):
                # check if parallel
                l1_dir = np.array([ent1.dirX, ent1.dirY])
                l2_dir = np.array([ent2.dirX, ent2.dirY])
                l1_dir_norm = np.linalg.norm(l1_dir)
                l2_dir_norm = np.linalg.norm(l2_dir)
                if np.abs(l1_dir[0] * l2_dir[1] - l1_dir[1] * l2_dir[0]) / (l1_dir_norm * l2_dir_norm) < 0.001:
                    return [(type_name, [g1, 1, g2, 0, props["value"]], None)]
                else:
                    raise ValueError(f"Lines {refs[0]} and {refs[1]} are not parallel, so Distance constraint cannot be applied.")

            # NOTE: Line-XXX is invalid. Reverse to XXX-Line (XXX = Point, Circle, Arc)
            # NOTE: When measuring the distance between Circle/Arc and Line, negative values mean that Line is inside the Circle/Arc.
            if isinstance(ent1, datalib.Line) and p1 == 0:
                return [(type_name, [g2, p2, g1, p1, props["value"]], None)]
            else:
                return [(type_name, [g1, p1, g2, p2, props["value"]], None)]

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        raise ValueError(f"Invalid references or value for {self.name}: {refs=} ({res1=}, {res2=}), {props=}")

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != self.name:
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        refs = []
        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        r1 = geoID_to_primitive.get((g1, p1))
        if r1:
            refs.append(r1)

        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos
        if g2 >= 0:  # g2 == -1 means no second entity (line length)
            r2 = geoID_to_primitive.get((g2, p2))
            if r2:
                refs.append(r2)

        val = fc_constraint.Value

        # 1 ref = Length, 2 refs = Distance
        if len(refs) == 1:
            return self.create_constraint("Length", tuple(refs), val, direction="MINIMUM")
        elif len(refs) == 2:
            return self.create_constraint(self.name, tuple(refs), val, direction="MINIMUM")

        raise ValueError(f"Invalid references or value for {self.name}: {refs=}, {val=}")

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2
