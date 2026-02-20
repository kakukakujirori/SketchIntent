from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class MidpointHandler(ConstraintHandler):
    """
    Midpoint indicates that a point lies at the midpoint of a line.

    In SketchGraphs, this is stored as Midpoint with 2 refs:
        - ref0: the point
        - ref1: the line
    or 3 refs:
        - ref0: one endpoint of the line
        - ref1: the other endpoint of the line
        - ref2: the midpoint

    In FreeCAD, this is implemented as:
        - Symmetric(g_line, 1, g_line, 2, g_mid_point, p_mid_point) — start/end of line symmetric about midpoint
        - PointOnObject(g_mid_point, p_mid_point, g_line) — optionally ensure midpoint is on line
    """

    @property
    def token_id(self) -> int:
        return ConstraintToken.Midpoint

    @property
    def name(self) -> str:
        return "Midpoint"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)

        if len(refs) == 2:
            # 2-ref form: point + line
            # Determine which is the point and which is the line
            res0 = primitive_geoID.get(refs[0])
            res1 = primitive_geoID.get(refs[1])
            if not res0 or not res1:
                return []

            g0, p0 = res0
            g1, p1 = res1

            # Identify point vs line: the one with pid>0 is the point
            if p0 > 0 and p1 == 0:
                mid_g, mid_p = g0, p0
                line_g = g1
            elif p1 > 0 and p0 == 0:
                mid_g, mid_p = g1, p1
                line_g = g0
            else:
                return []

            # Symmetric: line's start and end are symmetric about the midpoint
            # Symmetric(line_g, 1, line_g, 2, mid_g, mid_p) — 6 args
            return [('Symmetric', [line_g, 1, line_g, 2, mid_g, mid_p], None)]

        elif len(refs) == 3:
            # 3-ref form: two endpoints + midpoint
            res0 = primitive_geoID.get(refs[0])
            res1 = primitive_geoID.get(refs[1])
            res2 = primitive_geoID.get(refs[2])
            if not res0 or not res1 or not res2:
                return []

            g0, p0 = res0
            g1, p1 = res1
            g2, p2 = res2

            # Symmetric(endpoint1_g, endpoint1_p, endpoint2_g, endpoint2_p, mid_g, mid_p)
            return [('Symmetric', [g0, p0, g1, p1, g2, p2], None)]

        return []

    def from_freecad(self, fc_constraint, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        # Midpoint comes in as Symmetric where First and Second are start/end of same line
        if fc_constraint.Type != 'Symmetric':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        g3, p3 = fc_constraint.Third, fc_constraint.ThirdPos

        # Check if it's a midpoint: First and Second are endpoints of same line
        if g1 == g2 and p1 == 1 and p2 == 2:
            # This is Midpoint(point on line)
            r_point = geoID_to_primitive.get((g3, p3))
            r_line = geoID_to_primitive.get((g1, 0))
            if r_point and r_line:
                return self.create_constraint('Midpoint', (r_point, r_line))

        return None

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) >= 2
