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

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        """
        Tangent residual. Dispatch on entity types:

        Edge-Edge (p=0):
          Line-Circle/Arc:  dist(center, line) - r₀
          Arc-Arc external: ‖c1 - c2‖ - (r1₀ + r2₀)
          Arc-Arc internal: ‖c1 - c2‖ - |r1₀ - r2₀|
          Line-Line:        should not appear (converted to Coincident in from_freecad)

        Point-specified (p>0):
          Arc end-point tangent to Line: dot(c - p, dir_line) / ‖dir_line‖
        """
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) < 2:
            return torch.zeros(1, dtype=state.dtype)

        ref1, ref2 = refs[0], refs[1]
        eid1 = ref1.split('.')[0]
        eid2 = ref2.split('.')[0]
        e1 = state.sketch.entities.get(eid1)
        e2 = state.sketch.entities.get(eid2)

        has_sub1 = '.' in ref1  # point-specified if True
        has_sub2 = '.' in ref2

        # ---- Edge-Edge case ----
        if not has_sub1 and not has_sub2:
            is_line1 = isinstance(e1, datalib.Line)
            is_line2 = isinstance(e2, datalib.Line)
            is_arc1 = isinstance(e1, (datalib.Circle, datalib.Arc))
            is_arc2 = isinstance(e2, (datalib.Circle, datalib.Arc))

            if is_line1 and is_arc2:
                # dist(center2, line1) - r2
                c = state.get_point(eid2 + '.center')
                s = state.get_point(eid1 + '.start')
                e = state.get_point(eid1 + '.end')
                d = e - s
                # signed distance from c to infinite line through s with dir d
                # n = perp(d) = (-d[1], d[0]); dist = dot(c - s, n) / ‖d‖
                n = torch.stack([-d[1], d[0]])
                dist = torch.dot(c - s, n) / (torch.linalg.norm(d) + 1e-15)
                r = state.get_radius(eid2)
                return (dist - r).unsqueeze(0)

            elif is_arc1 and is_line2:
                c = state.get_point(eid1 + '.center')
                s = state.get_point(eid2 + '.start')
                e = state.get_point(eid2 + '.end')
                d = e - s
                n = torch.stack([-d[1], d[0]])
                dist = torch.dot(c - s, n) / (torch.linalg.norm(d) + 1e-15)
                r = state.get_radius(eid1)
                return (dist - r).unsqueeze(0)

            elif is_arc1 and is_arc2:
                c1 = state.get_point(eid1 + '.center')
                c2 = state.get_point(eid2 + '.center')
                r1 = state.get_radius(eid1)
                r2 = state.get_radius(eid2)
                dist_cc = torch.linalg.norm(c1 - c2)
                r1_val = float(e1.radius)  # type: ignore[union-attr]
                r2_val = float(e2.radius)  # type: ignore[union-attr]
                # Determine external vs internal from current geometry
                import math
                import numpy as np
                c1_np = np.array([e1.xCenter, e1.yCenter])  # type: ignore[union-attr]
                c2_np = np.array([e2.xCenter, e2.yCenter])  # type: ignore[union-attr]
                d_np = np.linalg.norm(c1_np - c2_np)
                if d_np > r1_val + r2_val - 1e-6:  # external
                    return (dist_cc - (r1 + r2)).unsqueeze(0)
                else:  # internal
                    return (dist_cc - torch.abs(r1 - r2)).unsqueeze(0)

            else:
                # Line-Line tangent should not appear
                return torch.zeros(1, dtype=state.dtype)

        # ---- Point-specified case ----
        # Most common: arc endpoint tangent to line
        # The point constraint enforces that the arc end is on the line;
        # here we enforce tangency: dot(center - point, line_dir) = 0
        if has_sub1:
            pt = state.get_point(ref1)  # arc endpoint
            arc_center = state.get_point(eid1 + '.center')
            # eid2 should be a line
            s = state.get_point(eid2 + '.start')
            e = state.get_point(eid2 + '.end')
            d = e - s
            radial = arc_center - pt
            dot = radial[0] * d[0] + radial[1] * d[1]
            return dot.unsqueeze(0)
        else:
            pt = state.get_point(ref2)
            arc_center = state.get_point(eid2 + '.center')
            s = state.get_point(eid1 + '.start')
            e = state.get_point(eid1 + '.end')
            d = e - s
            radial = arc_center - pt
            dot = radial[0] * d[0] + radial[1] * d[1]
            return dot.unsqueeze(0)

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

        if p1 > 0 and p2 > 0:
            # Tangent with point specification: Tangent(e1, s1, e2, s2)
            return [('Tangent', [g1, p1, g2, p2], datum)]
        elif p1 == 0 and p2 > 0:
            # Tangent with one point specification must decrate the point first
            return [('Tangent', [g2, p2, g1, p1], datum)]
        elif p2 == 0 and p1 > 0:
            # Tangent with one point specification must decrate the point first
            return [('Tangent', [g1, p1, g2, p2], datum)]
        elif p1 == 0 and p2 == 0:
            # Edge-Edge tangent: Tangent(e1, e2)
            return [('Tangent', [g1, g2], datum)]
        else:
            raise ValueError(f"Invalid point specification for {self.name}: {p1=}, {p2=}")

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
