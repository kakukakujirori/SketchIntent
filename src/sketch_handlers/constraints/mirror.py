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
class MirrorHandler(ConstraintHandler):
    """
    Mirror / Symmetric: Two points are symmetric about an axis.

    In SketchGraphs: Mirror with 3 refs:
        - ref0: point on entity1
        - ref1: point on entity2
        - ref2: axis (line edge or point)

    In FreeCAD: Symmetric(e1, s1, e2, s2, axis_g [, axis_p])
    """

    @property
    def token_id(self) -> int:
        return ConstraintToken.Mirror

    @property
    def name(self) -> str:
        return "Mirror"

    def _sort_ref_tokens(self, ref_tokens: tuple[int, ...]) -> tuple[int, ...]:
        if len(ref_tokens) == 3:
            sorted_refs = sorted(ref_tokens[:2]) + [ref_tokens[2]]
            return tuple(sorted_refs)
        return tuple(sorted(ref_tokens))

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) < 3:
            return torch.zeros(2, dtype=state.dtype)

        # Determine axis parameters
        axis_ref = refs[2]
        axis_eid = axis_ref.split('.')[0]
        axis_entity = state.sketch.entities.get(axis_eid)

        if not isinstance(axis_entity, datalib.Line) or '.' in axis_ref:
            raise ValueError(f"Mirror axis must be a Line (edge), but got {axis_ref}. For point symmetry, use Midpoint.")

        s_axis = state.get_point(axis_eid + '.start')
        e_axis = state.get_point(axis_eid + '.end')
        d_axis = e_axis - s_axis
        d_axis = d_axis / (torch.linalg.norm(d_axis) + 1e-15)
        n_axis = torch.stack([-d_axis[1], d_axis[0]])

        # Point-point symmetry (2 DoF)
        p1 = state.get_point(refs[0])
        p2 = state.get_point(refs[1])

        # 1. Midpoint must lie on the axis (1 DoF)
        mid = (p1 + p2) * 0.5
        dist = torch.dot(mid - s_axis, n_axis)

        # 2. Vector p2-p1 must be parallel to axis normal (perpendicular to axis direction) (1 DoF)
        para = torch.dot(p2 - p1, d_axis)

        return torch.stack([dist, para])

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of refs for {self.name}: {len(refs)}")

        res0 = primitive_geoID.get(refs[0])
        res1 = primitive_geoID.get(refs[1])
        res2 = primitive_geoID.get(refs[2])
        if not res0 or not res1 or not res2:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {refs[0]}, {refs[1]}, {refs[2]}")

        g0, p0 = res0
        g1, p1 = res1
        g2, p2 = res2

        # Determine which ref is the axis (edge with pid=0)
        if p2 == 0:
            # ref2 is axis (edge)
            return [('Symmetric', [g0, p0, g1, p1, g2], None)]
        elif p0 == 0:
            # ref0 is axis (edge)
            return [('Symmetric', [g1, p1, g2, p2, g0], None)]
        elif p1 == 0:
            # ref1 is axis (edge)
            return [('Symmetric', [g0, p0, g2, p2, g1], None)]
        else:
            # All refs are points — axis point case, put ref2 as axis
            return [('Symmetric', [g0, p0, g1, p1, g2, p2], None)]

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Symmetric':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1, p1 = fc_constraint.First, fc_constraint.FirstPos
        g2, p2 = fc_constraint.Second, fc_constraint.SecondPos
        g3, p3 = fc_constraint.Third, fc_constraint.ThirdPos

        r1 = geoID_to_primitive.get((g1, p1))
        r2 = geoID_to_primitive.get((g2, p2))
        r3 = geoID_to_primitive.get((g3, p3 if p3 > 0 else 0))

        if not r1 or not r2 or not r3:
            raise ValueError(f"Invalid geometry IDs for {self.name}: {g1}, {g2}, {g3}")

        return self.create_constraint('Mirror', (r1, r2, r3))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 3
