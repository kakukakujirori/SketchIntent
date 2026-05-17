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
class EqualHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Equal

    @property
    def name(self) -> str:
        return "Equal"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) < 2:
            return torch.zeros(1, dtype=state.dtype)
        eid1 = refs[0].split('.')[0]
        eid2 = refs[1].split('.')[0]
        e1 = state.sketch.entities.get(eid1)
        e2 = state.sketch.entities.get(eid2)
        is_curve1 = isinstance(e1, (datalib.Circle, datalib.Arc))
        is_curve2 = isinstance(e2, (datalib.Circle, datalib.Arc))
        is_line1 = isinstance(e1, datalib.Line)
        is_line2 = isinstance(e2, datalib.Line)

        if is_curve1 and is_curve2:
            r1 = state.get_radius(eid1)
            r2 = state.get_radius(eid2)
            return (r1 - r2).unsqueeze(0)
        elif is_line1 and is_line2:
            # Line lengths: stabilized norm difference
            s1 = state.get_point(eid1 + '.start')
            e1p = state.get_point(eid1 + '.end')
            s2 = state.get_point(eid2 + '.start')
            e2p = state.get_point(eid2 + '.end')
            len1 = torch.sqrt(torch.sum((e1p - s1)**2) + 1e-12)
            len2 = torch.sqrt(torch.sum((e2p - s2)**2) + 1e-12)
            return (len1 - len2).unsqueeze(0)
        else:
            raise ValueError(f"Equal constraint between incompatible types: {type(e1)} and {type(e2)}")

    def apply_jacobian_surrogate(self, constraint, state, jac):
        """
        Inject a directional surrogate into the dir-component columns when the
        line is axis-aligned and the autodiff gradient there vanishes. Mirrors
        FreeCAD's ConstraintEqualLineLength::errorgrad surrogate, adapted to our
        6-param Line representation [pntX, pntY, dirX, dirY, s, e].

        Only applied between two Lines; circle radii give a clean ±1 gradient
        and need no surrogate.

        Surrogate magnitude: 1e-4, threshold: 1e-6 (100× above the
        test_solver_consistency SVD atol=1e-6, so matrix_rank picks it up;
        FreeCAD itself uses 1e-10 paired with a 1e-13 pivot threshold —
        same 3+ order separation).
        """
        refs = self._get_ref_strings(constraint)
        if len(refs) != 2:
            return jac
        eid1 = refs[0].split('.')[0]
        eid2 = refs[1].split('.')[0]
        e1 = state.sketch.entities.get(eid1)
        e2 = state.sketch.entities.get(eid2)
        if not (isinstance(e1, datalib.Line) and isinstance(e2, datalib.Line)):
            return jac

        threshold = 1e-6
        surrogate = 1e-4
        jac = jac.clone()

        for line_idx, eid in enumerate((eid1, eid2)):
            # residual = len1 - len2  →  +1 for line1, -1 for line2
            sign_factor = +1.0 if line_idx == 0 else -1.0
            sl = state.slices.get(eid)
            if sl is None:
                continue
            dirX_col = sl.start + 2
            dirY_col = sl.start + 3
            dirX = float(state.q[dirX_col])
            dirY = float(state.q[dirY_col])
            if abs(float(jac[0, dirX_col])) < threshold:
                sign = +1.0 if dirX >= 0.0 else -1.0
                jac[0, dirX_col] = sign_factor * sign * surrogate
            if abs(float(jac[0, dirY_col])) < threshold:
                sign = +1.0 if dirY >= 0.0 else -1.0
                jac[0, dirY_col] = sign_factor * sign * surrogate

        return jac

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)
        if not self.validate_refs(refs):
            raise ValueError(f"Invalid number of references for {self.name}: {refs}")

        res1 = primitive_geoID.get(refs[0])
        res2 = primitive_geoID.get(refs[1])
        if not res1 or not res2:
            raise ValueError(f"Invalid references for {self.name}: {refs}")

        # FreeCAD wrongly accepts Equal(A, A) as DoF-reducing; reject here.
        if refs[0] == refs[1]:
            raise ValueError(f"Tautological {self.name}({refs[0]}, {refs[1]})")

        # Equal(e1, e2) — edge-edge only
        return [('Equal', [res1[0], res2[0]], None)]

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Equal':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        g2 = fc_constraint.Second
        r1 = geoID_to_primitive.get((g1, 0))
        r2 = geoID_to_primitive.get((g2, 0))
        if not r1 or not r2:
            return None

        return self.create_constraint(self.name, (r1, r2))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 2
