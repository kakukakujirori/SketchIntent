from typing import Any, TYPE_CHECKING
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from src.sketch_handlers.constants import ConstraintToken, FC_GEO_UNDEF
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.registry import CONSTRAINT_HANDLERS

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


@CONSTRAINT_HANDLERS.register_module()
class HorizontalHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Horizontal

    @property
    def name(self) -> str:
        return "Horizontal"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def residual(self, constraint: datalib.Constraint, state: 'SketchState') -> 'torch.Tensor':
        import torch
        refs = self._get_ref_strings(constraint)
        if len(refs) == 1:
            # Line form: y_start == y_end
            eid = refs[0]
            s = state.get_point(eid + '.start')
            e = state.get_point(eid + '.end')
            return (s[1] - e[1]).unsqueeze(0)
        elif len(refs) == 2:
            # Two-point form
            p1 = state.get_point(refs[0])
            p2 = state.get_point(refs[1])
            return (p1[1] - p2[1]).unsqueeze(0)
        return torch.zeros(1, dtype=state.dtype)

    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        refs = self._get_ref_strings(constraint)

        if len(refs) == 1:
            # Line constraint: Horizontal(e1)
            res1 = primitive_geoID.get(refs[0])
            assert res1 is not None, f"Invalid references for {self.name}: {refs=} ({res1=})"
            return [('Horizontal', [res1[0]], None)]
        elif len(refs) == 2:
            # Two-point constraint: Horizontal(e1, s1, e2, s2)
            res1 = primitive_geoID.get(refs[0])
            res2 = primitive_geoID.get(refs[1])
            assert res1 and res2, f"Invalid references for {self.name}: {refs=} ({res1=}, {res2=})"
            g1, p1 = res1
            g2, p2 = res2
            return [('Horizontal', [g1, p1, g2, p2], None)]

        raise ValueError(f"Invalid references for {self.name}: {refs=}")

    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        if fc_constraint.Type != 'Horizontal':
            raise ValueError(f"Invalid constraint type for {self.name}: {fc_constraint.Type}")

        g1 = fc_constraint.First
        p1 = fc_constraint.FirstPos
        g2 = fc_constraint.Second
        p2 = fc_constraint.SecondPos

        # Explicitly disallow axis/origin references (geoID < 0)
        if g1 < 0 or (g2 < 0 and g2 != FC_GEO_UNDEF):
            raise ValueError(f"{self.name} references to X/Y axes or Origin (geo < 0) are not supported.")

        refs = []
        if g2 >= 0 and p1 > 0 and p2 > 0:
            # Two-point form
            r1 = geoID_to_primitive.get((g1, p1))
            r2 = geoID_to_primitive.get((g2, p2))
            if r1 and r2:
                refs = [r1, r2]
        elif g2 < 0 and p1 == 0:
            # Line form
            r1 = geoID_to_primitive.get((g1, 0))
            if r1:
                refs = [r1]
        else:
            # This covers cases like g2=-2000 (und) but p1 > 0 (point-to-origin alignment)
            raise ValueError(f"Unsupported {self.name} constraint structure: g1={g1}, p1={p1}, g2={g2}, p2={p2}")

        assert self.validate_refs(tuple(refs)), f"{refs=}"

        return self.create_constraint(self.name, tuple(refs))

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        return len(refs) == 1 or len(refs) == 2
