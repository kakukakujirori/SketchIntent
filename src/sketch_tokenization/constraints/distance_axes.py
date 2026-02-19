from .distance import DistanceHandler
from ..registry import CONSTRAINT_HANDLERS
from ..constants import ConstraintToken
import sketchgraphs.data as datalib
import numpy as np

@CONSTRAINT_HANDLERS.register_module()
class DistanceXHandler(DistanceHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.DistanceX

    @property
    def name(self) -> str:
        return "DistanceX"

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint("Distance", refs, val)

    def create_constraint(self, constraint_type_name: str, refs: tuple[str, ...], value: float | None = None) -> datalib.Constraint:
        c = super().create_constraint(constraint_type_name, refs, value)
        from sketchgraphs.data._constraint import EnumParameter, DirectionValue
        c.parameters.append(EnumParameter('direction', DirectionValue.HORIZONTAL))
        return c

@CONSTRAINT_HANDLERS.register_module()
class DistanceYHandler(DistanceHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.DistanceY

    @property
    def name(self) -> str:
        return "DistanceY"

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        refs = tuple(tokenpos_to_primitive[idx] for idx in ref_indices)
        val = float(params[0]) if len(params) > 0 else 0.0
        return self.create_constraint("Distance", refs, val)

    def create_constraint(self, constraint_type_name: str, refs: tuple[str, ...], value: float | None = None) -> datalib.Constraint:
        c = super().create_constraint(constraint_type_name, refs, value)
        from sketchgraphs.data._constraint import EnumParameter, DirectionValue
        c.parameters.append(EnumParameter('direction', DirectionValue.VERTICAL))
        return c
