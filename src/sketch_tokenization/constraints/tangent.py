from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from .base import ConstraintHandler
from ..constants import ConstraintToken

from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class TangentHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Tangent

    @property
    def name(self) -> str:
        return "Tangent"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)
        return self.token_id, ref_tokens, np.array([]), ()

    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)

    def validate_refs(self, ref_tokens: tuple[int, ...]) -> bool:
        return len(ref_tokens) == 2
