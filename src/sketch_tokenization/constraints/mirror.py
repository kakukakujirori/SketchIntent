from typing import Any
import numpy as np
import sketchgraphs.data as datalib
from ..constants import ConstraintToken
from .base import ConstraintHandler
from ..registry import CONSTRAINT_HANDLERS

@CONSTRAINT_HANDLERS.register_module()
class MirrorHandler(ConstraintHandler):
    @property
    def token_id(self) -> int:
        return ConstraintToken.Mirror

    @property
    def name(self) -> str:
        return "Mirror"

    def encode(self, constraint: datalib.Constraint, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        ref_tokens = self._get_sorted_ref_tokens(constraint, primitive_tokenpos, entities)

        # Mirror Logic: 2-ref Mirror usually refers to external entities, so drop it
        if len(ref_tokens) != 3:
            raise NotImplementedError(f"Mirror constraint with {len(ref_tokens)} refs (expected 3)")

        return self.token_id, ref_tokens, np.array([]), ()
    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, token_pos_to_id_map: dict[int, str]) -> datalib.Constraint:
        refs = tuple(token_pos_to_id_map[idx] for idx in ref_indices)
        return self.create_constraint(self.name, refs)
