"""Token definitions for img2cad_diffusion models.

This module centralizes all token definitions for both primitive and constraint models,
eliminating duplicate Token class definitions that previously existed in dataset.py
and constraint_data.py.
"""

import enum
from typing import Dict, List

from sketchgraphs.data import Arc, Circle, Line, Point


# =============================================================================
# Coordinate Token Constants
# =============================================================================

NON_COORD_TOKEN = 1  # 0 is reserved for padding

# Number of parameters per entity type
NUM_PARAMS: Dict[type, int] = {
    Arc: 6,
    Circle: 3,
    Line: 4,
    Point: 2,
}

# Build COORD_TOKEN_MAP for primitive entities
PRIMITIVE_COORD_TOKEN_MAP: Dict[type, List[int]] = {}
_tok = NON_COORD_TOKEN + 1
for _ent_type in [Arc, Circle, Line, Point]:
    PRIMITIVE_COORD_TOKEN_MAP[_ent_type] = list(range(_tok, _tok + NUM_PARAMS[_ent_type]))
    _tok += NUM_PARAMS[_ent_type]

# Constraint coordinate tokens (reference parameter types)
CONSTRAINT_COORD_TOKENS = [NON_COORD_TOKEN + 1, NON_COORD_TOKEN + 2]  # [2, 3]


# =============================================================================
# Token Enumerations
# =============================================================================

class PrimitiveToken(enum.IntEnum):
    """Token values for primitive entity models.

    These tokens represent the non-parameter value tokens used in the
    primitive sequence representation.
    """

    Pad = 0
    Start = 1
    Stop = 2
    Arc = 3
    Circle = 4
    Line = 5
    Point = 6


class ConstraintToken(enum.IntEnum):
    """Token values for constraint models.

    These tokens represent the non-parameter value tokens used in the
    constraint sequence representation, including geometric constraint types.
    """

    Pad = 0
    Mask = 1
    Start = 2
    Stop = 3
    Coincident = 4
    Concentric = 5
    Equal = 6
    Fix = 7
    Horizontal = 8
    Midpoint = 9
    Normal = 10
    Offset = 11
    Parallel = 12
    Perpendicular = 13
    Quadrant = 14
    Tangent = 15
    Vertical = 16
