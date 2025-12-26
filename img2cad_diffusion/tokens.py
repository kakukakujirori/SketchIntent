"""Token definitions for img2cad_diffusion models.

This module centralizes all token definitions for both primitive and constraint models,
eliminating duplicate Token class definitions that previously existed in dataset.py
and constraint_data.py.
"""

import enum
from typing import Dict, List

from sketchgraphs.data import Arc, Circle, Line, Point


# Generic special tokens (shared across Primitive/Constraint)
class SpecialToken(enum.IntEnum):
    Pad = 0   # Universal padding token
    Mask = 1  # Universal mask token (for diffusion)


# =============================================================================
# Parameter Semantic Tokens (coord_tokens)
# =============================================================================

class PrimitiveParamToken(enum.IntEnum):
    """Semantic tokens for primitive parameter types in coord_tokens.

    These tokens indicate what semantic meaning each value token has for primitives.
    """
    Pad = SpecialToken.Pad   # Padding
    Type = 1                 # Entity type token (Arc, Line, etc.)
    Clockwise = 2            # Arc clockwise attribute
    Construction = 3         # isConstruction attribute
    # >>> COORDINATE PARAMETERS >>>
    StartX = 4
    StartY = 5
    EndX = 6
    EndY = 7
    CenterX = 8
    CenterY = 9
    Radius = 10
    # <<< COORDINATE PARAMETERS <<<


class ConstraintParamToken(enum.IntEnum):
    """Semantic tokens for constraint parameter types in coord_tokens.

    These tokens indicate what semantic meaning each value token has for constraints.
    """
    Pad = SpecialToken.Pad   # Padding
    Type = 1                 # Constraint type token (Coincident, Parallel, etc.)
    Position = 2             # Primitive position reference


# Number of parameters per entity type (TODO: REMOVE AFTER REFACTORING)
NUM_PARAMS: Dict[type, int] = {
    Arc: 6,
    Circle: 3,
    Line: 4,
    Point: 2,
}


# =============================================================================
# Token Enumerations
# =============================================================================

class PrimitiveToken(enum.IntEnum):
    """Token values for primitive entity models.

    These tokens represent the non-parameter value tokens used in the
    primitive sequence representation.
    """

    Pad = SpecialToken.Pad  # References shared constant
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

    Pad = SpecialToken.Pad    # References shared constant
    Mask = SpecialToken.Mask  # References shared constant
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


# =============================================================================

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class TokenVocabulary:
    """Centralized token vocabulary for primitive and constraint models.

    PAD and MASK are shared across both primitive and constraint models.
    All token indices are pre-computed based on num_bins for consistency.

    Attributes
    ----------
    num_bins : int
        Number of quantization bins for coordinate values.
    """
    num_bins: int

    # Shared tokens (universal across primitive/constraint)
    PAD: ClassVar[int] = 0   # Universal padding
    MASK: ClassVar[int] = 1  # Universal mask (for diffusion)

    # Computed token ranges (set in __post_init__)
    _primitive_special_start: int = field(init=False)
    _primitive_coord_start: int = field(init=False)
    _primitive_construction_start: int = field(init=False)
    _primitive_clockwise_start: int = field(init=False)

    def __post_init__(self):
        """Compute all token ranges based on num_bins."""
        # Primitive tokens: [Pad, Start, Stop, Arc, Circle, Line, Point] = 7 tokens
        self._primitive_special_start = 0
        num_primitive_special = len(PrimitiveToken)  # 7

        # Coordinate bins: [0 ... num_bins-1]
        self._primitive_coord_start = num_primitive_special

        # Construction tokens: [True, False]
        self._primitive_construction_start = num_primitive_special + self.num_bins

        # Clockwise tokens (Arc only): [True, False]
        self._primitive_clockwise_start = self._primitive_construction_start + 2

    @property
    def primitive_vocab_size(self) -> int:
        """Total vocabulary size for primitive model."""
        # special + bins + construction(2) + clockwise(2)
        return len(PrimitiveToken) + self.num_bins + 2 + 2

    @property
    def constraint_vocab_size(self) -> int:
        """Total vocabulary size for constraint model.

        Constraint model uses pointer network, so vocab includes:
        - Fixed constraint tokens (Pad, Mask, Start, Stop, constraint types)
        - Dynamic tokens from primitives (length varies)
        """
        return len(ConstraintToken)

    def coord_to_token(self, bin_idx: int) -> int:
        """Convert coordinate bin index [0, num_bins-1] to token index."""
        return self._primitive_coord_start + bin_idx

    def token_to_coord(self, token: int) -> int:
        """Convert token index back to coordinate bin index."""
        return token - self._primitive_coord_start

    def construction_token(self, is_construction: bool) -> int:
        """Get token for isConstruction attribute."""
        return self._primitive_construction_start + (0 if is_construction else 1)

    def clockwise_token(self, is_clockwise: bool) -> int:
        """Get token for clockwise attribute (Arc only)."""
        return self._primitive_clockwise_start + (0 if is_clockwise else 1)

    def is_coord_token(self, token: int) -> bool:
        """Check if token is a coordinate bin token."""
        return self._primitive_coord_start <= token < self._primitive_construction_start

    def is_construction_token(self, token: int) -> bool:
        """Check if token is a construction attribute token."""
        return self._primitive_construction_start <= token < self._primitive_clockwise_start

    def is_clockwise_token(self, token: int) -> bool:
        """Check if token is a clockwise attribute token."""
        return self._primitive_clockwise_start <= token < self._primitive_clockwise_start + 2

    @classmethod
    def from_config(cls, num_bins: int) -> 'TokenVocabulary':
        """Create a TokenVocabulary from configuration."""
        return cls(num_bins=num_bins)

