"""Token definitions for img2cad_diffusion models.

This module centralizes all token definitions for both primitive and constraint models,
eliminating duplicate Token class definitions that previously existed in dataset.py
and constraint_data.py.
"""
import enum

from sketchgraphs.data import Arc, Circle, Line, Point


# =============================================================================
# Token Enumerations
# =============================================================================

class SpecialToken(enum.IntEnum):
    """Base special tokens. All Token classes should include these at the start.
    """
    Pad = 0   # Universal padding token
    Mask = 1  # Universal mask token (for diffusion)


class PrimitiveToken(enum.IntEnum):
    """Token values for primitive entity models.

    These tokens represent the non-parameter value tokens used in the
    primitive sequence representation.
    """
    Pad = SpecialToken.Pad    # Padding
    Mask = SpecialToken.Mask  # Mask (unused but included for consistency)
    Start = 2
    Stop = 3
    Arc = 4
    Circle = 5
    Line = 6
    Point = 7


class ConstraintToken(enum.IntEnum):
    """Token values for constraint models.

    These tokens represent the non-parameter value tokens used in the
    constraint sequence representation, including geometric constraint types.
    """

    Pad = SpecialToken.Pad    # References shared constant
    Mask = SpecialToken.Mask  # References shared constant
    Start = 2
    Stop = 3
    # Geometric constraints (no numerical parameters)
    Coincident = 4
    Concentric = 5
    Equal = 6
    Fix = 7  # FIXME: BETTER TO REMOVE?
    Horizontal = 8
    Midpoint = 9
    Normal = 10
    Offset = 11
    Parallel = 12
    Perpendicular = 13
    Quadrant = 14
    Tangent = 15
    Vertical = 16
    # Dimension constraints (numerical parameters not tokenized, only references)
    Angle = 17
    Diameter = 18
    Distance = 19
    Length = 20
    Radius = 21
    # Special tokens
    InnerPad = 22  # Padding within fixed-length constraint blocks (loss computed)

    @classmethod
    def has_value(cls, value):
        return value in [i.value for i in cls]


# Mapping from constraint type to acceptable number of references
# Used for validation during detokenization
CONSTRAINT_ACCEPTABLE_REF_COUNTS: dict[ConstraintToken, set[int]] = {
    ConstraintToken.Pad: {0},
    ConstraintToken.Mask: {0},
    ConstraintToken.Start: {0},
    ConstraintToken.Stop: {0},
    ConstraintToken.Coincident: {2},
    ConstraintToken.Concentric: {2},
    ConstraintToken.Equal: {2},
    ConstraintToken.Fix: {1},
    ConstraintToken.Horizontal: {1, 2},  # Can be 1 (line) or 2 (two points)
    ConstraintToken.Midpoint: {2, 3},    # 2 (point-curve) or 3 (point-point-point)
    ConstraintToken.Normal: {2},
    ConstraintToken.Offset: {2},
    ConstraintToken.Parallel: {2},
    ConstraintToken.Perpendicular: {2},
    ConstraintToken.Quadrant: {2},
    ConstraintToken.Tangent: {2},
    ConstraintToken.Vertical: {1, 2},    # Can be 1 (line) or 2 (two points)
    # Dimension constraints
    ConstraintToken.Angle: {2},          # Two lines/edges
    ConstraintToken.Diameter: {1},       # Circle or arc
    ConstraintToken.Distance: {1, 2},    # 1 (line length) or 2 (point-point, point-line, line-line)
    ConstraintToken.Length: {1},         # Line
    ConstraintToken.Radius: {1},         # Circle or arc
    # Special tokens
    ConstraintToken.InnerPad: {0},
}

# =============================================================================
# Parameter Semantic Tokens (coord_tokens)
# =============================================================================

class PrimitiveParamToken(enum.IntEnum):
    """Semantic tokens for primitive parameter types in coord_tokens.

    These tokens indicate what semantic meaning each value token has for primitives.
    """
    Pad = SpecialToken.Pad    # Padding
    Mask = SpecialToken.Mask  # Mask (unused but included for consistency)
    Type = 2                  # Entity type token (Arc, Line, etc.)
    Clockwise = 3             # Arc clockwise attribute
    Construction = 4          # isConstruction attribute
    # >>> COORDINATE PARAMETERS >>>
    StartX = 5
    StartY = 6
    EndX = 7
    EndY = 8
    CenterX = 9
    CenterY = 10
    Radius = 11
    # <<< COORDINATE PARAMETERS <<<


class ConstraintParamToken(enum.IntEnum):
    """Semantic tokens for constraint parameter types in coord_tokens.

    These tokens indicate what semantic meaning each value token has for constraints.
    """
    Pad = SpecialToken.Pad    # Padding
    Mask = SpecialToken.Mask  # Mask (unused but included for consistency)
    Type = 2                  # Constraint type token (Coincident, Parallel, etc.)
    Position = 3              # Primitive position reference
    InnerPad = 4              # Padding within fixed-length constraint blocks


# Number of parameters per entity type (TODO: REMOVE AFTER REFACTORING)
NUM_PARAMS: dict[type, int] = {
    Arc: 6,
    Circle: 3,
    Line: 4,
    Point: 2,
}