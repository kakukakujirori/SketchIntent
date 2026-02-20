import enum

class SpecialToken(enum.IntEnum):
    Pad = 0
    Mask = 1
    Start = 2
    Stop = 3

class PrimitiveToken(enum.IntEnum):
    # Base Token Types
    Pad = SpecialToken.Pad
    Mask = SpecialToken.Mask
    Start = SpecialToken.Start
    Stop = SpecialToken.Stop

    # Types
    Line = 4
    Point = 5
    Circle = 6
    Arc = 7

class PrimitiveAttrToken(enum.IntEnum):
    # Base Token Types
    Pad = SpecialToken.Pad
    Mask = SpecialToken.Mask

    # Semantic Labels for Coords
    Type = 2
    Construction = 3
    StartX = 4
    StartY = 5
    MidX = 6
    MidY = 7
    EndX = 8
    EndY = 9
    CenterX = 10
    CenterY = 11
    Radius = 12

class ConstraintToken(enum.IntEnum):
    # Base Token Types
    Pad = SpecialToken.Pad
    Mask = SpecialToken.Mask
    Start = SpecialToken.Start
    Stop = SpecialToken.Stop

    # Geometric constraints (no numerical parameters)
    Coincident = 4
    Concentric = 5
    Equal = 6
    Fix = 7  # FIXME: BETTER TO REMOVE?
    Horizontal = 8
    Midpoint = 9
    Mirror = 10
    Normal = 11
    Offset = 12
    Parallel = 13
    Perpendicular = 14
    Quadrant = 15
    Tangent = 16
    Vertical = 17

    # Dimension constraints (numerical parameters not tokenized, only references)
    Angle = 18
    Diameter = 19
    Distance = 20
    DistanceX = 21
    DistanceY = 22
    Length = 23
    Radius = 24
    Symmetric = 25

    # Special tokens
    InnerPad = 26  # Padding within fixed-length constraint blocks (loss computed)

class ConstraintAttrToken(enum.IntEnum):
    Pad = SpecialToken.Pad    # Padding
    Mask = SpecialToken.Mask  # Mask (unused but included for consistency)
    Type = 2                  # Constraint type token (Coincident, Parallel, etc.)
    Reference = 3             # Primitive position reference
    InnerPad = 4              # Padding within fixed-length constraint blocks

# Offsets
PRIMITIVE_VAL_OFFSET = len(PrimitiveToken)
CONSTRAINT_VAL_OFFSET = len(ConstraintToken)
