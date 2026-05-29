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

    # Special tokens
    InnerPad = 25  # Padding within fixed-length constraint blocks (loss computed)

class ConstraintAttrToken(enum.IntEnum):
    Pad = SpecialToken.Pad    # Padding
    Mask = SpecialToken.Mask  # Mask (unused but included for consistency)
    Type = 2                  # Constraint type token (Coincident, Parallel, etc.)
    Reference = 3             # Primitive position reference
    InnerPad = 4              # Padding within fixed-length constraint blocks

# Offsets
PRIMITIVE_VAL_OFFSET = len(PrimitiveToken)
CONSTRAINT_VAL_OFFSET = len(ConstraintToken)

# Token IDs of dimensional constraints (have numerical parameters).
# Used by RLOO relational_only mode to mask these out of next_token_logits.
DIMENSIONAL_CONSTRAINT_TOKEN_IDS: tuple[int, ...] = (
    ConstraintToken.Angle,
    ConstraintToken.Diameter,
    ConstraintToken.Distance,
    ConstraintToken.DistanceX,
    ConstraintToken.DistanceY,
    ConstraintToken.Length,
    ConstraintToken.Radius,
)

# Every real constraint-type head token (relational + dimensional), i.e. the
# ConstraintToken members that can appear as a constraint block's head, with the
# special tokens (Pad/Mask/Start/Stop/InnerPad) removed.  Used for per-type head
# counting in the constraint-mix logging and the relational class prior reward.
_NON_CONSTRAINT_TYPE_TOKENS = frozenset(
    (
        ConstraintToken.Pad,
        ConstraintToken.Mask,
        ConstraintToken.Start,
        ConstraintToken.Stop,
        ConstraintToken.InnerPad,
    )
)
CONSTRAINT_TYPE_TOKEN_IDS: tuple[int, ...] = tuple(
    int(t) for t in ConstraintToken if t not in _NON_CONSTRAINT_TYPE_TOKENS
)

# "Key" relational constraints whose design intent we most want to surface:
# Coincident plus the structurally rare relations (Equal/Tangent/Parallel/
# Perpendicular/Mirror/Concentric) that the policy almost never samples on its
# own.  Concentric is included because it is a genuine, fairly common human
# relation (~3% of human-GT constraints) that the model-generated pseudo-rel GT
# lacks entirely.
KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS: tuple[int, ...] = (
    ConstraintToken.Coincident,
    ConstraintToken.Equal,
    ConstraintToken.Tangent,
    ConstraintToken.Parallel,
    ConstraintToken.Perpendicular,
    ConstraintToken.Mirror,
    ConstraintToken.Concentric,
)

# Distance-family dimensional tokens.  Near-zero values of these act as
# Coincident / alignment dimensional substitutes (see SolverReward hygiene).
DISTANCE_FAMILY_CONSTRAINT_TOKEN_IDS: tuple[int, ...] = (
    ConstraintToken.Distance,
    ConstraintToken.DistanceX,
    ConstraintToken.DistanceY,
)

# Angle-family dimensional tokens.  Values near 0/90/180 deg act as
# Parallel / Perpendicular dimensional substitutes (see SolverReward hygiene).
ANGLE_FAMILY_CONSTRAINT_TOKEN_IDS: tuple[int, ...] = (
    ConstraintToken.Angle,
)

# FreeCAD constants
FC_GEO_UNDEF = -2000
