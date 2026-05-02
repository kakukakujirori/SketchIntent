from .accuracy import TokenAccuracy
from .constraint_iou import ConstraintIoU
from .constraint_bleu import ConstraintBLEU

__all__ = [
    "TokenAccuracy",
    "ConstraintIoU",
    "ConstraintBLEU",
    "compute_null_space_distance",
]

# Lazy import to avoid torchmetrics dependency when only null_space_distance is needed
def __getattr__(name):
    if name == "compute_null_space_distance":
        from .null_space_distance import compute_null_space_distance
        return compute_null_space_distance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")