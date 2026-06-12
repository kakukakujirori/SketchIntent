"""Shared configuration for the paper-evaluation pipeline."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_EVAL_ROOT = REPO_ROOT / "data_eval"

ANNOTATION_LABELS = frozenset({"D", "E", "J", "R", "S", "T"})

# Edit these paths when model inference outputs are available. Each root must
# mirror data_eval, i.e. contain 0-11 subdirectories with <sketch_id>.py files.
MODEL_OUTPUT_ROOTS = {
    "base": REPO_ROOT / "model_outputs" / "base",
    "solver_rl": REPO_ROOT / "model_outputs" / "solver_rl",
}

DIMENSIONAL_DISPLAY_TYPES = {
    "Distance",
    "DistanceX",
    "DistanceY",
    "Length",
    "Angle",
    "Diameter",
    "Radius",
}

BASE_POLICY = "canonical_solid"


def data_root() -> Path:
    """Return the canonical completed evaluation dataset root."""
    return DATA_EVAL_ROOT


def model_specs() -> list[tuple[str, str]]:
    """Return configured model prediction roots for build_file_bundle."""
    return [(label, str(path)) for label, path in MODEL_OUTPUT_ROOTS.items()]


def missing_model_roots() -> dict[str, Path]:
    """Return configured model roots that do not currently exist."""
    return {label: path for label, path in MODEL_OUTPUT_ROOTS.items() if not path.exists()}
