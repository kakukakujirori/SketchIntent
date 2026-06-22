"""Shared configuration for the paper-evaluation pipeline."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_EVAL_ROOT = REPO_ROOT / "data_eval"

ANNOTATION_LABELS = frozenset({"D", "E", "J", "R", "S", "T"})

# Edit these paths when model inference outputs are available. Each root must
# mirror data_eval, i.e. contain 0-11 subdirectories with <sketch_id>.py files.
# Sec.5 model rows. solver_rl -> DimParsimony c1 at its converged best step 70000
# (single-shot / iter1; the run stopped at 70k with FC 0.750, DoF 4.64 = its max
# operating point, NOT matched to Ours). Previous Sec.5 numbers used an older
# solver-only ("c0") iter5 output at model_outputs/solver_rl.
MODEL_OUTPUT_ROOTS = {
    "base": REPO_ROOT / "model_outputs" / "base",
    "solver_rl": REPO_ROOT / "model_outputs"
    / "Vitruvion_scaffold_v12_ablA_noML_dimprior_dimpars_c1_step_70000",
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
