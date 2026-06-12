"""Build canonical-remapped FileBundles for metric scripts."""
from __future__ import annotations

import sys

from .canonical import canonical_dir_for
from .config import REPO_ROOT, data_root, model_specs
from .file_bundle import build_file_bundle

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_bundle(
    batch: str,
    base_fcstd: str,
    files: dict[str, str],
    converter,
    *,
    geometry_tol: float = 5e-1,
):
    """Load one sketch's annotations and remap them to canonical geometry."""
    batch_dir = data_root() / batch
    humans = [(init, str(batch_dir)) for init in sorted(files)]
    file_maps = {init: {base_fcstd: actual} for init, actual in files.items()}
    canonical_dir = canonical_dir_for(batch, base_fcstd)
    if canonical_dir is None:
        raise FileNotFoundError(f"canonical .py not found for {batch}/{base_fcstd}")
    return build_file_bundle(
        batch=batch,
        filename=base_fcstd,
        humans=humans,
        file_maps=file_maps,
        converter=converter,
        geometry_tol=geometry_tol,
        canonical_dir=str(canonical_dir),
        model_specs=model_specs(),
        coordinate_scale=1.0,
        bbox_normalize_dst_length=1.0,
    )
