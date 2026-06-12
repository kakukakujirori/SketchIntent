"""Path bootstrap for running EDA scripts as plain files."""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    notebooks = repo_root / "notebooks"
    for path in (repo_root, notebooks, notebooks / "v4_exp"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root
