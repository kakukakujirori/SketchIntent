"""Path bootstrap for running EDA scripts as plain files."""
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root
