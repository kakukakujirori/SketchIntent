"""Runtime warnings shared by the Section 5 reproduction scripts."""
from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

from src.eval.config import MODEL_OUTPUT_ROOTS, missing_model_roots

MODEL_LABELS = tuple(MODEL_OUTPUT_ROOTS)


def warn_missing_model_roots(*, script: str, batches: Iterable[str]) -> None:
    """Warn when configured model prediction roots are missing."""
    missing = missing_model_roots()
    if not missing:
        return
    batch_text = " ".join(str(b) for b in batches)
    roots = ", ".join(f"{label}={path}" for label, path in missing.items())
    print(
        f"[warn] {script}: configured model output root(s) do not exist: {roots}. "
        f"Model rows for affected batches ({batch_text}) will be missing or NA. "
        "Edit MODEL_OUTPUT_ROOTS in src/eval/config.py.",
        file=sys.stderr,
        flush=True,
    )


def validate_model_outputs(items: Iterable[tuple[str, str, dict]], *, script: str) -> None:
    """Fail fast when an existing model output root is incomplete."""
    active_roots = {
        label: root
        for label, root in MODEL_OUTPUT_ROOTS.items()
        if root.exists()
    }
    if not active_roots:
        return

    missing = []
    for batch, base_fcstd, _files in items:
        py_name = Path(base_fcstd).with_suffix(".py").name
        for label, root in active_roots.items():
            path = root / str(batch) / py_name
            if not path.exists():
                missing.append((label, str(batch), py_name, path))

    if not missing:
        return

    examples = "\n".join(
        f"  {label} batch={batch} file={py_name} expected={path}"
        for label, batch, py_name, path in missing[:10]
    )
    more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
    raise FileNotFoundError(
        f"{script}: configured model output root exists but prediction .py files are missing "
        f"({len(missing)} missing).\n{examples}{more}"
    )


def has_model_rows(rows: Iterable[dict]) -> bool:
    labels = set(MODEL_LABELS)
    return any(row.get("variant") in labels for row in rows)
