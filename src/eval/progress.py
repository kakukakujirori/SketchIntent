"""Progress helpers for long-running EDA scripts."""
from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress(items: Iterable[T], *, desc: str, total: int | None = None, every: int = 10) -> Iterable[T]:
    """Wrap an iterable with tqdm when available, otherwise use sparse stderr prints."""
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return _fallback_progress(items, desc=desc, total=total, every=every)
    return tqdm(items, desc=desc, total=total)


def _fallback_progress(items: Iterable[T], *, desc: str, total: int | None, every: int) -> Iterator[T]:
    for index, item in enumerate(items, start=1):
        if index == 1 or index % every == 0 or (total is not None and index == total):
            suffix = f"{index}/{total}" if total is not None else str(index)
            print(f"{desc}: {suffix}", file=sys.stderr, flush=True)
        yield item
