"""Dataset discovery for the completed ``data_eval`` layout."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import ANNOTATION_LABELS, data_root

_FCSTD_RE = re.compile(r"^(?P<base>.+)_(?P<init>[A-Z])\.FCStd$", re.IGNORECASE)


@dataclass
class SketchGroup:
    """One sketch identity within one assignment batch."""

    sketch_id: str
    batch: str
    files: dict[str, str] = field(default_factory=dict)  # initial -> filename

    @property
    def base_fcstd(self) -> str:
        return f"{self.sketch_id}.FCStd"

    @property
    def is_complete(self) -> bool:
        return len(self.files) == 3

    def file_path(self, initial: str) -> Path:
        return data_root() / self.batch / self.files[initial]


def _scan_groups() -> tuple[dict[tuple[str, str], SketchGroup], Counter]:
    """Scan all FCStd annotations, returning groups and exclusion counters."""
    groups: dict[tuple[str, str], SketchGroup] = {}
    stats: Counter = Counter()
    batches = sorted(
        (p for p in data_root().iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for batch_dir in batches:
        for path in sorted(batch_dir.iterdir()):
            fname = path.name
            if not fname.lower().endswith(".fcstd"):
                continue
            if fname.lower().endswith("_mid.fcstd"):
                stats["excluded_mid"] += 1
                continue
            if fname.startswith("_"):
                stats["excluded_underscore"] += 1
                continue
            m = _FCSTD_RE.match(fname)
            if not m or m.group("init").upper() not in ANNOTATION_LABELS:
                stats["excluded_unparsable"] += 1
                continue
            init = m.group("init").upper()
            sketch_id = m.group("base")
            key = (batch_dir.name, sketch_id)
            groups.setdefault(key, SketchGroup(sketch_id=sketch_id, batch=batch_dir.name))
            groups[key].files[init] = fname
    return groups, stats


def discover_manifest(
    require_k: int = 3,
) -> tuple[dict[str, dict[str, dict[str, str]]], Counter]:
    """Return the legacy manifest shape used by evaluation scripts.

    ``manifest[batch][base_fcstd][initial] = actual_filename``.
    """
    groups, stats = _scan_groups()
    manifest: dict[str, dict[str, dict[str, str]]] = {}
    for (_batch, _sketch_id), group in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        if len(group.files) != require_k:
            stats[f"excluded_k{len(group.files)}"] += 1
            continue
        manifest.setdefault(group.batch, {})[group.base_fcstd] = dict(sorted(group.files.items()))
        stats["included_sketches"] += 1
        stats["included_annotations"] += len(group.files)
    return manifest, stats


def discover_groups() -> tuple[list[SketchGroup], list[SketchGroup], Counter]:
    """Return complete and incomplete groups for EDA extraction."""
    groups, stats = _scan_groups()
    complete, incomplete = [], []
    for (_batch, _sketch_id), group in sorted(groups.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        if group.is_complete:
            complete.append(group)
        else:
            incomplete.append(group)
            stats[f"excluded_k{len(group.files)}"] += 1
    stats["included_sketches"] = len(complete)
    stats["included_annotations"] = sum(len(g.files) for g in complete)
    return complete, incomplete, stats
