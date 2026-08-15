"""Shared plumbing: dataset iteration, run metadata, output guards, cluster bootstrap."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.bundle_loader import build_bundle  # noqa: E402
from src.eval.config import BASE_POLICY, data_root  # noqa: E402
from src.eval.file_bundle import base_policy_pass, build_base_eids  # noqa: E402
from src.eval.manifest import discover_manifest  # noqa: E402
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402


@dataclass(frozen=True)
class SketchItem:
    batch: str
    base_fcstd: str
    files: dict[str, str]

    @property
    def sketch_id(self) -> str:
        return self.base_fcstd[: -len(".FCStd")]

    @property
    def key(self) -> str:
        return f"{self.batch}/{self.sketch_id}"

    @property
    def human_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.files))


def build_context() -> tuple[SketchContext, SketchFreecadConverter]:
    context = SketchContext()
    return context, SketchFreecadConverter(context, verbose=False)


def discover_items(batches: Sequence[str] | None = None, limit: int | None = None) -> list[SketchItem]:
    manifest, _stats = discover_manifest()
    selected = batches or sorted(manifest, key=int)
    items = [
        SketchItem(batch=batch, base_fcstd=base, files=files)
        for batch in selected
        for base, files in sorted(manifest.get(batch, {}).items())
    ]
    return items[:limit] if limit is not None else items


def shard_of(key: str, n_shards: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % n_shards


def select_shard(items: Sequence[SketchItem], shard: int, n_shards: int) -> list[SketchItem]:
    return [item for item in items if shard_of(item.key, n_shards) == shard]


def load_bundle(item: SketchItem, converter, geometry_tol: float):
    return build_bundle(item.batch, item.base_fcstd, item.files, converter, geometry_tol=geometry_tol)


def usable_labels(bundle, item: SketchItem) -> tuple[list[str], list[str]]:
    """Human labels (all must pass the base policy) and model labels that pass it individually."""
    humans = [n for n in item.human_labels if n in bundle.audits]
    if len(humans) != len(item.human_labels):
        return [], []
    if not all(base_policy_pass(bundle.audits[n], BASE_POLICY)[0] for n in humans):
        return [], []
    models = [
        label
        for label in bundle.loaded_model_labels()
        if label in bundle.audits and base_policy_pass(bundle.audits[label], BASE_POLICY)[0]
    ]
    return humans, models


# ---------------------------------------------------------------------------
# Run metadata & output directory guard
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        return ""


def _freecad_version() -> str:
    try:
        from src.sketch_handlers.freecad_utils import ensure_freecad_imported

        FreeCAD, _, _ = ensure_freecad_imported()
        return " ".join(str(part) for part in FreeCAD.Version())
    except Exception:
        return "unavailable"


def dataset_input_digest() -> str:
    """Fingerprint of every dataset input file (path, size, mtime) for the mutation guard."""
    root = data_root()
    entries = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".fcstd", ".py"):
            continue
        stat = path.stat()
        entries.append(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def script_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def environment_metadata() -> dict:
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty_files=_git("status", "--short").splitlines(),
        python_version=sys.version.split()[0],
        freecad_version=_freecad_version(),
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        numpy_version=np.__version__,
        dataset_input_digest=dataset_input_digest(),
    )


def prepare_output_dir(out_dir: Path, config: dict, *, force: bool = False) -> str:
    """Create out_dir and refuse to mix results produced under a different config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = config_hash(config)
    marker = out_dir / "run_config.json"
    if marker.exists() and not force:
        previous = json.loads(marker.read_text())
        if previous.get("config_hash") != digest:
            raise SystemExit(
                f"{marker} holds config_hash={previous.get('config_hash')} but this run is "
                f"{digest}. Use a different --out or pass --force to overwrite."
            )
    marker.write_text(
        json.dumps(
            dict(config_hash=digest, config=config, started_at=_utc_now()),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return digest


def write_environment(
    out_dir: Path,
    command: Sequence[str],
    scripts: Iterable[Path],
    *,
    filename: str = "environment.json",
) -> None:
    payload = environment_metadata()
    payload.update(command_line=list(command), script_digest=script_digest(scripts))
    (out_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True))


def finalize_run_config(out_dir: Path, **extra) -> None:
    marker = out_dir / "run_config.json"
    payload = json.loads(marker.read_text())
    payload.update(finished_at=_utc_now(), **extra)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Sketch-cluster bootstrap
# ---------------------------------------------------------------------------

def cluster_index_groups(cluster_ids: Sequence) -> list[np.ndarray]:
    groups: dict = {}
    for row, cluster in enumerate(cluster_ids):
        groups.setdefault(cluster, []).append(row)
    return [np.asarray(rows, dtype=np.int64) for _, rows in sorted(groups.items(), key=lambda kv: str(kv[0]))]


def sketch_cluster_bootstrap(
    cluster_ids: Sequence,
    statistic: Callable[[np.ndarray], float],
    *,
    n_replicates: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Resample whole sketches with replacement; every row of a drawn sketch is kept.

    ``statistic`` receives the resampled row indices and returns a scalar, or NaN when
    it is undefined for that replicate. The NaNs are preserved so callers can report
    how many replicates were usable.
    """
    groups = cluster_index_groups(cluster_ids)
    if not groups:
        return np.full(n_replicates, np.nan)
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        drawn = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[j] for j in drawn])
        try:
            replicates[i] = float(statistic(rows))
        except Exception:
            replicates[i] = np.nan
    return replicates


def percentile_ci(replicates: np.ndarray, alpha: float = 0.05) -> tuple[float, float, int]:
    valid = replicates[np.isfinite(replicates)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high), int(valid.size)


__all__ = [
    "BASE_POLICY",
    "REPO_ROOT",
    "SketchItem",
    "build_base_eids",
    "build_context",
    "cluster_index_groups",
    "config_hash",
    "discover_items",
    "environment_metadata",
    "finalize_run_config",
    "load_bundle",
    "percentile_ci",
    "prepare_output_dir",
    "select_shard",
    "shard_of",
    "sketch_cluster_bootstrap",
    "usable_labels",
    "write_csv",
    "write_environment",
]
