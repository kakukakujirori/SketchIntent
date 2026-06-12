"""Canonical input-sketch lookup."""
from __future__ import annotations

import copy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter

from .config import REPO_ROOT, data_root

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402
from src.sketch_handlers.sketch_utils import complete_constraint_values  # noqa: E402


def canonical_dir_for(
    batch: str,
    base_fcstd: str,
) -> Path | None:
    """Directory containing the canonical unannotated ``<sketch>.py`` in ``data_eval``."""
    py_name = Path(base_fcstd).with_suffix(".py").name
    directory = data_root() / batch
    if (directory / py_name).exists():
        return directory
    return None


def canonical_py_for(
    batch: str,
    base_fcstd: str,
) -> Path | None:
    directory = canonical_dir_for(batch, base_fcstd)
    return None if directory is None else directory / Path(base_fcstd).with_suffix(".py").name


def find_canonical_py(
    filename: str,
    data_dirs: Mapping[str, str],
    annotators: Sequence[str],
    canonical_dir: str | None = None,
) -> str:
    """Find the canonical unannotated Python sketch for ``filename``."""
    py_name = Path(filename).with_suffix(".py").name
    search_dirs: list[str] = []
    if canonical_dir:
        search_dirs.append(canonical_dir)
    search_dirs.extend(data_dirs[ann] for ann in annotators)

    for directory in search_dirs:
        path = Path(directory) / py_name
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Canonical .py not found for {filename}")


def load_sketch_from_fcstd(path: str, converter: SketchFreecadConverter) -> datalib.Sketch:
    """Load an annotated FCStd sketch and complete display constraint values."""
    doc = None
    try:
        doc, sketch_fc = load_FCStd_sketch(path)
        sketch = converter.from_freecad(sketch_fc)
        sketch, _ = complete_constraint_values(sketch)
        return sketch
    finally:
        if doc is not None:
            try:
                FreeCAD, _, _ = ensure_freecad_imported()
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                pass


def load_sketch_from_py(path: str, converter: SketchFreecadConverter) -> datalib.Sketch:
    """Load a Python sketch command file and complete display constraint values."""
    sketch = converter.from_freecad_commands(path)
    sketch, _ = complete_constraint_values(sketch)
    return sketch


def metric_base_eids(metric: str, canonical_sketch: datalib.Sketch) -> list[str]:
    """Return canonical entity ids used by paper metrics."""
    if metric == "canonical_solid":
        return [
            eid
            for eid, ent in canonical_sketch.entities.items()
            if not getattr(ent, "isConstruction", False)
        ]
    if metric == "canonical_all":
        return list(canonical_sketch.entities.keys())
    raise ValueError(metric)


def _offset_point(point, offset: np.ndarray) -> np.ndarray:
    return np.asarray(point, dtype=float) + offset


def _entity_points(entity: datalib.Entity) -> list[np.ndarray]:
    if isinstance(entity, datalib.Line):
        return [
            np.asarray(entity.start_point, dtype=float),
            np.asarray(entity.end_point, dtype=float),
        ]
    if isinstance(entity, datalib.Circle):
        center = np.asarray(entity.center_point, dtype=float)
        radius = float(entity.radius)
        return [
            center + np.array([radius, 0.0]),
            center - np.array([radius, 0.0]),
            center + np.array([0.0, radius]),
            center - np.array([0.0, radius]),
        ]
    if isinstance(entity, datalib.Arc):
        return [
            np.asarray(entity.start_point, dtype=float),
            np.asarray(entity.end_point, dtype=float),
            np.asarray(entity.mid_point, dtype=float),
        ]
    if isinstance(entity, datalib.Point):
        return [np.asarray([entity.x, entity.y], dtype=float)]
    return []


def _bbox_center(entities: Sequence[datalib.Entity]) -> np.ndarray:
    points = [point for entity in entities for point in _entity_points(entity)]
    if not points:
        return np.zeros(2, dtype=float)
    stacked = np.vstack(points)
    return (np.min(stacked, axis=0) + np.max(stacked, axis=0)) / 2.0


def _solid_or_all_entities(sketch: datalib.Sketch) -> list[datalib.Entity]:
    solid = [
        entity
        for entity in sketch.entities.values()
        if not getattr(entity, "isConstruction", False)
    ]
    return solid if solid else list(sketch.entities.values())


def _bbox_alignment_offset(
    source_sketch: datalib.Sketch,
    canonical_sketch: datalib.Sketch,
) -> np.ndarray:
    source_center = _bbox_center(_solid_or_all_entities(source_sketch))
    canonical_center = _bbox_center(_solid_or_all_entities(canonical_sketch))
    return canonical_center - source_center


def _entity_distance(
    a: datalib.Entity,
    b: datalib.Entity,
    source_offset: np.ndarray | None = None,
) -> float:
    source_offset = np.zeros(2, dtype=float) if source_offset is None else source_offset
    if type(a) is not type(b):
        return float("inf")
    if isinstance(a, datalib.Line):
        a0 = _offset_point(a.start_point, source_offset)
        a1 = _offset_point(a.end_point, source_offset)
        b0 = np.asarray(b.start_point, dtype=float)
        b1 = np.asarray(b.end_point, dtype=float)
        direct = max(np.linalg.norm(a0 - b0), np.linalg.norm(a1 - b1))
        reverse = max(np.linalg.norm(a0 - b1), np.linalg.norm(a1 - b0))
        return min(direct, reverse)
    if isinstance(a, datalib.Circle):
        return max(
            np.linalg.norm(_offset_point(a.center_point, source_offset) - np.asarray(b.center_point, dtype=float)),
            abs(float(a.radius) - float(b.radius)),
        )
    if isinstance(a, datalib.Arc):
        ac = _offset_point(a.center_point, source_offset)
        bc = np.asarray(b.center_point, dtype=float)
        a0 = _offset_point(a.start_point, source_offset)
        a1 = _offset_point(a.end_point, source_offset)
        am = _offset_point(a.mid_point, source_offset)
        b0 = np.asarray(b.start_point, dtype=float)
        b1 = np.asarray(b.end_point, dtype=float)
        bm = np.asarray(b.mid_point, dtype=float)
        center_radius = max(np.linalg.norm(ac - bc), abs(float(a.radius) - float(b.radius)))
        direct = max(center_radius, np.linalg.norm(a0 - b0), np.linalg.norm(a1 - b1), np.linalg.norm(am - bm))
        reverse = max(center_radius, np.linalg.norm(a0 - b1), np.linalg.norm(a1 - b0), np.linalg.norm(am - bm))
        return min(direct, reverse)
    if isinstance(a, datalib.Point):
        ap = _offset_point([a.x, a.y], source_offset)
        bp = np.asarray([b.x, b.y], dtype=float)
        return float(np.linalg.norm(ap - bp))
    return float("inf")


def _copy_entity_with_id(
    entity: datalib.Entity,
    new_id: str,
    offset: np.ndarray | None = None,
) -> datalib.Entity:
    ent = copy.deepcopy(entity)
    ent.entityId = new_id
    if offset is not None:
        dx, dy = float(offset[0]), float(offset[1])
        if isinstance(ent, datalib.Line):
            ent.pntX += dx
            ent.pntY += dy
        elif isinstance(ent, (datalib.Circle, datalib.Arc)):
            ent.xCenter += dx
            ent.yCenter += dy
        elif isinstance(ent, datalib.Point):
            ent.x += dx
            ent.y += dy
    return ent


def _count_construction(
    ids: Sequence[str] | Mapping[str, datalib.Entity],
    entities: Mapping[str, datalib.Entity],
    is_construction: bool,
) -> int:
    return sum(
        1
        for eid in ids
        if getattr(entities[eid], "isConstruction", False) is is_construction
    )


def build_entity_remap(
    source_sketch: datalib.Sketch,
    canonical_sketch: datalib.Sketch,
    tol: float,
) -> tuple[dict[str, str], dict[str, datalib.Entity], dict[str, int | float | str]]:
    """Map source entity ids onto canonical ids, preserving extras."""
    source_to_new = {}
    matched_canonical = {}
    used_canonical = set()
    extra_entities = {}
    source_offset = _bbox_alignment_offset(source_sketch, canonical_sketch)

    for sid, source_ent in source_sketch.entities.items():
        candidates = []
        for cid, canonical_ent in canonical_sketch.entities.items():
            if cid in used_canonical:
                continue
            dist = _entity_distance(source_ent, canonical_ent, source_offset)
            if dist <= tol:
                same_flag = getattr(source_ent, "isConstruction", False) == getattr(canonical_ent, "isConstruction", False)
                candidates.append((dist, 0 if same_flag else 1, cid))
        if candidates:
            _, _, cid = min(candidates)
            source_to_new[sid] = cid
            matched_canonical[cid] = sid
            used_canonical.add(cid)
        else:
            extra_id = f"extra_{sid}"
            while extra_id in canonical_sketch.entities or extra_id in extra_entities:
                extra_id = f"extra_{extra_id}"
            source_to_new[sid] = extra_id
            extra_entities[extra_id] = _copy_entity_with_id(source_ent, extra_id, source_offset)

    missing = [eid for eid in canonical_sketch.entities if eid not in used_canonical]
    flag_changed = [
        cid
        for cid, sid in matched_canonical.items()
        if getattr(canonical_sketch.entities[cid], "isConstruction", False)
        != getattr(source_sketch.entities[sid], "isConstruction", False)
    ]
    audit = {
        "matched": len(matched_canonical),
        "missing_solid": _count_construction(missing, canonical_sketch.entities, False),
        "missing_construction": _count_construction(missing, canonical_sketch.entities, True),
        "extra_solid": _count_construction(extra_entities, extra_entities, False),
        "extra_construction": _count_construction(extra_entities, extra_entities, True),
        "construction_flag_changed": len(flag_changed),
        "dropped_constraints": 0,
        "alignment_dx": float(source_offset[0]),
        "alignment_dy": float(source_offset[1]),
        "missing_ids": ",".join(missing),
        "extra_ids": ",".join(extra_entities),
        "flag_changed_ids": ",".join(flag_changed),
    }
    return source_to_new, extra_entities, audit


def remap_sketch_to_canonical(
    source_sketch: datalib.Sketch,
    canonical_sketch: datalib.Sketch,
    source_to_new: Mapping[str, str],
    extra_entities: Mapping[str, datalib.Entity],
) -> tuple[datalib.Sketch, int]:
    """Return source constraints remapped onto canonical geometry."""
    entities = copy.deepcopy(canonical_sketch.entities)
    entities.update(copy.deepcopy(extra_entities))
    constraints = {}
    dropped = 0

    for cid, constraint in source_sketch.constraints.items():
        try:
            constraints[cid] = _remap_constraint(constraint, source_to_new)
        except KeyError:
            dropped += 1

    return datalib.Sketch(entities=entities, constraints=constraints), dropped


def _remap_constraint(
    constraint: datalib.Constraint,
    source_to_new: Mapping[str, str],
) -> datalib.Constraint:
    params = []
    for param in constraint.parameters:
        if isinstance(param, LocalReferenceParameter):
            params.append(
                LocalReferenceParameter(
                    parameterId=param.parameterId,
                    value=_remap_ref(str(param.value), source_to_new),
                )
            )
        else:
            params.append(copy.deepcopy(param))
    return constraint._replace(parameters=params)


def _remap_ref(ref: str, source_to_new: Mapping[str, str]) -> str:
    eid, suffix = ref.split(".", 1) if "." in ref else (ref, None)
    new_eid = eid if eid.startswith("-") else source_to_new[eid]
    return f"{new_eid}.{suffix}" if suffix else new_eid
