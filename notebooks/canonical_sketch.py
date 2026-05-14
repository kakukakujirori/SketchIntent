import copy
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter

from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import complete_constraint_values


CANONICAL_METRICS = ("canonical_solid", "canonical_all")
RELATIONAL_CONSTRAINT_SET = "relational"
SELF_DIMENSIONING_CONSTRAINT_SET = "relational + self_dimensioning"
CONSTRAINT_SETS = (RELATIONAL_CONSTRAINT_SET, SELF_DIMENSIONING_CONSTRAINT_SET)

DIMENSIONAL_CONSTRAINTS = {
    "Distance",
    "DistanceX",
    "DistanceY",
    "Length",
    "Diameter",
    "Radius",
    "Angle",
    "Offset",
    "Centerline_Dimension",
    "Minor_Diameter",
    "Major_Diameter",
    "Rho",
    "Block",
    "InternalAlignment",
}

SELF_DIMENSIONING_CONSTRAINTS = {
    "Distance",
    "DistanceX",
    "DistanceY",
    "Length",
    "Diameter",
    "Radius",
    "Angle",
}


@dataclass(frozen=True)
class CanonicalComparisonSketches:
    filename: str
    canonical_path: str
    canonical_sketch: datalib.Sketch
    remapped_relational_sketches: dict[str, datalib.Sketch]
    remapped_constraint_sketches: dict[str, dict[str, datalib.Sketch]]
    geometry_audits: dict[str, dict[str, int | float | str]]
    dimensional_counts: dict[str, int]
    self_dimensioning_counts: dict[str, int]
    dropped_constraints: dict[str, dict[str, int]]


def is_dimensional_constraint(constraint: datalib.Constraint) -> bool:
    return constraint.constraint_type.name in DIMENSIONAL_CONSTRAINTS


def _local_reference_entity_ids(constraint: datalib.Constraint) -> list[str]:
    refs = []
    for param in constraint.parameters:
        if isinstance(param, LocalReferenceParameter):
            refs.append(str(param.value).split(".", 1)[0])
    return refs


def is_self_dimensioning_constraint(constraint: datalib.Constraint) -> bool:
    if constraint.constraint_type.name not in SELF_DIMENSIONING_CONSTRAINTS:
        return False
    entity_ids = _local_reference_entity_ids(constraint)
    return bool(entity_ids) and len(set(entity_ids)) == 1


def count_dimensional_constraints(sketch: datalib.Sketch) -> int:
    return sum(1 for c in sketch.constraints.values() if is_dimensional_constraint(c))


def count_self_dimensioning_constraints(sketch: datalib.Sketch) -> int:
    return sum(1 for c in sketch.constraints.values() if is_self_dimensioning_constraint(c))


def constraint_in_set(constraint: datalib.Constraint, constraint_set: str) -> bool:
    if constraint_set == RELATIONAL_CONSTRAINT_SET:
        return not is_dimensional_constraint(constraint)
    if constraint_set == SELF_DIMENSIONING_CONSTRAINT_SET:
        return (
            not is_dimensional_constraint(constraint)
            or is_self_dimensioning_constraint(constraint)
        )
    raise ValueError(f"Unknown constraint set: {constraint_set}")


def get_constraint_set_sketch(sketch: datalib.Sketch, constraint_set: str) -> datalib.Sketch:
    sk = copy.deepcopy(sketch)
    sk.constraints = {
        cid: c
        for cid, c in sk.constraints.items()
        if constraint_in_set(c, constraint_set)
    }
    return sk


def get_relational_sketch(sketch: datalib.Sketch) -> datalib.Sketch:
    return get_constraint_set_sketch(sketch, RELATIONAL_CONSTRAINT_SET)


def find_canonical_py(
    filename: str,
    data_dirs: Mapping[str, str],
    annotators: Sequence[str],
    canonical_dir: str | None = None,
) -> str:
    py_name = os.path.splitext(filename)[0] + ".py"
    search_dirs = []
    if canonical_dir:
        search_dirs.append(canonical_dir)
    search_dirs.extend(data_dirs[ann] for ann in annotators)

    for directory in search_dirs:
        path = os.path.join(directory, py_name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Canonical .py not found for {filename}")


def load_sketch_from_fcstd(path: str, converter: SketchFreecadConverter) -> datalib.Sketch:
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
    sketch = converter.from_freecad_commands(path)
    sketch, _ = complete_constraint_values(sketch)
    return sketch


def build_canonical_comparison_sketches(
    filename: str,
    data_dirs: Mapping[str, str],
    file_maps: Mapping[str, Mapping[str, str]],
    annotators: Sequence[str],
    converter: SketchFreecadConverter,
    geometry_tol: float,
    canonical_dir: str | None = None,
) -> CanonicalComparisonSketches:
    canonical_path = find_canonical_py(filename, data_dirs, annotators, canonical_dir)
    canonical_sketch = load_sketch_from_py(canonical_path, converter)

    full_sketches = {}
    dimensional_counts = {}
    self_dimensioning_counts = {}
    for annotator in annotators:
        fcstd_path = os.path.join(data_dirs[annotator], file_maps[annotator][filename])
        full_sketches[annotator] = load_sketch_from_fcstd(fcstd_path, converter)
        dimensional_counts[annotator] = count_dimensional_constraints(full_sketches[annotator])
        self_dimensioning_counts[annotator] = count_self_dimensioning_constraints(full_sketches[annotator])

    remapped_constraint_sketches = {
        constraint_set: {}
        for constraint_set in CONSTRAINT_SETS
    }
    dropped_constraints = {
        constraint_set: {}
        for constraint_set in CONSTRAINT_SETS
    }
    geometry_audits = {}
    for annotator in annotators:
        source_to_new, extra_entities, audit = build_entity_remap(
            full_sketches[annotator],
            canonical_sketch,
            geometry_tol,
        )

        for constraint_set in CONSTRAINT_SETS:
            filtered_sketch = get_constraint_set_sketch(full_sketches[annotator], constraint_set)
            remapped, dropped = remap_sketch_to_canonical(
                filtered_sketch,
                canonical_sketch,
                source_to_new,
                extra_entities,
            )
            remapped_constraint_sketches[constraint_set][annotator] = remapped
            dropped_constraints[constraint_set][annotator] = dropped

        audit["dropped_constraints"] = dropped_constraints[RELATIONAL_CONSTRAINT_SET][annotator]
        audit["dropped_self_dimensioning_constraints"] = dropped_constraints[SELF_DIMENSIONING_CONSTRAINT_SET][annotator]
        geometry_audits[annotator] = audit

    return CanonicalComparisonSketches(
        filename=filename,
        canonical_path=canonical_path,
        canonical_sketch=canonical_sketch,
        remapped_relational_sketches=remapped_constraint_sketches[RELATIONAL_CONSTRAINT_SET],
        remapped_constraint_sketches=remapped_constraint_sketches,
        geometry_audits=geometry_audits,
        dimensional_counts=dimensional_counts,
        self_dimensioning_counts=self_dimensioning_counts,
        dropped_constraints=dropped_constraints,
    )


def metric_valid(metric: str, audits: Mapping[str, Mapping[str, int | float | str]]) -> tuple[bool, str]:
    if metric == "canonical_solid":
        bad = [ann for ann, a in audits.items() if int(a["missing_solid"]) > 0]
        return not bad, "missing canonical solid: " + ",".join(bad) if bad else ""
    if metric == "canonical_all":
        bad = [
            ann
            for ann, a in audits.items()
            if int(a["missing_solid"]) > 0 or int(a["missing_construction"]) > 0
        ]
        return not bad, "missing canonical geometry: " + ",".join(bad) if bad else ""
    raise ValueError(metric)


def metric_base_eids(metric: str, canonical_sketch: datalib.Sketch) -> list[str]:
    if metric == "canonical_solid":
        return [
            eid
            for eid, ent in canonical_sketch.entities.items()
            if not getattr(ent, "isConstruction", False)
        ]
    if metric == "canonical_all":
        return list(canonical_sketch.entities.keys())
    raise ValueError(metric)


def metric_include_quadratic(metric: str, audits: Mapping[str, Mapping[str, int | float | str]]) -> bool:
    if metric in CANONICAL_METRICS:
        return True
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


def align_pred_entity_ids_to_gt(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
) -> tuple[datalib.Sketch, int]:
    """Align a generated prediction's compact entity ids to a GT sketch by order.

    Dataset-tokenized model outputs commonly use compact ids (0, 1, 2, ...),
    while the source SketchGraphs GT may retain sparse ids (for example
    1, 4, 7, ...).  When solver feedback has already overwritten primitive
    geometry by order, response metrics must evaluate predicted constraints in
    the GT entity coordinate chart before applying GT edit probes / relations.

    Returns the aligned sketch and the number of predicted constraints dropped
    because a referenced entity could not be mapped.  If the prediction has more
    entities than the GT, the function leaves it unchanged and reports the extra
    entity count; callers should treat that as a diagnostic.
    """
    pred_ids = list(pred_sketch.entities.keys())
    gt_ids = list(gt_sketch.entities.keys())
    if len(pred_ids) > len(gt_ids):
        return pred_sketch, len(pred_ids) - len(gt_ids)

    source_to_new = {src: dst for src, dst in zip(pred_ids, gt_ids)}
    entities = copy.deepcopy(gt_sketch.entities)
    constraints: dict[str, datalib.Constraint] = {}
    dropped = 0

    for cid, constraint in pred_sketch.constraints.items():
        try:
            constraints[cid] = _remap_constraint(constraint, source_to_new)
        except KeyError:
            dropped += 1

    return datalib.Sketch(entities=entities, constraints=constraints), dropped
