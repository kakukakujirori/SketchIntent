"""FileBundle construction and source-conditioned evaluation helpers."""
from __future__ import annotations

import copy
import hashlib
import os
import random
import re
import sys
from dataclasses import dataclass, field

import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter

from .canonical import (
    build_entity_remap,
    find_canonical_py,
    load_sketch_from_fcstd,
    load_sketch_from_py,
    metric_base_eids,
    remap_sketch_to_canonical,
)
from .config import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.edit_response_distance import (  # noqa: E402
    DIMENSIONAL_CONSTRAINTS,
    collect_probes,
    constraint_signature,
    _is_relational,
)
from src.metrics.histcad_2d import compute_histcad_2d_freecad  # noqa: E402
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, get_solver_status, load_freecad_commands  # noqa: E402
from src.sketch_handlers.sketch_utils import complete_constraint_values, get_sketch_bbox  # noqa: E402


@dataclass
class FileBundle:
    batch: str
    filename: str
    canonical_sketch: datalib.Sketch
    sketches: dict[str, datalib.Sketch]
    audits: dict[str, dict]
    model_statuses: dict[str, str] = field(default_factory=dict)
    model_dofs: dict[str, int | None] = field(default_factory=dict)

    def loaded_model_labels(self) -> list[str]:
        return [label for label, status in self.model_statuses.items() if status == "loaded"]

    @property
    def model_status(self) -> str:
        if not self.model_statuses:
            return "missing"
        return next(iter(self.model_statuses.values()))

    @property
    def model_dof(self) -> int | None:
        if not self.model_dofs:
            return None
        return next(iter(self.model_dofs.values()))


def is_dimensional(constraint) -> bool:
    return constraint.constraint_type.name in DIMENSIONAL_CONSTRAINTS


def base_policy_pass(audit, policy: str) -> tuple[bool, str]:
    if policy == "none":
        return True, ""
    if policy == "canonical_solid":
        ok = int(audit["missing_solid"]) == 0
        return ok, "" if ok else "missing canonical solid"
    if policy == "canonical_all":
        ok = int(audit["missing_solid"]) == 0 and int(audit["missing_construction"]) == 0
        return ok, "" if ok else "missing canonical geometry"
    raise ValueError(policy)


def build_base_eids(canonical_sketch, policy: str):
    return None if policy == "none" else metric_base_eids(policy, canonical_sketch)


def vitruvion_solver_status(path: str) -> tuple[int | None, str]:
    FreeCAD, _, _ = ensure_freecad_imported()
    doc = None
    try:
        doc, fc_sketch = load_freecad_commands(path)
        status = get_solver_status(fc_sketch)
        if isinstance(status, dict):
            return status.get("DoF"), status.get("Status")
        return None, str(status)
    except Exception:
        return None, "load_error"
    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                pass


_HEADER_RE = re.compile(
    r"Status:\s*(?P<status>\w+).*?Final solver status:\s*(?P<solver>\w+)"
)
_REJECTED_HEADER_STATUSES = frozenset({"Distorted", "SolverError"})


def parse_pred_header(path: str) -> tuple[str | None, str | None]:
    try:
        with open(path) as f:
            line = f.readline()
    except OSError:
        return None, None
    match = _HEADER_RE.search(line)
    if not match:
        return None, None
    return match.group("status"), match.group("solver")


def model_path_for(root, batch, filename):
    if root is None:
        return None
    path = os.path.join(root, batch, os.path.splitext(filename)[0] + ".py")
    return path if os.path.exists(path) else None


def _coordinate_transform_from_canonical(
    canonical_sketch,
    *,
    coordinate_scale: float,
    bbox_normalize_dst_length: float | None,
) -> tuple[float, float, float]:
    if bbox_normalize_dst_length is None:
        return 0.0, 0.0, float(coordinate_scale)

    (x0, y0), (x1, y1) = get_sketch_bbox(canonical_sketch)
    w, h = float(x1 - x0), float(y1 - y0)
    long_axis = max(w, h)
    if long_axis <= 0:
        return 0.0, 0.0, float(coordinate_scale)
    return (
        float((x0 + x1) * 0.5),
        float((y0 + y1) * 0.5),
        float(coordinate_scale) * float(bbox_normalize_dst_length) / long_axis,
    )


def _transform_sketch_coordinates(
    sketch: datalib.Sketch,
    *,
    offset_x: float,
    offset_y: float,
    scale: float,
) -> datalib.Sketch:
    out = copy.deepcopy(sketch)
    for ent in out.entities.values():
        if isinstance(ent, datalib.Line):
            ent.pntX = (ent.pntX - offset_x) * scale
            ent.pntY = (ent.pntY - offset_y) * scale
            ent.startParam *= scale
            ent.endParam *= scale
        elif isinstance(ent, (datalib.Circle, datalib.Arc)):
            ent.xCenter = (ent.xCenter - offset_x) * scale
            ent.yCenter = (ent.yCenter - offset_y) * scale
            ent.radius *= scale
        elif isinstance(ent, datalib.Point):
            ent.x = (ent.x - offset_x) * scale
            ent.y = (ent.y - offset_y) * scale
    out, _invalid = complete_constraint_values(out)
    return out


def transform_bundle_coordinates(
    bundle: FileBundle,
    *,
    coordinate_scale: float = 1.0,
    bbox_normalize_dst_length: float | None = 1.0,
) -> FileBundle:
    if coordinate_scale == 1.0 and bbox_normalize_dst_length is None:
        return bundle
    offset_x, offset_y, scale = _coordinate_transform_from_canonical(
        bundle.canonical_sketch,
        coordinate_scale=coordinate_scale,
        bbox_normalize_dst_length=bbox_normalize_dst_length,
    )
    bundle.canonical_sketch = _transform_sketch_coordinates(
        bundle.canonical_sketch,
        offset_x=offset_x,
        offset_y=offset_y,
        scale=scale,
    )
    bundle.sketches = {
        name: _transform_sketch_coordinates(
            sketch,
            offset_x=offset_x,
            offset_y=offset_y,
            scale=scale,
        )
        for name, sketch in bundle.sketches.items()
    }
    return bundle


def build_file_bundle(
    *,
    batch,
    filename,
    humans,
    file_maps,
    converter,
    geometry_tol,
    canonical_dir,
    model_root=None,
    model_specs: list[tuple[str, str]] | None = None,
    coordinate_scale: float = 1.0,
    bbox_normalize_dst_length: float | None = 1.0,
):
    """Load human sketches plus optional model variants into one remapped bundle."""
    data_dirs = {name: directory for name, directory in humans}
    annotators = tuple(data_dirs.keys())
    canonical_path = find_canonical_py(filename, data_dirs, annotators, canonical_dir=canonical_dir)
    canonical = load_sketch_from_py(canonical_path, converter)
    raw = {}
    for name, directory in humans:
        raw[name] = load_sketch_from_fcstd(
            os.path.join(directory, file_maps[name][filename]), converter
        )

    specs: list[tuple[str, str]] = []
    if model_root is not None:
        specs.append(("Vitruvion", model_root))
    if model_specs:
        specs.extend(model_specs)

    model_statuses: dict[str, str] = {}
    model_dofs: dict[str, int | None] = {}
    for label, root in specs:
        model_path = model_path_for(root, batch, filename)
        if model_path is None:
            model_statuses[label] = "missing"
            model_dofs[label] = None
            continue
        header_status, header_solver = parse_pred_header(model_path)
        dof, solver_status = vitruvion_solver_status(model_path)
        model_dofs[label] = dof
        accepted_solver = (header_solver or solver_status) == "Converged"
        rejected_label = header_status in _REJECTED_HEADER_STATUSES
        if accepted_solver and not rejected_label:
            try:
                raw[label] = load_sketch_from_py(model_path, converter)
                model_statuses[label] = "loaded"
            except Exception as exc:  # noqa: BLE001
                model_statuses[label] = f"error_load({type(exc).__name__}:{str(exc)[:80]})"
        else:
            model_statuses[label] = (
                f"filtered(dof={dof},st={solver_status},"
                f"hdr_status={header_status},hdr_solver={header_solver})"
            )

    sketches, audits = {}, {}
    for name, full in raw.items():
        try:
            source_to_new, extras, audit = build_entity_remap(full, canonical, geometry_tol)
            remapped, dropped = remap_sketch_to_canonical(full, canonical, source_to_new, extras)
        except Exception as exc:  # noqa: BLE001
            if name in model_statuses:
                model_statuses[name] = f"error_remap({type(exc).__name__}:{str(exc)[:80]})"
                continue
            raise
        audit["dropped_constraints"] = dropped
        sketches[name] = remapped
        audits[name] = audit
    bundle = FileBundle(
        batch=batch,
        filename=filename,
        canonical_sketch=canonical,
        sketches=sketches,
        audits=audits,
        model_statuses=model_statuses,
        model_dofs=model_dofs,
    )
    return transform_bundle_coordinates(
        bundle,
        coordinate_scale=coordinate_scale,
        bbox_normalize_dst_length=bbox_normalize_dst_length,
    )


def stable_seed(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256("::".join(str(part) for part in parts).encode()).digest()[:8],
        "big",
        signed=False,
    )


def sorted_dim_ids(sketch):
    return sorted(
        [cid for cid, constraint in sketch.constraints.items() if is_dimensional(constraint)],
        key=lambda cid: (constraint_signature(sketch.constraints[cid]), cid),
    )


def dropout_variant(sketch, *, drop_rate, seed):
    dim_ids = sorted_dim_ids(sketch)
    n = len(dim_ids)
    n_dropped = max(0, min(n, int(round(n * drop_rate / 100.0))))
    if n_dropped == 0:
        drop = set()
    elif n_dropped == n:
        drop = set(dim_ids)
    else:
        drop = set(random.Random(seed).sample(dim_ids, n_dropped))
    out = copy.deepcopy(sketch)
    out.constraints = {key: value for key, value in out.constraints.items() if key not in drop}
    return out, n, n - n_dropped, n_dropped


def _constraint_ref_strings(constraint):
    return [
        str(param.value)
        for param in constraint.parameters
        if isinstance(param, LocalReferenceParameter)
    ]


def _is_point_like_ref(sketch, ref: str) -> bool:
    eid, sep, _suffix = ref.partition(".")
    ent = sketch.entities.get(eid)
    if isinstance(ent, datalib.Point):
        return True
    return bool(sep)


def is_point_point_coincident(constraint, sketch) -> bool:
    if constraint.constraint_type.name != "Coincident":
        return False
    refs = _constraint_ref_strings(constraint)
    if len(refs) != 2:
        return False
    return all(_is_point_like_ref(sketch, ref) for ref in refs)


def preservation_relations(sketch, *, exclude_point_point_coincident=False):
    rels = [constraint for constraint in sketch.constraints.values() if _is_relational(constraint)]
    if exclude_point_point_coincident:
        rels = [constraint for constraint in rels if not is_point_point_coincident(constraint, sketch)]
    return rels


def source_probes(sketch, *, exclude_point_point_coincident=False):
    """Extract source-conditioned dimensional probes and preserved relations."""
    probes, _weights = collect_probes([sketch], kind="dim")
    rels = preservation_relations(
        sketch,
        exclude_point_point_coincident=exclude_point_point_coincident,
    )
    return probes, [1.0] * len(probes), rels, [1.0] * len(rels)


def validate_probes_on_source(source_sketch, probes, rels, context, converter, config):
    """Return mask where each probe is feasible on source for all edit signs."""
    if not probes:
        return []
    result = compute_histcad_2d_freecad(
        sketch=source_sketch,
        edit_probes=probes,
        relation_probes=rels,
        context=context,
        converter=converter,
        config=config,
    )
    feasible = [True] * len(probes)
    for probe_result in result.per_probe:
        if not probe_result.er:
            feasible[probe_result.probe_index] = False
    return feasible


def histcad_score(*, pred_sketch, gt_sketch, edit_probes, rels, context, converter, config):
    try:
        result = compute_histcad_2d_freecad(
            sketch=pred_sketch,
            edit_probes=edit_probes,
            relation_probes=rels,
            context=context,
            converter=converter,
            config=config,
        )
        n_modify_fail = n_insert_fail = 0
        pred_sigs = {constraint_signature(c) for c in pred_sketch.constraints.values()}
        for probe_result in result.per_probe:
            if not probe_result.er:
                if probe_result.probe_signature in pred_sigs:
                    n_modify_fail += 1
                else:
                    n_insert_fail += 1
        return (
            result.er_rate,
            result.cpcsr_rate,
            result.oes_rate,
            result.n_probes,
            result.n_relations,
            n_modify_fail,
            n_insert_fail,
            "",
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, None, 0, 0, str(exc)[:200]
