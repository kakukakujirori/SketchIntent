"""
exp_source_conditioned_dropout.py — Source-conditioned dimensional dropout.

For each rotation [Pred=A, GT=(B,C)]:
  - Q_i = Source_i's dimensional constraints (edit probes)
  - R_i = Source_i's relational constraints (preservation set)
  - Pred(r) = A with r% dimensional constraints dropped
  - Score(Pred(r), G_i) using Q_i and R_i

Also evaluates Vitruvion model output (when available) using the same
source-conditioned probes for direct comparison.

Optionally validates probes on source (--source-validation).
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, math, os, random, re, sys
from dataclasses import dataclass, field
from itertools import combinations
from statistics import median
from typing import Mapping, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.dirname(__file__)
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter
import torch

from src.metrics.edit_response_distance import (
    DIMENSIONAL_CONSTRAINTS, collect_probes, compute_native_response,
    compute_edit_response_distance, constraint_signature,
    constraint_jacobian, edit_response_for_probe,
    pairwise_distance_observable, _is_relational,
)
from src.metrics.histcad_2d import (
    HistCADConfig, compute_histcad_2d_freecad,
    compute_histcad_2d_pair_freecad, select_preserved_relations,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import (
    ensure_freecad_imported, load_freecad_commands,
)
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import complete_constraint_values, get_sketch_bbox
from canonical_sketch import (
    build_entity_remap, find_canonical_py, load_sketch_from_fcstd,
    load_sketch_from_py, metric_base_eids, remap_sketch_to_canonical,
)

HUMAN_ROOTS = os.environ.get("VITRUVION_HUMAN_ROOTS", "/home/ryotaro/Downloads/fiverr")
DEFAULT_HUMANS: dict[str, list[tuple[str, str]]] = {
    "0":  [("Johnny",  f"{HUMAN_ROOTS}/Johnny/0"),
           ("Sagar",   f"{HUMAN_ROOTS}/Sagar/0"),
           ("Ryotaro", f"{HUMAN_ROOTS}/Ryotaro/0")],
    "4":  [("Johnny",  f"{HUMAN_ROOTS}/Johnny/4"),
           ("Sagar",   f"{HUMAN_ROOTS}/Sagar/4"),
           ("Tanaka",  f"{HUMAN_ROOTS}/Tanaka/4")],
    "6":  [("Eko",     f"{HUMAN_ROOTS}/Eko/6"),
           ("Johnny",  f"{HUMAN_ROOTS}/Johnny/6"),
           ("Ryotaro", f"{HUMAN_ROOTS}/Ryotaro/6")],
    "8":  [("Eko",     f"{HUMAN_ROOTS}/Eko/8"),
           ("Ryotaro", f"{HUMAN_ROOTS}/Ryotaro/8"),
           ("Tanaka",  f"{HUMAN_ROOTS}/Tanaka/8")],
    "9":  [("Eko",     f"{HUMAN_ROOTS}/Eko/9"),
           ("Sagar",   f"{HUMAN_ROOTS}/Sagar/9"),
           ("Ryotaro", f"{HUMAN_ROOTS}/Ryotaro/9")],
    "11": [("Eko",     f"{HUMAN_ROOTS}/Eko/11"),
           ("Johnny",  f"{HUMAN_ROOTS}/Johnny/11"),
           ("Tanaka",  f"{HUMAN_ROOTS}/Tanaka/11")],
}
DEFAULT_BATCHES: list[str] = ["0", "4", "6", "8", "9", "11"]

# Default Vitruvion model variants for `parse_vitruvion_specs` (used when neither
# --vitruvion-spec nor --vitruvion-root is passed). `base` is SFT Vitruvion at
# epoch 29; `solver_rl` is Casey-style solver-RL at step 56400.
DEFAULT_VITRUVION_SPECS: list[tuple[str, str]] = [
    ("base", f"{HUMAN_ROOTS}/Vitruvion_ori_2026-05-02_20-35-42_epoch_29"),
    ("solver_rl", f"{HUMAN_ROOTS}/Vitruvion_2026-05-03_17-08-23_step_56400"),
]


@dataclass
class FileBundle:
    batch: str
    filename: str
    canonical_sketch: datalib.Sketch
    sketches: dict[str, datalib.Sketch]
    audits: dict[str, dict]
    # Per-label model status / DoF. Keys are Vitruvion variant labels.
    model_statuses: dict[str, str] = field(default_factory=dict)
    model_dofs: dict[str, int | None] = field(default_factory=dict)

    def loaded_model_labels(self) -> list[str]:
        return [lbl for lbl, st in self.model_statuses.items() if st == "loaded"]

    @property
    def model_status(self) -> str:
        """Legacy single-model accessor: returns first label's status or 'missing'."""
        if not self.model_statuses:
            return "missing"
        return next(iter(self.model_statuses.values()))

    @property
    def model_dof(self) -> int | None:
        if not self.model_dofs:
            return None
        return next(iter(self.model_dofs.values()))


# ── shared helpers (from exp_dimensional_dropout_ablation.py) ─────────────

def repo_path(p): return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)

def parse_signs(t):
    s = tuple(float(x) for x in t.split(",") if x.strip())
    return s or (-1.0, 1.0)

def parse_percentages(t):
    v = [float(x) for x in t.split(",") if x.strip()]
    for x in v:
        assert 0 <= x <= 100
    return v

def parse_human_specs(specs):
    if not specs:
        return {b: list(v) for b, v in DEFAULT_HUMANS.items()}
    out: dict[str, list[tuple[str, str]]] = {}
    for s in specs:
        batch, name, path = s.split(":", 2)
        out.setdefault(batch, []).append((name, path))
    return out


def parse_vitruvion_specs(
    specs: list[str] | None,
    legacy_root: str | None = None,
) -> list[tuple[str, str]]:
    """Parse Vitruvion model spec strings.

    Each spec is either 'label:path' (preferred) or just 'path' (legacy,
    labeled 'Vitruvion'). If neither `specs` nor `legacy_root` is provided,
    returns `DEFAULT_VITRUVION_SPECS`.
    """
    if not specs and not legacy_root:
        return list(DEFAULT_VITRUVION_SPECS)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    if legacy_root:
        out.append(("Vitruvion", legacy_root))
        seen.add("Vitruvion")
    for s in (specs or []):
        if ":" in s:
            label, path = s.split(":", 1)
            label = label.strip()
            path = path.strip()
        else:
            label, path = "Vitruvion", s.strip()
        if not label or not path:
            continue
        if label in seen:
            out = [(lbl, p) if lbl != label else (label, path) for lbl, p in out]
        else:
            out.append((label, path))
            seen.add(label)
    return out

def get_fcstd_map(d):
    if not os.path.isdir(d): return {}
    return {f: f for f in os.listdir(d)
            if f.endswith(".FCStd") and not f.endswith("_mid.FCStd")
            and not f.startswith("_")}

def is_dimensional(c): return c.constraint_type.name in DIMENSIONAL_CONSTRAINTS

def base_policy_pass(audit, policy):
    if policy == "none": return True, ""
    if policy == "canonical_solid":
        ok = int(audit["missing_solid"]) == 0
        return ok, "" if ok else "missing canonical solid"
    if policy == "canonical_all":
        ok = int(audit["missing_solid"]) == 0 and int(audit["missing_construction"]) == 0
        return ok, "" if ok else "missing canonical geometry"
    raise ValueError(policy)

def build_base_eids(cs, policy):
    return None if policy == "none" else metric_base_eids(policy, cs)

def vitruvion_solver_status(path):
    FreeCAD, _, _ = ensure_freecad_imported()
    doc = None
    try:
        doc, fc_sketch = load_freecad_commands(path)
        status = fc_sketch.getSolverStatus()
        if isinstance(status, dict):
            return status.get("DoF"), status.get("Status")
        return None, str(status)
    except Exception:
        return None, "load_error"
    finally:
        if doc is not None:
            try: FreeCAD.closeDocument(doc.Name)
            except Exception: pass


_HEADER_RE = re.compile(
    r"Status:\s*(?P<status>\w+).*?Final solver status:\s*(?P<solver>\w+)"
)


def parse_pred_header(path: str) -> tuple[str | None, str | None]:
    """Read the first-line header of a Vitruvion .py output.

    Returns (status_label, final_solver_status), or (None, None) if the
    header is absent / unparseable. status_label is the inference-time
    classification {Solved, UnderConstrained, Distorted, SolverError};
    final_solver_status is FreeCAD's solver verdict on the saved sketch
    {Converged, Redundant, Conflicting, Failed, Malformed, Distorted}.
    """
    try:
        with open(path) as f:
            line = f.readline()
    except OSError:
        return None, None
    m = _HEADER_RE.search(line)
    if not m:
        return None, None
    return m.group("status"), m.group("solver")


# Header status labels that exclude a Vitruvion output from d_feas evaluation.
# Distorted (geometry moved during solve) and SolverError (Redundant /
# Conflicting / Failed / Malformed in the inference-time check) are excluded.
# UnderConstrained and Solved are both accepted: d_feas is well-defined for
# under-constrained outputs (gain residual handles the DoF mismatch).
_REJECTED_HEADER_STATUSES = frozenset({"Distorted", "SolverError"})


def model_path_for(root, batch, filename):
    if root is None: return None
    path = os.path.join(root, batch, os.path.splitext(filename)[0] + ".py")
    return path if os.path.exists(path) else None


def _coordinate_transform_from_canonical(
    canonical_sketch,
    *,
    coordinate_scale: float,
    bbox_normalize_dst_length: float | None,
) -> tuple[float, float, float]:
    """Return (offset_x, offset_y, scale) for x' = (x - offset) * scale."""
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


def build_file_bundle(*, batch, filename, humans, file_maps, converter,
                      geometry_tol, canonical_dir,
                      model_root=None,
                      model_specs: list[tuple[str, str]] | None = None,
                      coordinate_scale: float = 1.0,
                      bbox_normalize_dst_length: float | None = 1.0):
    """Load human sketches plus zero or more model variants into a FileBundle.

    Either `model_root` (legacy, label='Vitruvion') or `model_specs` (a list of
    (label, root) pairs) may be passed. Specs from both sources are merged.
    """
    data_dirs = {n: d for n, d in humans}
    annotators = tuple(data_dirs.keys())
    cp = find_canonical_py(filename, data_dirs, annotators, canonical_dir=canonical_dir)
    canonical = load_sketch_from_py(cp, converter)
    raw = {}
    for name, d in humans:
        raw[name] = load_sketch_from_fcstd(
            os.path.join(d, file_maps[name][filename]), converter)

    specs: list[tuple[str, str]] = []
    if model_root is not None:
        specs.append(("Vitruvion", model_root))
    if model_specs:
        specs.extend(model_specs)

    model_statuses: dict[str, str] = {}
    model_dofs: dict[str, int | None] = {}
    for label, root in specs:
        mp = model_path_for(root, batch, filename)
        if mp is None:
            model_statuses[label] = "missing"
            model_dofs[label] = None
            continue
        header_status, header_solver = parse_pred_header(mp)
        dof, solver_st = vitruvion_solver_status(mp)
        model_dofs[label] = dof
        # Gate: accept Converged outputs whose inference-time header is not
        # Distorted / SolverError. UnderConstrained outputs are admitted
        # because d_feas is well-defined for them (the gain residual handles
        # rank deficiency). The header's "Final solver status" takes
        # precedence when present; we fall back to FreeCAD's runtime status
        # only when the header is missing.
        accepted_solver = (header_solver or solver_st) == "Converged"
        rejected_label = header_status in _REJECTED_HEADER_STATUSES
        if accepted_solver and not rejected_label:
            raw[label] = load_sketch_from_py(mp, converter)
            model_statuses[label] = "loaded"
        else:
            model_statuses[label] = (
                f"filtered(dof={dof},st={solver_st},"
                f"hdr_status={header_status},hdr_solver={header_solver})"
            )

    sketches, audits = {}, {}
    for name, full in raw.items():
        s2n, extras, audit = build_entity_remap(full, canonical, geometry_tol)
        remapped, dropped = remap_sketch_to_canonical(full, canonical, s2n, extras)
        audit["dropped_constraints"] = dropped
        sketches[name] = remapped
        audits[name] = audit
    bundle = FileBundle(batch, filename, canonical, sketches, audits,
                        model_statuses=model_statuses, model_dofs=model_dofs)
    return transform_bundle_coordinates(
        bundle,
        coordinate_scale=coordinate_scale,
        bbox_normalize_dst_length=bbox_normalize_dst_length,
    )

def finite_or_none(v):
    try: x = float(v)
    except (TypeError, ValueError): return None
    return x if math.isfinite(x) else None

def safe_median(vals):
    vs = [float(v) for v in vals if finite_or_none(v) is not None]
    return float(median(vs)) if vs else None

def norm_by_scale(val, scale, eps):
    if val is None or scale is None: return None
    if abs(scale) <= eps: return 0.0 if abs(val) <= eps else math.inf
    return val / scale

def stable_seed(*parts):
    return int.from_bytes(
        hashlib.sha256("::".join(str(p) for p in parts).encode()).digest()[:8],
        "big", signed=False)

def sorted_dim_ids(sketch):
    return sorted(
        [cid for cid, c in sketch.constraints.items() if is_dimensional(c)],
        key=lambda cid: (constraint_signature(sketch.constraints[cid]), cid))

def dropout_variant(sketch, *, drop_rate, seed):
    dim_ids = sorted_dim_ids(sketch)
    n = len(dim_ids)
    nd = max(0, min(n, int(round(n * drop_rate / 100.0))))
    if nd == 0: drop = set()
    elif nd == n: drop = set(dim_ids)
    else: drop = set(random.Random(seed).sample(dim_ids, nd))
    out = copy.deepcopy(sketch)
    out.constraints = {k: v for k, v in out.constraints.items() if k not in drop}
    return out, n, n - nd, nd


# ── source-conditioned probe helpers ──────────────────────────────────────

def _constraint_ref_strings(constraint):
    return [
        str(p.value)
        for p in constraint.parameters
        if isinstance(p, LocalReferenceParameter)
    ]


def _is_point_like_ref(sketch, ref: str) -> bool:
    """True for explicit point references, not edge/object references."""
    eid, sep, _suffix = ref.partition(".")
    ent = sketch.entities.get(eid)
    if isinstance(ent, datalib.Point):
        return True
    return bool(sep)


def is_point_point_coincident(constraint, sketch) -> bool:
    """Coincident between two point-like refs.

    Datalib Coincident also represents Point-On-Object and line-line collinear
    cases. Those are intentionally kept in the preservation set; only
    point-to-point topology coincidences are filtered by the ablation.
    """
    if constraint.constraint_type.name != "Coincident":
        return False
    refs = _constraint_ref_strings(constraint)
    if len(refs) != 2:
        return False
    return all(_is_point_like_ref(sketch, r) for r in refs)


def preservation_relations(sketch, *, exclude_point_point_coincident=False):
    rels = [c for c in sketch.constraints.values() if _is_relational(c)]
    if exclude_point_point_coincident:
        rels = [c for c in rels if not is_point_point_coincident(c, sketch)]
    return rels


def source_probes(sketch, *, exclude_point_point_coincident=False):
    """Extract Q_i (dim probes) and R_i (relational constraints) from source."""
    probes, _ = collect_probes([sketch], kind="dim")
    rels = preservation_relations(
        sketch,
        exclude_point_point_coincident=exclude_point_point_coincident,
    )
    return probes, [1.0] * len(probes), rels, [1.0] * len(rels)


def validate_probes_on_source(source_sketch, probes, rels, context, converter, config):
    """Return bool mask: True if probe is feasible on source for ALL signs."""
    if not probes:
        return []
    result = compute_histcad_2d_freecad(
        sketch=source_sketch, edit_probes=probes, relation_probes=rels,
        context=context, converter=converter, config=config)
    feasible = [True] * len(probes)
    for pr in result.per_probe:
        if not pr.er:
            feasible[pr.probe_index] = False
    return feasible


def classify_edit_types(probes, pred_sketch):
    """For each probe, return 'modify' if signature exists in pred, else 'insert'."""
    pred_sigs = {constraint_signature(c) for c in pred_sketch.constraints.values()}
    return ["modify" if constraint_signature(p) in pred_sigs else "insert"
            for p in probes]


# ── scoring functions ─────────────────────────────────────────────────────

def response_distance(*, pred_sketch, gt_sketch, edit_probes, edit_weights,
                      context, base_eids):
    try:
        j_gt = j_pred = None
        if base_eids is not None:
            j_gt = pairwise_distance_observable(gt_sketch, base_eids=base_eids)
            j_pred = pairwise_distance_observable(pred_sketch, base_eids=base_eids)
        r = compute_edit_response_distance(
            gt_sketch=gt_sketch, pred_sketch=pred_sketch,
            edit_probes=edit_probes, context=context,
            probe_weights=edit_weights,
            observable_jacobian_gt=j_gt, observable_jacobian_pred=j_pred)
        return r.distance, r.n_dim_pred, r.n_dim_gt, r.skipped_probes, ""
    except Exception as e:
        return None, None, None, None, str(e)[:200]


def histcad_score(*, pred_sketch, gt_sketch, edit_probes, rels,
                  context, converter, config):
    try:
        result = compute_histcad_2d_freecad(
            sketch=pred_sketch, edit_probes=edit_probes,
            relation_probes=rels, context=context,
            converter=converter, config=config)
        # Collect ER reason breakdown
        n_modify_fail = n_insert_fail = 0
        pred_sigs = {constraint_signature(c)
                     for c in pred_sketch.constraints.values()}
        for pr in result.per_probe:
            if not pr.er:
                sig = pr.probe_signature
                if sig in pred_sigs:
                    n_modify_fail += 1
                else:
                    n_insert_fail += 1
        return (result.er_rate, result.cpcsr_rate, result.oes_rate,
                result.n_probes, result.n_relations,
                n_modify_fail, n_insert_fail, "")
    except Exception as e:
        return None, None, None, None, None, 0, 0, str(e)[:200]


def edit_attainability(*, sketch, edit_probes, edit_weights, context, eps=1e-8):
    try:
        native = compute_native_response(sketch, context, eps=eps)
    except Exception:
        return None, len(edit_probes)
    total = total_w = 0.0
    skipped = 0
    for probe, w in zip(edit_probes, edit_weights):
        j_q = constraint_jacobian(probe, native.state, context)
        s_q = edit_response_for_probe(native, probe, context, eps=eps)
        if j_q is None or s_q is None:
            skipped += 1; continue
        gain = j_q @ s_q
        I = torch.eye(gain.shape[0], dtype=gain.dtype)
        total += float(w) * float(torch.linalg.norm(gain - I))
        total_w += float(w)
    return (total / total_w if total_w > 0 else None), skipped


def score_pair(*, batch, filename, rotation_id, sample_id, drop_rate,
               pred_name, gt_name, pred_sketch, gt_sketch,
               edit_probes, edit_weights, rels, context, converter,
               histcad_cfg, base_eids, eps, source_validated):
    d_resp, ndp, ndg, skip_r, e1 = response_distance(
        pred_sketch=pred_sketch, gt_sketch=gt_sketch,
        edit_probes=edit_probes, edit_weights=edit_weights,
        context=context, base_eids=base_eids)
    er, cpcsr, oes, nhp, nhr, n_mf, n_if, e2 = histcad_score(
        pred_sketch=pred_sketch, gt_sketch=gt_sketch,
        edit_probes=edit_probes, rels=rels,
        context=context, converter=converter, config=histcad_cfg)
    att, att_skip = edit_attainability(
        sketch=pred_sketch, edit_probes=edit_probes,
        edit_weights=edit_weights, context=context, eps=eps)
    return {
        "batch": batch, "filename": filename,
        "rotation_id": rotation_id, "sample_id": sample_id,
        "drop_rate": drop_rate,
        "pred": pred_name, "gt": gt_name,
        "source_validated": source_validated,
        "n_edit_probes": len(edit_probes),
        "d_resp": d_resp, "n_dim_pred": ndp, "n_dim_gt": ndg,
        "skipped_resp": skip_r,
        "histcad_er": er, "histcad_cpcsr": cpcsr, "histcad_oes": oes,
        "histcad_n_probes": nhp, "histcad_n_relations": nhr,
        "er_modify_fail": n_mf, "er_insert_fail": n_if,
        "pred_attain_error": att, "pred_attain_skipped": att_skip,
        "error": " ".join(e for e in (e1, e2) if e),
    }


# ── rotation evaluation ──────────────────────────────────────────────────

def evaluate_rotation(*, bundle, refs, heldout, drop_rates, samples_per_rate,
                      context, converter, histcad_cfg, base_policy, eps,
                      eval_model=True,
                      exclude_point_point_coincident_preservation=False):
    rotation_id = f"{bundle.batch}:{bundle.filename}:{'+'.join(refs)}__query={heldout}"
    participants = set(refs) | {heldout}
    bad = []
    for n in participants:
        ok, reason = base_policy_pass(bundle.audits[n], base_policy)
        if not ok: bad.append(f"{n}:{reason}")
    if bad:
        return [{"batch": bundle.batch, "filename": bundle.filename,
                 "rotation_id": rotation_id, "error": ",".join(bad)}], []

    heldout_sketch = bundle.sketches[heldout]
    base_eids = build_base_eids(bundle.canonical_sketch, base_policy)

    # Per-source: extract probes, validate feasibility
    source_info = {}  # gt_name -> {probes, weights, rels, rel_weights, valid_mask}
    for gt_name in refs:
        gt_sk = bundle.sketches[gt_name]
        probes, weights, rels, rel_w = source_probes(
            gt_sk,
            exclude_point_point_coincident=(
                exclude_point_point_coincident_preservation
            ),
        )
        mask = validate_probes_on_source(
            gt_sk, probes, rels, context, converter, histcad_cfg)
        source_info[gt_name] = {
            "probes": probes, "weights": weights,
            "rels": rels, "rel_weights": rel_w,
            "valid_mask": mask,
            "n_before_validation": len(probes),
            "n_after_validation": sum(mask) if mask else 0,
        }

    # h_scale per source: d_resp between other humans using source's probes
    h_scales = {}  # (gt_name, validated) -> float|None
    for gt_name in refs:
        other_humans = [n for n in refs if n != gt_name]
        info = source_info[gt_name]
        for validated in (False, True):
            if validated:
                probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
                weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
            else:
                probes, weights = info["probes"], info["weights"]
            if not probes:
                h_scales[(gt_name, validated)] = None
                continue
            ds = []
            for oh in other_humans:
                d, *_ = response_distance(
                    pred_sketch=bundle.sketches[oh],
                    gt_sketch=bundle.sketches[gt_name],
                    edit_probes=probes, edit_weights=weights,
                    context=context, base_eids=base_eids)
                if finite_or_none(d) is not None:
                    ds.append(d)
            h_scales[(gt_name, validated)] = safe_median(ds)

    summary_rows, detail_rows = [], []
    dim_ids = sorted_dim_ids(heldout_sketch)

    for drop_rate in drop_rates:
        sc = 1 if drop_rate in (0.0, 100.0) else samples_per_rate
        for sample_id in range(sc):
            seed = stable_seed(bundle.batch, bundle.filename, rotation_id,
                               drop_rate, sample_id)
            pred_var, n_orig, n_kept, n_dropped = dropout_variant(
                heldout_sketch, drop_rate=drop_rate, seed=seed)

            base_info = {
                "batch": bundle.batch, "filename": bundle.filename,
                "rotation_id": rotation_id, "refs": "|".join(refs),
                "heldout": heldout, "sample_id": sample_id,
                "drop_rate": drop_rate, "keep_rate": 100.0 - drop_rate,
                "n_dim_original": n_orig, "n_dim_kept": n_kept,
                "n_dim_dropped": n_dropped,
            }

            for validated in (False, True):
                vt = validated
                per_source = []
                for gt_name in refs:
                    info = source_info[gt_name]
                    if vt:
                        probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
                        weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
                        rels = info["rels"]
                    else:
                        probes, weights, rels = info["probes"], info["weights"], info["rels"]

                    if not probes:
                        per_source.append(None)
                        continue

                    row = score_pair(
                        batch=bundle.batch, filename=bundle.filename,
                        rotation_id=rotation_id, sample_id=sample_id,
                        drop_rate=drop_rate,
                        pred_name=f"{heldout}_drop{drop_rate:g}",
                        gt_name=gt_name,
                        pred_sketch=pred_var, gt_sketch=bundle.sketches[gt_name],
                        edit_probes=probes, edit_weights=weights, rels=rels,
                        context=context, converter=converter,
                        histcad_cfg=histcad_cfg, base_eids=base_eids,
                        eps=eps, source_validated=vt)
                    row["n_probes_before_validation"] = info["n_before_validation"]
                    row["n_probes_after_validation"] = info["n_after_validation"]
                    row["h_scale"] = h_scales.get((gt_name, vt))
                    per_source.append(row)
                    detail_rows.append(row)

                # Multi-ref aggregate
                valid = [r for r in per_source if r is not None]
                d_vals = [finite_or_none(r.get("d_resp")) for r in valid]
                oes_vals = [finite_or_none(r.get("histcad_oes")) for r in valid]
                er_vals = [finite_or_none(r.get("histcad_er")) for r in valid]
                cpcsr_vals = [finite_or_none(r.get("histcad_cpcsr")) for r in valid]
                att_vals = [finite_or_none(r.get("pred_attain_error")) for r in valid]

                min_d = min([v for v in d_vals if v is not None], default=None)
                max_oes = max([v for v in oes_vals if v is not None], default=None)
                min_att = min([v for v in att_vals if v is not None], default=None)

                # norm_d_resp using h_scale of the best source
                best_src = None
                if min_d is not None:
                    for r in valid:
                        if finite_or_none(r.get("d_resp")) == min_d:
                            best_src = r; break
                hs = best_src.get("h_scale") if best_src else None

                er_mf = sum(r.get("er_modify_fail", 0) for r in valid)
                er_if = sum(r.get("er_insert_fail", 0) for r in valid)

                srow = dict(base_info)
                srow.update({
                    "source_validated": vt,
                    "min_d_resp": min_d,
                    "norm_d_resp": norm_by_scale(min_d, hs, eps),
                    "h_scale": hs,
                    "max_histcad_oes": max_oes,
                    "median_histcad_oes": safe_median(oes_vals),
                    "median_histcad_er": safe_median(er_vals),
                    "median_histcad_cpcsr": safe_median(cpcsr_vals),
                    "min_attain_error": min_att,
                    "median_attain_error": safe_median(att_vals),
                    "total_er_modify_fail": er_mf,
                    "total_er_insert_fail": er_if,
                    "valid_sources": len(valid),
                    "error": "",
                })
                summary_rows.append(srow)

    # ── Evaluate each loaded Vitruvion variant if available ──
    if eval_model:
        for model_label in bundle.loaded_model_labels():
            model_ok, model_reason = base_policy_pass(
                bundle.audits[model_label], base_policy)
            if not model_ok:
                continue
            model_sketch = bundle.sketches[model_label]
            for validated in (False, True):
                vt = validated
                per_source = []
                for gt_name in refs:
                    info = source_info[gt_name]
                    if vt:
                        probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
                        weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
                    else:
                        probes, weights = info["probes"], info["weights"]
                    rels = info["rels"]
                    if not probes:
                        per_source.append(None); continue
                    row = score_pair(
                        batch=bundle.batch, filename=bundle.filename,
                        rotation_id=rotation_id, sample_id=0,
                        drop_rate=-1,  # sentinel: model, not dropout
                        pred_name=model_label, gt_name=gt_name,
                        pred_sketch=model_sketch,
                        gt_sketch=bundle.sketches[gt_name],
                        edit_probes=probes, edit_weights=weights, rels=rels,
                        context=context, converter=converter,
                        histcad_cfg=histcad_cfg, base_eids=base_eids,
                        eps=eps, source_validated=vt)
                    row["n_probes_before_validation"] = info["n_before_validation"]
                    row["n_probes_after_validation"] = info["n_after_validation"]
                    row["h_scale"] = h_scales.get((gt_name, vt))
                    per_source.append(row)
                    detail_rows.append(row)

                valid = [r for r in per_source if r is not None]
                d_vals = [finite_or_none(r.get("d_resp")) for r in valid]
                oes_vals = [finite_or_none(r.get("histcad_oes")) for r in valid]
                att_vals = [finite_or_none(r.get("pred_attain_error")) for r in valid]
                min_d = min([v for v in d_vals if v is not None], default=None)
                max_oes = max([v for v in oes_vals if v is not None], default=None)
                min_att = min([v for v in att_vals if v is not None], default=None)
                best_src = None
                if min_d is not None:
                    for r in valid:
                        if finite_or_none(r.get("d_resp")) == min_d:
                            best_src = r; break
                hs = best_src.get("h_scale") if best_src else None
                srow = {
                    "batch": bundle.batch, "filename": bundle.filename,
                    "rotation_id": rotation_id, "refs": "|".join(refs),
                    "heldout": heldout, "sample_id": 0,
                    "drop_rate": -1, "keep_rate": -1,
                    "n_dim_original": -1, "n_dim_kept": -1, "n_dim_dropped": -1,
                    "source_validated": vt,
                    "pred": model_label,
                    "min_d_resp": min_d,
                    "norm_d_resp": norm_by_scale(min_d, hs, eps),
                    "h_scale": hs,
                    "max_histcad_oes": max_oes,
                    "median_histcad_oes": safe_median(oes_vals),
                    "median_histcad_er": safe_median(
                        [finite_or_none(r.get("histcad_er")) for r in valid]),
                    "median_histcad_cpcsr": safe_median(
                        [finite_or_none(r.get("histcad_cpcsr")) for r in valid]),
                    "min_attain_error": min_att,
                    "median_attain_error": safe_median(att_vals),
                    "total_er_modify_fail": sum(r.get("er_modify_fail", 0) for r in valid),
                    "total_er_insert_fail": sum(r.get("er_insert_fail", 0) for r in valid),
                    "valid_sources": len(valid),
                    "error": "",
                }
                summary_rows.append(srow)

    return summary_rows, detail_rows


def write_csv(path, rows):
    full = repo_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with open(full, "w", newline="") as f:
        if not keys: return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {full}")


def print_summary(rows):
    valid = [r for r in rows if not r.get("error")]
    print("\n--- Source-conditioned dropout ablation ---")
    for vt in (False, True):
        vrows = [r for r in valid if r.get("source_validated") == vt]
        tag = "validated" if vt else "unvalidated"
        print(f"\n  [{tag}] n={len(vrows)}")
        # Group dropout rows by drop_rate; group model rows (rate=-1) by pred label.
        groups: dict[tuple, list[dict]] = {}
        for r in vrows:
            rate = float(r["drop_rate"])
            if rate == -1:
                key = (rate, r.get("pred") or "Vitruvion")
            else:
                key = (rate, None)
            groups.setdefault(key, []).append(r)
        for (rate, pred_label) in sorted(groups, key=lambda k: (k[0], k[1] or "")):
            rs = groups[(rate, pred_label)]
            label = (pred_label or "Vitruvion") if rate == -1 else f"drop={rate:5.1f}%"
            print(f"    {label:14s} n={len(rs):4d}"
                  f" med_OES={safe_median([r.get('max_histcad_oes') for r in rs])}"
                  f" med_ER={safe_median([r.get('median_histcad_er') for r in rs])}"
                  f" med_norm_d={safe_median([r.get('norm_d_resp') for r in rs])}"
                  f" med_att={safe_median([r.get('min_attain_error') for r in rs])}"
                  f" er_mod_fail={sum(r.get('total_er_modify_fail',0) for r in rs)}"
                  f" er_ins_fail={sum(r.get('total_er_insert_fail',0) for r in rs)}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--vitruvion-root",
        default=None,
        help=(
            "Legacy single-model root. Prefer --vitruvion-spec. "
            "If neither flag is given, evaluates DEFAULT_VITRUVION_SPECS "
            "(base + solver_rl)."
        ),
    )
    p.add_argument(
        "--vitruvion-spec",
        action="append",
        default=None,
        help=(
            "label:path, repeatable. e.g. "
            "--vitruvion-spec base:/path/to/Vitruvion_ori_... "
            "--vitruvion-spec solver_rl:/path/to/Vitruvion_2026-05-03_..."
        ),
    )
    p.add_argument("--batches", nargs="+", default=list(DEFAULT_BATCHES))
    p.add_argument("--human", action="append", default=None)
    p.add_argument("--ref-size", type=int, default=2)
    p.add_argument("--drop-rates", default="100,75,50,25,0")
    p.add_argument("--samples-per-rate", type=int, default=1)
    p.add_argument("--base-policy", default="canonical_solid",
                   choices=("none", "canonical_solid", "canonical_all"))
    p.add_argument("--geometry-tol", type=float, default=5e-1)
    p.add_argument("--canonical-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--coordinate-scale", type=float, default=1.0)
    p.add_argument(
        "--bbox-normalize-dst-length",
        type=float,
        default=1.0,
        help="Center by canonical bbox and rescale its long axis to this length.",
    )
    p.add_argument(
        "--no-bbox-normalization",
        action="store_true",
        help="Use raw coordinates except for any --coordinate-scale multiplier.",
    )
    p.add_argument("--delta-basis", default="bbox_diag")
    p.add_argument("--delta-scale", type=float, default=0.01)
    p.add_argument("--delta-absolute", type=float, default=1.0)
    p.add_argument("--delta-signs", default="-1,+1")
    p.add_argument("--newton-max-iter", type=int, default=80)
    p.add_argument("--newton-tol", type=float, default=1e-6)
    p.add_argument("--er-target-tol", type=float, default=1e-3)
    p.add_argument("--relation-tol", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument(
        "--exclude-point-point-coincident-preservation",
        action="store_true",
        help=(
            "Remove only point-point Coincident constraints from HistCAD "
            "preservation sets. Point-line and line-line Coincident remain."
        ),
    )
    p.add_argument("--output-csv",
                   default="artifacts/source_conditioned_dropout_scores.csv")
    p.add_argument("--details-csv",
                   default="artifacts/source_conditioned_dropout_details.csv")
    return p.parse_args()


def main():
    args = parse_args()
    humans_by_batch = parse_human_specs(args.human)
    drop_rates = parse_percentages(args.drop_rates)
    model_specs = parse_vitruvion_specs(args.vitruvion_spec, args.vitruvion_root)
    if model_specs:
        print("[Vitruvion specs] " + ", ".join(f"{l}={p}" for l, p in model_specs))
    cfg = HistCADConfig(
        delta_basis=args.delta_basis, delta_scale=args.delta_scale,
        delta_absolute=args.delta_absolute,
        delta_signs=parse_signs(args.delta_signs),
        newton_max_iter=args.newton_max_iter, newton_tol=args.newton_tol,
        er_target_tol=args.er_target_tol,
        relation_tol_default=args.relation_tol, relation_source="gt")

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    summary_rows, detail_rows = [], []
    n_files = 0
    bbox_dst = None if args.no_bbox_normalization else args.bbox_normalize_dst_length

    for batch in args.batches:
        humans = [(n, d) for n, d in humans_by_batch.get(batch, [])
                  if os.path.isdir(d)]
        if len(humans) < args.ref_size + 1:
            print(f"[skip] batch {batch}: need {args.ref_size+1} humans")
            continue
        file_maps = {n: get_fcstd_map(d) for n, d in humans}
        common = sorted(set.intersection(*(set(m) for m in file_maps.values())))
        print(f"Batch {batch}: "
              + " | ".join(f"{n}={len(file_maps[n])}" for n, _ in humans)
              + f" | common={len(common)}")

        for filename in common:
            if args.limit is not None and n_files >= args.limit:
                break
            n_files += 1
            try:
                bundle = build_file_bundle(
                    batch=batch, filename=filename, humans=humans,
                    file_maps=file_maps, converter=converter,
                    geometry_tol=args.geometry_tol,
                    canonical_dir=args.canonical_dir,
                    model_specs=model_specs,
                    coordinate_scale=args.coordinate_scale,
                    bbox_normalize_dst_length=bbox_dst)
            except Exception as e:
                print(f"  [skip] {batch}/{filename}: {e}")
                continue

            names = tuple(n for n, _ in humans)
            n_rot = 0
            for refs in combinations(names, args.ref_size):
                for heldout in names:
                    if heldout in refs: continue
                    srows, drows = evaluate_rotation(
                        bundle=bundle, refs=tuple(refs), heldout=heldout,
                        drop_rates=drop_rates,
                        samples_per_rate=args.samples_per_rate,
                        context=context, converter=converter,
                        histcad_cfg=cfg, base_policy=args.base_policy,
                        eps=args.eps,
                        exclude_point_point_coincident_preservation=(
                            args.exclude_point_point_coincident_preservation
                        ))
                    for row in srows:
                        row["coordinate_scale"] = args.coordinate_scale
                        row["bbox_normalize_dst_length"] = bbox_dst
                    for row in drows:
                        row["coordinate_scale"] = args.coordinate_scale
                        row["bbox_normalize_dst_length"] = bbox_dst
                    summary_rows.extend(srows)
                    detail_rows.extend(drows)
                    n_rot += 1
            print(f"  [done] {batch}/{filename}: {n_rot} rotations")

        if args.limit is not None and n_files >= args.limit:
            break

    print_summary(summary_rows)
    write_csv(args.output_csv, summary_rows)
    write_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
