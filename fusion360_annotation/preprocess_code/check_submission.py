import argparse
import copy
import glob
import os
from tqdm import tqdm

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch, load_freecad_commands
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter

FreeCAD, Part, Sketcher = ensure_freecad_imported()


def test_constraints(doc: 'FreeCAD.Document', fc_sketch: 'Sketcher.SketchObject') -> list[str]:
    if fc_sketch is None:
        raise ValueError("No sketch found")

    errors = []

    # 1. Tautology Detection (e.g., Equal(A,A))
    for i, c in enumerate(fc_sketch.Constraints):
        if c.Type in ["Equal"] and c.First == c.Second and c.First != -2000:
            errors.append(f"[TAUTOLOGY] #{i + 1} {c.Type}(Edge{c.First + 1}, Edge{c.Second + 1})")

    # 2. External Reference Detection (e.g., -1=X-axis, -2=Y-axis)
    for i, c in enumerate(fc_sketch.Constraints):
        for r in [c.First, c.Second, c.Third]:
            if r in (-1, -2):
                errors.append(f"[EXT REF] #{i + 1} {c.Type} references {'X-axis' if r == -1 else 'Y-axis'}")
                break

    # 3. Check if there is only one Block constraint, and if removing it results in DoF=2
    blocks = [(i, c) for i, c in enumerate(fc_sketch.Constraints) if c.Type == "Block"]
    if len(blocks) != 1:
        errors.append(f"[BLOCK] Block constraint count = {len(blocks)} (expected 1)")
    else:
        bi, bc = blocks[0]
        fc_sketch.delConstraint(bi)
        doc.recompute()
        dof_without_block = fc_sketch.DoF
        fc_sketch.addConstraint(Sketcher.Constraint("Block", bc.First))
        doc.recompute()
        if dof_without_block != 2:
            errors.append(f"[BLOCK] DoF without Block = {dof_without_block} (expected 2)")

    return errors


def align_sketch(target_sketch, base_entities_dict):
    """
    Align the whole target_sketch by anisotropic scaling and translation,
    using the common entities as the reference.
    """
    from src.sketch_handlers.sketch_utils import get_sketch_bbox, complete_constraint_values

    new_sketch = copy.deepcopy(target_sketch)
    base_ids = set(base_entities_dict.keys())

    # 1. Compute the BBox of the common entities (Target side and Base side)
    target_common = {eid: new_sketch.entities[eid] for eid in base_ids if eid in new_sketch.entities}

    t_bbox = get_sketch_bbox(datalib.Sketch(entities=target_common))
    tx_min, ty_min = t_bbox[0]
    tx_max, ty_max = t_bbox[1]

    b_bbox = get_sketch_bbox(datalib.Sketch(entities=base_entities_dict))
    bx_min, by_min = b_bbox[0]
    bx_max, by_max = b_bbox[1]

    # Compute center points and sizes (width / height)
    tcx, tcy = (tx_min + tx_max) / 2, (ty_min + ty_max) / 2
    bcx, bcy = (bx_min + bx_max) / 2, (by_min + by_max) / 2
    tw, th = (tx_max - tx_min), (ty_max - ty_min)
    bw, bh = (bx_max - bx_min), (by_max - by_min)

    # Independent X/Y scaling factors
    fx = bw / tw if tw > 1e-9 else 1.0
    fy = bh / th if th > 1e-9 else 1.0

    # [Check] make sure the anisotropic scale ratio is not too extreme.
    # A wildly different horizontal/vertical scale ratio may mean the geometric
    # structure has changed.
    SCALE_RATIO_TOL = 1.10  # allowed range of the horizontal/vertical scale ratio (10%)
    scale_ratio = fx / fy if fy > 1e-9 else float('inf')
    if not (1.0 / SCALE_RATIO_TOL <= scale_ratio <= SCALE_RATIO_TOL):
        raise ValueError(
            f"Anisotropic scale ratio fx/fy={scale_ratio:.4f} exceeds tolerance "
            f"(allowed: {1/SCALE_RATIO_TOL:.2f}–{SCALE_RATIO_TOL:.2f}). "
            f"Sketch geometry may have been distorted."
        )

    # 2. Apply the transform to the whole sketch
    for ent in new_sketch.entities.values():
        for px in ['pntX', 'xCenter', 'x']:
            if hasattr(ent, px):
                val = getattr(ent, px)
                setattr(ent, px, (val - tcx) * fx + bcx)
        for py in ['pntY', 'yCenter', 'y']:
            if hasattr(ent, py):
                val = getattr(ent, py)
                setattr(ent, py, (val - tcy) * fy + bcy)

        if hasattr(ent, 'radius'):
            ent.radius = ent.radius * (fx + fy) / 2.0

    # 3. Overwrite the common entities with base_entities_dict so they match exactly, and validate added entities
    missing_ids = base_ids - set(new_sketch.entities.keys())
    if missing_ids:
        raise ValueError(f"Original primitive(s) missing in target sketch: {missing_ids}. Primitives cannot be deleted.")

    # [Check] verify that each transformed common entity is consistent with the original entity.
    # NOTE: this check MUST run before the overwrite (below). If compared after
    #       the overwrite, we would be comparing a copy of base_ent against
    #       base_ent itself, which is a vacuous validation that always passes.
    bbox_size = max(bw, bh) if max(bw, bh) > 1e-9 else 1.0
    POS_TOL_NORMALIZED = 0.015  # allowed positional error relative to bbox size (1.5%)
    pos_tol_abs = POS_TOL_NORMALIZED * bbox_size

    for eid in base_ids:
        tgt_ent = new_sketch.entities[eid]
        base_ent = base_entities_dict[eid]
        if type(tgt_ent) != type(base_ent):
            raise ValueError(f"Entity type changed for ID '{eid}': {type(base_ent).__name__} => {type(tgt_ent).__name__}")
        for attr in ['pntX', 'pntY', 'xCenter', 'yCenter', 'x', 'y']:
            if hasattr(tgt_ent, attr) and hasattr(base_ent, attr):
                err = abs(getattr(tgt_ent, attr) - getattr(base_ent, attr))
                if err > pos_tol_abs:
                    raise ValueError(
                        f"Relative structure drift detected for entity '{eid}' attribute '{attr}': "
                        f"error={err:.6f} exceeds tolerance={pos_tol_abs:.6f} "
                        f"(={POS_TOL_NORMALIZED*100:.1f}% of bbox size={bbox_size:.4f}). "
                        f"Internal proportions or relative positions must be preserved."
                    )

    for eid, ent in list(new_sketch.entities.items()):
        if eid in base_ids:
            new_ent = copy.deepcopy(base_entities_dict[eid])
            new_ent.entityId = eid
            new_sketch.entities[eid] = new_ent
        else:
            # Newly added geometries must be construction
            if not eid.startswith('-') and getattr(ent, 'isConstruction', False) == False:
                raise ValueError(f"Added primitive ID '{eid}' ({type(ent).__name__}) must be a construction geometry.")

    # 4. Recompute the dimensional constraints
    new_sketch, _ = complete_constraint_values(new_sketch)

    return new_sketch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    context = SketchContext()
    converter = SketchFreecadConverter(context, raise_on_error=True)

    files = sorted([x for x in glob.glob(os.path.join(args.dir, "*.FCStd")) if not x.endswith("_mid.FCStd")])
    passed = 0
    for f in tqdm(files, disable=True):
        f_ori_py = f.replace(".FCStd", ".py")
        if not os.path.isfile(f_ori_py):
            print(f"ERROR: Failed to open {f_ori_py}")
            continue

        doc_tgt, fc_sketch_tgt = load_FCStd_sketch(f)
        doc_ori, fc_sketch_ori = load_freecad_commands(f_ori_py)
        if fc_sketch_ori is None:
            print(f"ERROR: Failed to open {f_ori_py}")
            continue
        if fc_sketch_tgt is None:
            print(f"ERROR: Failed to open {f}")
            continue

        errors = []
        errors += test_constraints(doc_tgt, fc_sketch_tgt)
        try:
            sketch_ori = converter.from_freecad(fc_sketch_ori)
            sketch_tgt = converter.from_freecad(fc_sketch_tgt)
            sketch_tgt = align_sketch(sketch_tgt, sketch_ori.entities)
        except Exception as e:
            errors.append(f"ERROR: Failed to align {f}: {e}")

        if errors:
            print(f"\nERROR in {f}:")
            for e in errors:
                print(e)
        else:
            passed += 1

        FreeCAD.closeDocument(doc_ori.Name)
        FreeCAD.closeDocument(doc_tgt.Name)

    print(f"\nTest passed: {passed}/{len(files)}")