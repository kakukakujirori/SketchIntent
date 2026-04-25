import copy
import glob
import json
import os
import sys

sys.path.append("../")
import sketchgraphs.data as datalib
from src.metrics.constraint_bleu import ConstraintBLEU
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_freecad_commands, load_FCStd_sketch
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import normalize_sketch, complete_constraint_values


FreeCAD, _, _ = ensure_freecad_imported()

"""
Verify if BLEU is working as expected.
"""
context = SketchContext()
converter = SketchFreecadConverter(context, raise_on_error=True)
bleu_preservable_removed = ConstraintBLEU(midpoint_tol=0.05)
bleu_auxiliary_removed = ConstraintBLEU(midpoint_tol=0.05)


def load_sketch(filepath: str) -> datalib.Sketch:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{filepath} not found")
    doc = None
    try:

        if filepath.endswith(".FCStd"):
            doc, fc_sketch = load_FCStd_sketch(filepath)
        elif filepath.endswith(".py"):
            doc, fc_sketch = load_freecad_commands(filepath)
        else:
            raise ValueError(f"{filepath} doesn't end with `.FCStd` nor `.py`")

        sketch = converter.from_freecad(fc_sketch)
        return sketch
    finally:
        FreeCAD.closeDocument(doc.Name) if doc else None


def align_sketch(target_sketch, base_entities_dict):
    """
    共通のエンティティを基準に、target_sketch 全体を異方性スケーリング・平行移動して整列させる。
    """
    from src.sketch_handlers.sketch_utils import get_sketch_bbox, complete_constraint_values
    from src.metrics.constraint_bleu import is_same_entity

    new_sketch = copy.deepcopy(target_sketch)
    base_ids = set(base_entities_dict.keys())

    # 1. 共通エンティティの BBox を計算（Target側と Base側）
    target_common = {eid: new_sketch.entities[eid] for eid in base_ids if eid in new_sketch.entities}

    t_bbox = get_sketch_bbox(datalib.Sketch(entities=target_common))
    tx_min, ty_min = t_bbox[0]
    tx_max, ty_max = t_bbox[1]

    b_bbox = get_sketch_bbox(datalib.Sketch(entities=base_entities_dict))
    bx_min, by_min = b_bbox[0]
    bx_max, by_max = b_bbox[1]

    # 中心点とサイズ（幅・高さ）を算出
    tcx, tcy = (tx_min + tx_max) / 2, (ty_min + ty_max) / 2
    bcx, bcy = (bx_min + bx_max) / 2, (by_min + by_max) / 2
    tw, th = (tx_max - tx_min), (ty_max - ty_min)
    bw, bh = (bx_max - bx_min), (by_max - by_min)

    # X/Y 独立したスケーリング係数
    fx = bw / tw if tw > 1e-9 else 1.0
    fy = bh / th if th > 1e-9 else 1.0

    # 2. スケッチ全体に変換を適用
    for ent in new_sketch.entities.values():
        for px in ['pntX', 'xCenter', 'x']:
            if hasattr(ent, px):
                val = getattr(ent, px)
                setattr(ent, px, (val - tcx) * fx + bcx)
        for py in ['pntY', 'yCenter', 'y']:
            if hasattr(ent, py):
                val = getattr(ent, py)
                setattr(ent, py, (val - tcy) * fy + bcy)

        # dirX/dirY: start_point = pntX + startParam*dirX なので dirもスケールが必要
        if hasattr(ent, 'dirX'):
            ent.dirX = ent.dirX * fx
        if hasattr(ent, 'dirY'):
            ent.dirY = ent.dirY * fy

        if hasattr(ent, 'radius'):
            ent.radius = ent.radius * (fx + fy) / 2.0

    # 3. 共通エンティティを base_entities_dict で上書きして完全に一致させる
    for eid, base_ent in base_entities_dict.items():
        if eid in new_sketch.entities:
            new_ent = copy.deepcopy(base_ent)
            new_ent.entityId = eid
            new_sketch.entities[eid] = new_ent

    # 4. 寸法制約の再計算
    new_sketch, _ = complete_constraint_values(new_sketch)

    # 5. [Test] 共通エンティティが一致しているか確認
    for eid in base_ids:
        if eid in new_sketch.entities:
            if not is_same_entity(new_sketch.entities[eid], base_entities_dict[eid]):
                raise ValueError(f"Alignment Test Failed: Entity '{eid}' does not match base reference.")

    return new_sketch


if __name__ == '__main__':

    gt1_folder = "2_eko_old"
    gt2_folder = "2_mine_old"
    preservable_removed_folder = "2_mine_old_preservable_removed"
    auxiliary_removed_folder = "2_mine_old_auxiliary_removed"

    for filepath in sorted(glob.glob(os.path.join(gt1_folder, "*.FCStd"))):
        basename = os.path.basename(filepath)
        print(f"{basename=}")

        # Load sketches
        gt1_path = os.path.join(gt1_folder, basename)
        gt2_path = os.path.join(gt2_folder, basename)
        preservable_removed_path = os.path.join(preservable_removed_folder, basename)
        auxiliary_removed_path = os.path.join(auxiliary_removed_folder, basename)

        sketch_base = load_sketch(gt1_path.replace(".FCStd", ".py"))
        sketch_gt1 = load_sketch(gt1_path)
        sketch_gt2 = load_sketch(gt2_path)
        sketch_preservable_removed = load_sketch(preservable_removed_path)
        sketch_auxiliary_removed = load_sketch(auxiliary_removed_path)

        sketch_gt1 = align_sketch(sketch_gt1, sketch_base.entities)
        sketch_gt2 = align_sketch(sketch_gt2, sketch_base.entities)
        sketch_preservable_removed = align_sketch(sketch_preservable_removed, sketch_base.entities)
        sketch_auxiliary_removed = align_sketch(sketch_auxiliary_removed, sketch_base.entities)

        # Calculate BLEU
        bleu_metrics_preservable_removed = bleu_preservable_removed(
            base_prims=sketch_base.entities,
            preds=sketch_preservable_removed,
            targets=[sketch_gt1, sketch_gt2],
        )
        bleu_metrics_auxiliary_removed = bleu_auxiliary_removed(
            base_prims=sketch_base.entities,
            preds=sketch_auxiliary_removed,
            targets=[sketch_gt1, sketch_gt2],
        )


    def to_numpy(x):
        import torch
        if isinstance(x, torch.Tensor):
            return x.item()
        if isinstance(x, list):
            return [to_numpy(y) for y in x]
        if isinstance(x, tuple):
            return (to_numpy(y) for y in x)
        if isinstance(x, dict):
            return {k: to_numpy(v) for k, v in x.items()}
        raise NotImplementedError


    print("================")
    print("\nPreservable Removed")
    print(json.dumps(to_numpy(bleu_preservable_removed.compute()), indent=2))
    print("\nAuxiliary Removed")
    print(json.dumps(to_numpy(bleu_auxiliary_removed.compute()), indent=2))