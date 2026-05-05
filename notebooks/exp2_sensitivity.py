import os
import sys
import copy
import random
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/home/sturk/my_works/My-Vitruvion')

import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ns_mod = _load_module('null_space_distance', '/home/sturk/my_works/My-Vitruvion/src/metrics/null_space_distance.py')
compute_null_space_distance = _ns_mod.compute_null_space_distance
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import complete_constraint_values
FreeCAD, _, _ = ensure_freecad_imported()

EKO = '/home/sturk/Downloads/fiverr/Eko/9'

# 7 Match files identified in Exp 1
match_files = [
    '122579_2663ded3_0000_sketch04.FCStd',
    '137627_576c94b5_0000_sketch01.FCStd',
    '144142_88858328_0000_sketch06.FCStd',
    '21337_062b39fa_0001_sketch04.FCStd',
    '22645_1ba0af00_0005_sketch00.FCStd',
    '24603_a4021250_0000_sketch00.FCStd',
    '88219_05eb53c9_0000_sketch01.FCStd'
]

def get_relational_sketch(sketch):
    DIM_C = {'Distance', 'Length', 'Diameter', 'Radius', 'Angle', 'Offset', 'Centerline_Dimension', 'Block', 'InternalAlignment'}
    sk = copy.deepcopy(sketch)
    sk.constraints = {cid: c for cid, c in sk.constraints.items() if c.constraint_type.name not in DIM_C}
    return sk

context = SketchContext()
converter = SketchFreecadConverter(context, verbose=False)

# results[num_dropped] = list of scores
results = defaultdict(list)
random.seed(42)

print(f"| {'Filename':<30} | {'Base Rank':<10} | {'Drop 1':<8} | {'Drop 2':<8} | {'Drop 3':<8} |")
print("|--------------------------------|------------|----------|----------|----------|")

for filename in match_files:
    path = os.path.join(EKO, filename)
    doc, sketch_fc = load_FCStd_sketch(path)
    sk_full = converter.from_freecad(sketch_fc)
    sk_full, _ = complete_constraint_values(sk_full)
    sk_gt = get_relational_sketch(sk_full)
    FreeCAD.closeDocument(doc.Name)

    base_eids = [eid for eid, ent in sk_gt.entities.items() if not getattr(ent, 'isConstruction', False)]
    
    # Calculate baseline to get rank
    res_base = compute_null_space_distance(sk_gt, sk_gt, context, base_eids=base_eids, skip_pred_primary_extraction=True)
    gt_rank = res_base['gt_row_dim']
    
    row_scores = []
    rel_cids = list(sk_gt.constraints.keys())
    
    for n_drop in range(1, 4):
        if len(rel_cids) <= n_drop:
            row_scores.append(0.0)
            continue
            
        # Try multiple random drops to be stable
        temp_scores = []
        for _ in range(3):
            dropped_cids = random.sample(rel_cids, n_drop)
            sk_pred = copy.deepcopy(sk_gt)
            for cid in dropped_cids:
                del sk_pred.constraints[cid]
            
            res = compute_null_space_distance(sk_gt, sk_pred, context, base_eids=base_eids, skip_pred_primary_extraction=True)
            temp_scores.append(res['intent_recall'])
        
        avg_recall = sum(temp_scores) / len(temp_scores)
        row_scores.append(avg_recall)
        results[n_drop].append(avg_recall)

    print(f"| {filename[:30]:<30} | {gt_rank:<10} | {row_scores[0]:.4f}   | {row_scores[1]:.4f}   | {row_scores[2]:.4f}   |")

print("|--------------------------------|------------|----------|----------|----------|")
mean_1 = np.mean(results[1])
mean_2 = np.mean(results[2])
mean_3 = np.mean(results[3])
print(f"| {'MEAN':<30} | {'-':<10} | {mean_1:.4f}   | {mean_2:.4f}   | {mean_3:.4f}   |")
