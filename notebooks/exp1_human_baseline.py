import argparse
import os
import sys
import copy
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/home/sturk/my_works/My-Vitruvion')

from src.metrics.null_space_distance import compute_null_space_distance
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import complete_constraint_values
FreeCAD, _, _ = ensure_freecad_imported()

EKO = '/home/sturk/Downloads/fiverr/Eko/9'
SAGAR = '/home/sturk/Downloads/fiverr/Sagar/9'
RYOTARO = '/home/sturk/Downloads/fiverr/Ryotaro/9'

DIMENSIONAL_CONSTRAINTS = {
    'Distance', 'Length', 'Diameter', 'Radius', 'Angle', 'Offset',
    'Centerline_Dimension', 'Minor_Diameter', 'Major_Diameter', 'Rho',
    'Block', 'InternalAlignment'
}

def get_relational_sketch(sketch):
    sk = copy.deepcopy(sketch)
    sk.constraints = {cid: c for cid, c in sk.constraints.items()
                      if c.constraint_type.name not in DIMENSIONAL_CONSTRAINTS}
    return sk

def check_primitives_match(sk1, sk2):
    k1 = list(sk1.entities.keys())
    k2 = list(sk2.entities.keys())
    if k1 != k2:
        return False
    for k in k1:
        e1 = sk1.entities[k]
        e2 = sk2.entities[k]
        if type(e1) != type(e2): return False
        if getattr(e1, 'isConstruction', False) != getattr(e2, 'isConstruction', False): return False
    return True

def get_files(d):
    return {f for f in os.listdir(d) if f.endswith('.FCStd') and not f.endswith('_mid.FCStd') and not f.startswith('_')}

def _summarize_pair(results):
    linear = np.array([r['linear_intent_recall'] for r in results], dtype=float)
    quadratic = np.array([r['quadratic_intent_recall'] for r in results], dtype=float)
    two_jet = np.array([r['two_jet_intent_recall'] for r in results], dtype=float)
    gt_linear_dim = np.array([r['gt_row_dim'] for r in results], dtype=float)
    pred_linear_dim = np.array([r['pred_row_dim'] for r in results], dtype=float)
    gt_quad_dim = np.array([r['gt_quadratic_dim'] for r in results], dtype=float)
    pred_quad_dim = np.array([r['pred_quadratic_dim'] for r in results], dtype=float)
    return {
        'n': len(results),
        'linear_mean': float(np.mean(linear)),
        'linear_std': float(np.std(linear)),
        'quadratic_mean': float(np.mean(quadratic)),
        'quadratic_std': float(np.std(quadratic)),
        'two_jet_mean': float(np.mean(two_jet)),
        'two_jet_std': float(np.std(two_jet)),
        'gt_linear_dim': float(np.mean(gt_linear_dim)),
        'pred_linear_dim': float(np.mean(pred_linear_dim)),
        'gt_quad_dim': float(np.mean(gt_quad_dim)),
        'pred_quad_dim': float(np.mean(pred_quad_dim)),
    }


def _print_results(title, bucket):
    print(f"\n--- {title} ---")
    for pred_ann in ['Eko', 'Sagar', 'Ryotaro']:
        for gt_ann in ['Eko', 'Sagar', 'Ryotaro']:
            if pred_ann == gt_ann:
                continue
            pair_results = bucket[pred_ann][gt_ann]
            if not pair_results:
                continue
            s = _summarize_pair(pair_results)
            print(
                f"Pred: {pred_ann:7s} | GT: {gt_ann:7s} | "
                f"Linear = {s['linear_mean']:.4f} (±{s['linear_std']:.4f}) | "
                f"Quadratic = {s['quadratic_mean']:.4f} (±{s['quadratic_std']:.4f}) | "
                f"2-Jet = {s['two_jet_mean']:.4f} (±{s['two_jet_std']:.4f}) | "
                f"d_lin GT/Pred = {s['gt_linear_dim']:.1f}/{s['pred_linear_dim']:.1f} | "
                f"d_quad GT/Pred = {s['gt_quad_dim']:.1f}/{s['pred_quad_dim']:.1f} | "
                f"n = {s['n']:2d}"
            )


def main():
    parser = argparse.ArgumentParser(description="Human baseline for linear/quadratic intent recall.")
    parser.add_argument("--eko", default=EKO)
    parser.add_argument("--sagar", default=SAGAR)
    parser.add_argument("--ryotaro", default=RYOTARO)
    parser.add_argument(
        "--include-mismatch",
        action="store_true",
        help="Also print primitive-mismatch cases. By default only Primitives Identical cases are reported.",
    )
    args = parser.parse_args()

    data_dirs = {
        'Eko': args.eko,
        'Sagar': args.sagar,
        'Ryotaro': args.ryotaro,
    }

    common = get_files(data_dirs['Eko']) & get_files(data_dirs['Sagar']) & get_files(data_dirs['Ryotaro'])
    common = sorted(list(common))

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)

    # Store scores as scores[match_key][pred_ann][gt_ann]
    scores = {
        'match': {
            'Eko': defaultdict(list),
            'Sagar': defaultdict(list),
            'Ryotaro': defaultdict(list)
        },
        'mismatch': {
            'Eko': defaultdict(list),
            'Sagar': defaultdict(list),
            'Ryotaro': defaultdict(list)
        }
    }

    for filename in common:
        sketches_full = {}
        docs = []

        try:
            for annotator, d in data_dirs.items():
                path = os.path.join(d, filename)
                doc, sketch_fc = load_FCStd_sketch(path)
                docs.append(doc)
                sk_data = converter.from_freecad(sketch_fc)
                sk_data, _ = complete_constraint_values(sk_data)
                sketches_full[annotator] = sk_data

            sketches_rel = {ann: get_relational_sketch(sk) for ann, sk in sketches_full.items()}

            # 3-way match check (all three must be identical including construction)
            is_3way_match = (
                check_primitives_match(sketches_rel['Eko'], sketches_rel['Sagar']) and
                check_primitives_match(sketches_rel['Sagar'], sketches_rel['Ryotaro'])
            )
            match_key = 'match' if is_3way_match else 'mismatch'
            if match_key == 'mismatch' and not args.include_mismatch:
                continue

            # Extract base entities dynamically for each pair using intersection
            def compute_pair_score(gt_ann, pred_ann):
                sk_gt = sketches_rel[gt_ann]
                sk_pred = sketches_rel[pred_ann]

                gt_base = [eid for eid, ent in sk_gt.entities.items() if not getattr(ent, 'isConstruction', False)]
                pred_base = [eid for eid, ent in sk_pred.entities.items() if not getattr(ent, 'isConstruction', False)]
                base_eids = [eid for eid in gt_base if eid in pred_base]

                res = compute_null_space_distance(
                    gt_primary_sketch=sk_gt,
                    pred_sketch=sk_pred,
                    context=context,
                    skip_pred_primary_extraction=True,
                    base_eids=base_eids,
                    include_quadratic=True,
                )
                scores[match_key][pred_ann][gt_ann].append(res)

            # ---- Pred = Eko ----
            pred_ann = 'Eko'
            for gt_ann in ['Sagar', 'Ryotaro']:
                compute_pair_score(gt_ann, pred_ann)

            # ---- Pred = Sagar ----
            pred_ann = 'Sagar'
            for gt_ann in ['Eko', 'Ryotaro']:
                compute_pair_score(gt_ann, pred_ann)

            # ---- Pred = Ryotaro ----
            pred_ann = 'Ryotaro'
            for gt_ann in ['Eko', 'Sagar']:
                compute_pair_score(gt_ann, pred_ann)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

        finally:
            for doc in docs:
                try:
                    FreeCAD.closeDocument(doc.Name)
                except:
                    pass

    _print_results("Match Results (Primitives Identical)", scores['match'])
    if args.include_mismatch:
        _print_results("Mismatch Results (Primitives Differ)", scores['mismatch'])


if __name__ == "__main__":
    main()
