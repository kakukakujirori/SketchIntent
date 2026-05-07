import argparse
import csv
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
ANNOTATORS = ('Eko', 'Sagar', 'Ryotaro')

DIMENSIONAL_CONSTRAINTS = {
    'Distance', 'Length', 'Diameter', 'Radius', 'Angle', 'Offset',
    'Centerline_Dimension', 'Minor_Diameter', 'Major_Diameter', 'Rho',
    'Block', 'InternalAlignment'
}

def is_dimensional_constraint(constraint):
    return constraint.constraint_type.name in DIMENSIONAL_CONSTRAINTS

def count_dimensional_constraints(sketch):
    return sum(1 for c in sketch.constraints.values() if is_dimensional_constraint(c))

def get_relational_sketch(sketch):
    sk = copy.deepcopy(sketch)
    sk.constraints = {cid: c for cid, c in sk.constraints.items()
                      if not is_dimensional_constraint(c)}
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

def get_file_map(d):
    files = {}
    for f in os.listdir(d):
        if not f.endswith('.FCStd') or f.endswith('_mid.FCStd') or f.startswith('_'):
            continue
        files[f] = f
    return files

def get_files(d):
    return set(get_file_map(d))

def print_file_overlap(file_maps):
    common = set.intersection(*(set(m) for m in file_maps.values()))
    print(
        "Input files: "
        + " | ".join(f"{ann}={len(files)}" for ann, files in file_maps.items())
        + f" | common={len(common)}"
    )
    for ann, files in file_maps.items():
        missing = sorted(common ^ (common & set(files)))
        if missing:
            print(f"  Missing from {ann}: {', '.join(missing)}")

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
    for pred_ann in ANNOTATORS:
        for gt_ann in ANNOTATORS:
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


def _print_perfile_diff(rows):
    """Print per-file signed diff: linear[Pred=A, GT=X] - linear[Pred=B, GT=X] for key pairs."""
    comparisons = [
        ('Eko', 'Sagar', 'Ryotaro'),
        ('Eko', 'Ryotaro', 'Sagar'),
        ('Sagar', 'Ryotaro', 'Eko'),
    ]
    idx = {
        (r['pred'], r['gt'], r['filename']): r['linear']
        for r in rows
        if r['match'] == 'match' and r['linear'] is not None
    }
    for pred_a, pred_b, gt in comparisons:
        files = sorted({r['filename'] for r in rows if r['match'] == 'match' and r['linear'] is not None})
        diffs = []
        for fn in files:
            a = idx.get((pred_a, gt, fn))
            b = idx.get((pred_b, gt, fn))
            if a is not None and b is not None:
                diffs.append((fn, a - b))
        if not diffs:
            continue
        vals = [d for _, d in diffs]
        n_pos = sum(1 for d in vals if d > 0)
        print(f"\n--- Per-file diff: Pred={pred_a} minus Pred={pred_b} (GT={gt}) ---")
        print(f"{'File':<35} {'diff (A-B)':>12}  {'A':>8}  {'B':>8}")
        for fn, d in diffs:
            a = idx[(pred_a, gt, fn)]
            b = idx[(pred_b, gt, fn)]
            sign = '+' if d > 0 else '-' if d < 0 else ' '
            print(f"  {fn:<33} {sign}{abs(d):.6f}   {a:.6f}  {b:.6f}")
        print(f"  Mean diff = {np.mean(vals):+.6f}  ({n_pos}/{len(vals)} files: Pred={pred_a} > Pred={pred_b})")


def _format_score(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def _format_int_pair(left, right, width=3):
    if left is None or right is None:
        return f"{'NA':>{width}}/{'NA':<{width}}"
    return f"{left:>{width}d}/{right:<{width}d}"


def _print_perfile_results(rows, dimensional_counts):
    """Print all ordered Eko/Sagar/Ryotaro pair scores for each processed file."""
    if not rows:
        print("\n--- Per-file Eko/Sagar/Ryotaro Comparisons ---")
        print("No per-file results.")
        return

    rows_by_file = defaultdict(list)
    for row in rows:
        rows_by_file[row['filename']].append(row)

    ann_order = {ann: i for i, ann in enumerate(ANNOTATORS)}
    print("\n--- Per-file Eko/Sagar/Ryotaro Comparisons ---")
    for filename in sorted(rows_by_file):
        file_rows = sorted(
            rows_by_file[filename],
            key=lambda r: (ann_order[r['pred']], ann_order[r['gt']]),
        )
        match_key = file_rows[0]['match']
        counts = dimensional_counts.get(filename, {})
        count_text = " | ".join(f"{ann}={counts.get(ann, 'NA')}" for ann in ANNOTATORS)
        print(f"\n{filename} [{match_key}]")
        print(f"Dimensional constraints: {count_text}")
        print(
            f"{'Pred':<8} {'GT':<8} {'Dim(P/G)':>8} "
            f"{'Linear':>10} {'Quadratic':>10} {'2-Jet':>10} "
            f"{'d_lin(G/P)':>12} {'d_quad(G/P)':>13} {'Status':<54}"
        )
        for row in file_rows:
            status = "OK"
            if row.get('error'):
                status = f"ERR: {row['error']}"
                if len(status) > 54:
                    status = status[:51] + "..."
            print(
                f"{row['pred']:<8} {row['gt']:<8} "
                f"{_format_int_pair(row.get('pred_dimensional_constraints'), row.get('gt_dimensional_constraints'))} "
                f"{_format_score(row['linear']):>10} "
                f"{_format_score(row['quadratic']):>10} "
                f"{_format_score(row['two_jet']):>10} "
                f"{_format_int_pair(row.get('gt_row_dim'), row.get('pred_row_dim'), width=5)} "
                f"{_format_int_pair(row.get('gt_quad_dim'), row.get('pred_quad_dim'), width=6)} "
                f"{status:<54}"
            )


def main():
    parser = argparse.ArgumentParser(description="Human baseline for linear/quadratic intent recall.")
    parser.add_argument("--eko", default=EKO)
    parser.add_argument("--sagar", default=SAGAR)
    parser.add_argument("--ryotaro", default=RYOTARO)
    parser.add_argument(
        "--include-mismatch",
        action="store_true",
        help="Deprecated: primitive-mismatch cases are included by default.",
    )
    parser.add_argument(
        "--match-only",
        action="store_true",
        help="Only compute and report primitive-identical cases.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'exp1_per_file.csv'),
        help="Path for per-file metrics CSV output.",
    )
    parser.add_argument(
        "--dump-sigmas",
        action="store_true",
        help="Include σ_i (principal angle cosines) in per-file CSV for row-space alignment analysis.",
    )
    args = parser.parse_args()

    data_dirs = {
        'Eko': args.eko,
        'Sagar': args.sagar,
        'Ryotaro': args.ryotaro,
    }

    file_maps = {ann: get_file_map(d) for ann, d in data_dirs.items()}
    common = set.intersection(*(set(files) for files in file_maps.values()))
    common = sorted(list(common))
    print_file_overlap(file_maps)

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)

    # Store scores as scores[match_key][pred_ann][gt_ann]
    scores = {
        'match': {ann: defaultdict(list) for ann in ANNOTATORS},
        'mismatch': {ann: defaultdict(list) for ann in ANNOTATORS},
    }
    per_file_rows = []
    dimensional_counts = {}

    for filename in common:
        sketches_full = {}
        docs = []

        try:
            for annotator, d in data_dirs.items():
                path = os.path.join(d, file_maps[annotator][filename])
                doc, sketch_fc = load_FCStd_sketch(path)
                docs.append(doc)
                sk_data = converter.from_freecad(sketch_fc)
                sk_data, _ = complete_constraint_values(sk_data)
                sketches_full[annotator] = sk_data

            dimensional_counts[filename] = {
                ann: count_dimensional_constraints(sk) for ann, sk in sketches_full.items()
            }
            sketches_rel = {ann: get_relational_sketch(sk) for ann, sk in sketches_full.items()}

            # 3-way match check (all three must be identical including construction)
            is_3way_match = (
                check_primitives_match(sketches_rel['Eko'], sketches_rel['Sagar']) and
                check_primitives_match(sketches_rel['Sagar'], sketches_rel['Ryotaro'])
            )
            match_key = 'match' if is_3way_match else 'mismatch'
            if match_key == 'mismatch' and args.match_only:
                continue

            # Extract base entities dynamically for each pair using intersection
            def compute_pair_score(gt_ann, pred_ann):
                sk_gt = sketches_rel[gt_ann]
                sk_pred = sketches_rel[pred_ann]
                row = {
                    'filename': filename,
                    'match': match_key,
                    'pred': pred_ann,
                    'gt': gt_ann,
                    'gt_dimensional_constraints': dimensional_counts[filename][gt_ann],
                    'pred_dimensional_constraints': dimensional_counts[filename][pred_ann],
                    'error': '',
                }

                gt_base = [eid for eid, ent in sk_gt.entities.items() if not getattr(ent, 'isConstruction', False)]
                pred_base = [eid for eid, ent in sk_pred.entities.items() if not getattr(ent, 'isConstruction', False)]
                base_eids = [eid for eid in gt_base if eid in pred_base]

                try:
                    res = compute_null_space_distance(
                        gt_primary_sketch=sk_gt,
                        pred_sketch=sk_pred,
                        context=context,
                        skip_pred_primary_extraction=True,
                        base_eids=base_eids,
                        include_quadratic=True,
                        return_sigmas=args.dump_sigmas,
                    )
                except Exception as e:
                    print(f"Error processing {filename} Pred={pred_ann} GT={gt_ann}: {e}")
                    row.update({
                        'linear': None,
                        'quadratic': None,
                        'two_jet': None,
                        'gt_row_dim': None,
                        'pred_row_dim': None,
                        'gt_quad_dim': None,
                        'pred_quad_dim': None,
                        'sigmas': '',
                        'error': str(e),
                    })
                    per_file_rows.append(row)
                    return

                scores[match_key][pred_ann][gt_ann].append(res)
                row.update({
                    'linear': res['linear_intent_recall'],
                    'quadratic': res['quadratic_intent_recall'],
                    'two_jet': res['two_jet_intent_recall'],
                    'gt_row_dim': res['gt_row_dim'],
                    'pred_row_dim': res['pred_row_dim'],
                    'gt_quad_dim': res['gt_quadratic_dim'],
                    'pred_quad_dim': res['pred_quadratic_dim'],
                    'sigmas': ','.join(f'{s:.6f}' for s in res['linear_sigmas']) if res.get('linear_sigmas') else '',
                })
                per_file_rows.append(row)

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

    _print_perfile_results(per_file_rows, dimensional_counts)

    _print_results("Match Results (Primitives Identical)", scores['match'])
    _print_results("Mismatch Results (Primitives Differ)", scores['mismatch'])

    # --- Per-file CSV ---
    if per_file_rows and args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        fieldnames = ['filename', 'match', 'pred', 'gt', 'linear', 'quadratic', 'two_jet',
                      'gt_row_dim', 'pred_row_dim', 'gt_quad_dim', 'pred_quad_dim',
                      'gt_dimensional_constraints', 'pred_dimensional_constraints',
                      'sigmas', 'error']
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_file_rows)
        print(f"\nPer-file metrics saved to: {args.output_csv}")

    # --- Per-file signed diff for key pairs (GT fixed, Pred varies) ---
    _print_perfile_diff(per_file_rows)


if __name__ == "__main__":
    main()
