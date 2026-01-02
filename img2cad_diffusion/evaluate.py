"""Evaluation script for Constraint MDD model with FreeCAD Solver feedback.

Run this script in the 'freecad' environment.
"""

import argparse
import os
import time
from typing import List, Dict

import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader

from img2cad_diffusion import constraint_models, constraint_data, primitive_data, convert_to_freecad
from img2cad_diffusion.constraint_data import Token, detokenize_constraints
from img2cad_diffusion.convert_sketch_to_freecad import draw_svg

import FreeCAD
import sketchgraphs.data as datalib


def simple_collate(batch):
    return datalib.collate_sketch_graphs(batch)


def solve_and_check(seq: list[datalib.NodeOp | datalib.EdgeOp], output_svg_path: str | None = None):
    # Setup FreeCAD Document
    output_basename = os.path.splitext(os.path.basename(output_svg_path))[0] if output_svg_path else "out"
    doc_name = f"MyDoc_{output_basename}"
    doc = FreeCAD.newDocument(doc_name)

    # script dump
    script_content = [
        "import FreeCAD",
        "import Part",
        "import Sketcher",
        "from FreeCAD import Vector",
        "doc = FreeCAD.newDocument('MyDoc_" + output_basename + "')",
        "body = doc.addObject('PartDesign::Body', 'Body')",
        "sketch = body.newObject('Sketcher::SketchObject', 'Sketch')",
        "",
    ]

    try:
        body = doc.addObject('PartDesign::Body', 'Body')
        sketch = body.newObject('Sketcher::SketchObject', 'Sketch')

        xy_plane = body.Origin.OriginFeatures[3]
        sketch.AttachmentSupport = (xy_plane, '')
        sketch.MapMode = 'FlatFace'

        seq = [convert_to_freecad.meter_to_millimeter(op) for op in seq]

        # Resolve subnodes, etc.
        node_idx_map = convert_to_freecad.resolve_subnodes(seq)
        seq = convert_to_freecad.resolve_redundant_coincident(seq, node_idx_map)
        seq = convert_to_freecad.resolve_tangent(seq, node_idx_map)

        # Build
        for op in seq:
            if isinstance(op, datalib.NodeOp):
                _, cmd = convert_to_freecad.convert_primitive(op, sketch)
            elif isinstance(op, datalib.EdgeOp):
                cmd = convert_to_freecad.convert_constraint(op, sketch, node_idx_map)
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

            if cmd:
                script_content.append(cmd)

        doc.recompute()
        sketch.solve()

        # dump
        if output_svg_path:
            output_python_path = output_svg_path.replace(".svg", ".py")
            with open(output_python_path, 'w') as f:
                script_content.append("doc.recompute()")
                f.write("\n".join(script_content))
            draw_svg(doc, sketch, output_svg_path, debug=False)

        is_fully = False
        is_over = False
        is_under = False
        is_redundant = False

        # 1. Redundant
        if hasattr(sketch, 'RedundantConstraints') and len(sketch.RedundantConstraints) > 0:
            is_redundant = True

        # 2. Conflicting (Over)
        if hasattr(sketch, 'ConflictingConstraints') and len(sketch.ConflictingConstraints) > 0:
            is_over = True

        # 3. Solver Status String Check
        if hasattr(sketch, 'getSolverStatus'):
            status_dict = sketch.getSolverStatus()
            status_str = status_dict.get('Status', 'Unknown')
            dof = status_dict.get('DoF', -1)

            if status_str == 'OverConstrained':
                is_over = True
            elif status_str == 'Conflicting':
                is_over = True
            elif status_str == 'Redundant':
                is_redundant = True
            elif status_str == 'Converged':
                if dof == 0:
                    is_fully = True
                else:
                    is_under = True
            elif status_str == 'Failed':
                pass
            if len(status_dict.get('ConflictingConstraints', [])) > 0:
                is_over = True
            if len(status_dict.get('RedundantConstraints', [])) > 0:
                is_redundant = True

        else:
            if hasattr(sketch, 'RedundantConstraints') and len(sketch.RedundantConstraints) > 0:
                is_redundant = True
            if hasattr(sketch, 'ConflictingConstraints') and len(sketch.ConflictingConstraints) > 0:
                is_over = True

        # If solver says converged but we don't know DoF, check redundant again
        # If Fully Constrained, usually DoF=0. FreeCAD 'FullyConstrained' status ensures this.

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            FreeCAD.closeDocument(doc_name)
        except:
            pass
        raise e

    FreeCAD.closeDocument(doc_name)

    return {
        'fully_constrained': is_fully,
        'over_constrained': is_over,
        'under_constrained': (not is_fully and not is_over and not is_redundant),
        'redundant': is_redundant
    }


@torch.no_grad()
def evaluate(checkpoint_path: str, device: str = 'cuda', dump_dir: str | None = None, use_gt: bool = False):
    print(f"Loading model from {checkpoint_path}...")
    model_module = constraint_models.ConstraintMDDModule.load_from_checkpoint(checkpoint_path, weights_only=False)
    model_module.to(device)
    model_module.eval()

    config = model_module.hparams
    config.data.batch_size = 1  # overload
    config.data.num_data_workers = 0  # overload
    assert config.data.validation_fraction is not None and config.data.validation_fraction > 0

    print(f"Setting up dataset...")
    datamodule = constraint_data.ConstraintDataModule(config.data)
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    print(f"Starting evaluation loop...")

    results = {
        'fully_constrained': [],
        'under_constrained': [],
        'over_constrained': [],
        'redundant': [],
        'valid_geometry': [],
        'total': 0,
        'conversion_errors': []
    }

    # Limit number of samples for quick baseline check
    SAMPLE_START_IDX = 0
    SAMPLE_END_IDX = SAMPLE_START_IDX + 100

    for i, batch in enumerate(tqdm.tqdm(val_loader, total=min(len(val_loader), (SAMPLE_END_IDX - SAMPLE_START_IDX)))):
        if i < SAMPLE_START_IDX:
            continue
        if SAMPLE_END_IDX <= i:
            break

        # Batch is TokenInput (dict of tensors)
        batch_input = {k: v.to(device) for k, v in batch.items()}

        if use_gt:
            # Use Ground Truth constraints instead of generation
            generated_seq = batch['c_val'][0].cpu().numpy()
        else:
            generated_c_val, _ = model_module.ProbPath.generate_sample(
                batch_input,
                num_steps=50, # Reduced steps for speed in baseline check
                device=device
            )

            # generated_c_val: [B, L] -> [1, L]
            generated_seq = generated_c_val[0].cpu().numpy()

        # Reconstruct Sketch
        # 1. Recover Primitives from 'val' (primitive tokens)
        prim_seq_tokens = batch['val'][0].numpy()
        sketch = primitive_data.sketch_from_tokens(prim_seq_tokens, config.data.num_position_bins)
        i2c_utils = primitive_data.i2c_utils
        i2c_utils.normalize_sketch(sketch) # Was likely normalized in dataset, but skecth_from_tokens might need check

        # 2. Reconstruct Sketch with Constraints (Refactored logic)
        # This function encapsulates:
        # - Re-tokenization for gather_idxs
        # - Constraint detokenization
        # - Subnode injection and linking
        # - Reference index remapping
        full_sequence = constraint_data.reconstruct_sketch_with_constraints(
            sketch,
            generated_seq,
            config.data.num_position_bins
        )

        # 5. Solver Check
        try:
            output_svg_path = os.path.join(dump_dir, f"sample_{i}.svg") if dump_dir else None
            status = solve_and_check(full_sequence, output_svg_path)

            results['total'] += 1
            if status['fully_constrained']:
                results['fully_constrained'].append(i)
            elif status['over_constrained']:
                results['over_constrained'].append(i)
            elif status['under_constrained']:
                results['under_constrained'].append(i)

            if status['redundant']:
                    results['redundant'].append(i)

        except Exception as e:
            print(f"Error in solver check: {e}")
            results['conversion_errors'].append(i)

    print("\nEvaluation Results:")
    print(f"Total Samples: {results['total']}")
    print(f"Valid Geometry Count: {results['valid_geometry']} (Approx)")
    if results['total'] > 0:
        print(f"Fully Constrained: {results['fully_constrained']} ({len(results['fully_constrained'])/results['total']*100:.2f}%)")
        print(f"Over Constrained: {results['over_constrained']} ({len(results['over_constrained'])/results['total']*100:.2f}%)")
        print(f"Under Constrained: {results['under_constrained']} ({len(results['under_constrained'])/results['total']*100:.2f}%)")
        print(f"Redundant Constraints Detected: {results['redundant']} ({len(results['redundant'])/results['total']*100:.2f}%)")
        print(f"Conversion Errors: {results['conversion_errors']} ({len(results['conversion_errors'])/results['total']*100:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run evaluation on (e.g. cuda, cuda:0, cpu)')
    parser.add_argument('--dump_dir', type=str, default=None, help='Directory to dump FreeCAD python scripts')
    parser.add_argument('--use_gt', action='store_true', help='Use ground truth constraints for debugging')
    args = parser.parse_args()

    if os.path.isdir(args.dump_dir):
        print(f"{args.dump_dir} exists. Please remove it first.")
        exit(1)
    os.mkdir(args.dump_dir)

    evaluate(args.checkpoint, args.device, args.dump_dir, args.use_gt)
