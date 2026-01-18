#!/usr/bin/env python
"""Save fully constrained GT and predictions that pass all validation criteria.

Iterates through fc_dataset, runs evaluation, and saves history files + GT SVG
only when both GT (refined) and WithSolver results pass all criteria:
- DoF == 2
- No malformed, conflicting, or redundant constraints
- Geometry validation passes (deviation < threshold)
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import sketchgraphs.data as datalib
from sketchgraphs.data import Arc, Circle, Line, Point, flat_array
from sketchgraphs.data._constraint import Constraint, LocalReferenceParameter

from img2cad_diffusion import sketch_utils
from img2cad_diffusion.constraint_data import ConstraintDataModule
from img2cad_diffusion.constraint_models_ar import ConstraintARModule
from img2cad_diffusion.solver_feedback import (
    get_solver_status,
    validate_constraint_geometry_impact,
    HAS_FREECAD,
)
from img2cad_diffusion.convert_sketch_to_freecad import convert_sketch_to_freecad
from img2cad_diffusion.detokenization import reconstruct_constraints, build_gather_idx_mapping
from img2cad_diffusion.tokenization import CONSTRAINT_TOKEN_LEN, tokenize_primitives


if not HAS_FREECAD:
    raise ImportError("FreeCAD is required for this script")

import FreeCAD


def build_sketch_with_original_entities(
    original_sketch: datalib.Sketch,
    cons_tokens: np.ndarray,
    num_bins: int,
    token_len_per_constraint: int,
) -> datalib.Sketch:
    """Build sketch using original entity coords + reconstructed constraints.

    Strategy: First tokenize the original sketch to get primitive tokens (preserving
    entity structure), then use reconstruct_sketch with those tokens and constraint
    tokens to build a sketch with correct constraint refs (entity IDs "1", "2", "3").
    Finally, replace entity coordinates with original (non-quantized) values.
    """
    from img2cad_diffusion.detokenization import reconstruct_sketch

    # Step 1: Extract non-quantized coordinates
    original_entities = list(original_sketch.entities.values())

    # Step 2: Tokenize original sketch to get primitive tokens (same structure as batch)
    prim_token_dict, _ = tokenize_primitives(original_sketch, num_bins, max_length=None)
    prim_tokens = prim_token_dict['val']

    # Step 3: Use standard reconstruct_sketch to build sketch with correct constraint refs
    # This creates entities with IDs "1", "2", "3" and constraints matching the tokens
    sketch = reconstruct_sketch(prim_tokens, cons_tokens, num_bins, token_len_per_constraint)

    # Step 4: Replace entity coordinates with original (non-quantized) values
    # Entity order should match since we used the same tokenization
    reconstructed_entities = list(sketch.entities.values())
    for recon_ent, orig_ent in zip(reconstructed_entities, original_entities):
        if recon_ent is None or orig_ent is None:
            continue
        if type(recon_ent) != type(orig_ent):
            continue  # Type mismatch - skip

        # Copy coordinates from original entity
        if isinstance(orig_ent, Line):
            recon_ent.dirX = orig_ent.dirX
            recon_ent.dirY = orig_ent.dirY
            recon_ent.pntX = orig_ent.pntX
            recon_ent.pntY = orig_ent.pntY
            recon_ent.startParam = orig_ent.startParam
            recon_ent.endParam = orig_ent.endParam
        elif isinstance(orig_ent, Circle):
            recon_ent.xCenter = orig_ent.xCenter
            recon_ent.yCenter = orig_ent.yCenter
            recon_ent.radius = orig_ent.radius
        elif isinstance(orig_ent, Arc):
            recon_ent.xCenter = orig_ent.xCenter
            recon_ent.yCenter = orig_ent.yCenter
            recon_ent.xDir = orig_ent.xDir
            recon_ent.yDir = orig_ent.yDir
            recon_ent.radius = orig_ent.radius
            recon_ent.startParam = orig_ent.startParam
            recon_ent.endParam = orig_ent.endParam
            recon_ent.clockwise = orig_ent.clockwise
        elif isinstance(orig_ent, Point):
            recon_ent.x = orig_ent.x
            recon_ent.y = orig_ent.y
        else:
            raise ValueError(f"Unsupported entity type: {type(orig_ent)}")

    return sketch


def sketch_to_freecad(
    original_sketch: datalib.Sketch,
    cons_tokens: np.ndarray | torch.Tensor,
    num_bins: int,
    token_len_per_constraint: int,
):
    """Convert original sequence + constraint tokens to FreeCAD sketch.

    Uses original (non-quantized) primitive coordinates for accurate dimensions.
    """
    if isinstance(cons_tokens, torch.Tensor):
        cons_tokens = cons_tokens.cpu().numpy()

    sketch = build_sketch_with_original_entities(
        original_sketch, cons_tokens, num_bins, token_len_per_constraint
    )
    doc, fc_sketch, history_commands, entity_to_geoid, mapping = convert_sketch_to_freecad(sketch)
    return doc, fc_sketch, history_commands, mapping


def evaluate_and_save(
    model: ConstraintARModule,
    datamodule: ConstraintDataModule,
    idx: int,
    config,
    output_dir: Path,
    original_sketch: datalib.Sketch,
    geo_threshold_mm: float = 50.0,
    seed: int = 42,
) -> dict:
    """Evaluate a single sample and save if all criteria pass.

    Returns dict with evaluation results.
    """
    from img2cad_diffusion.convert_sketch_to_freecad import FreeCADConversionError
    from img2cad_diffusion.test_solver_feedback_ar import (
        cleanup_constraints,
        generate_sample_with_solver_feedback,
    )

    device = model.device
    num_bins = config.data.num_position_bins
    max_len = config.data.max_token_length

    # Get data item
    fc_dataset = datamodule._fc_test_dataset
    data_item = fc_dataset[idx]
    batch = {}
    for k, v in data_item.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = [v]

    result = {
        'index': idx,
        'saved': False,
        'gt_refined': None,
        'solver': None,
    }

    # Evaluate GT
    gt_tokens = batch['c_val'][0]
    try:
        doc, fc_sketch, gt_history, mapping = sketch_to_freecad(
            original_sketch, gt_tokens, num_bins, CONSTRAINT_TOKEN_LEN)
        gt_status = get_solver_status(fc_sketch, mapping)
        FreeCAD.closeDocument(doc.Name)
    except FreeCADConversionError:
        result['error'] = 'GT conversion failed'
        return result

    # Cleanup GT if needed
    problem_count = (
        len(gt_status.MappedMalformedIndices) +
        len(gt_status.MappedConflictingIndices) +
        len(gt_status.MappedRedundantIndices)
    )
    if problem_count > 0:
        gt_refined_tokens, gt_refined_status, gt_refined_history = cleanup_constraints(
            batch['val'][0], gt_tokens, num_bins, max_len)
        if gt_refined_status is None:
            result['error'] = 'GT cleanup failed'
            return result
    else:
        gt_refined_tokens = gt_tokens
        gt_refined_status = gt_status
        gt_refined_history = gt_history

    # Check GT refined criteria
    gt_ok = (
        gt_refined_status.DoF == 2 and
        len(gt_refined_status.MappedMalformedIndices) == 0 and
        len(gt_refined_status.MappedConflictingIndices) == 0 and
        len(gt_refined_status.MappedRedundantIndices) == 0
    )
    result['gt_refined'] = {
        'DoF': gt_refined_status.DoF,
        'malformed': len(gt_refined_status.MappedMalformedIndices),
        'conflicting': len(gt_refined_status.MappedConflictingIndices),
        'redundant': len(gt_refined_status.MappedRedundantIndices),
        'status_ok': gt_ok,
    }

    if not gt_ok:
        return result

    # Run GT geometry validation
    try:
        sketch = build_sketch_with_original_entities(
            original_sketch, gt_refined_tokens.cpu().numpy(),
            num_bins, CONSTRAINT_TOKEN_LEN)
        geo_results, fc_sketch, _, _ = validate_constraint_geometry_impact(
            sketch, threshold_mm=geo_threshold_mm, verbose=False)
        FreeCAD.closeDocument(fc_sketch.Document.Name)

        max_dev = max((r.max_deviation for r in geo_results), default=0)
        gt_geo_ok = max_dev < geo_threshold_mm
        result['gt_refined']['max_dev'] = max_dev
        result['gt_refined']['geo_ok'] = gt_geo_ok
    except Exception as e:
        result['gt_refined']['geo_ok'] = False
        result['gt_refined']['geo_error'] = str(e)[:50]
        return result

    if not gt_geo_ok:
        return result

    # Generate with solver feedback
    torch.manual_seed(seed)
    with torch.no_grad():
        solver_sample, _, solver_history = generate_sample_with_solver_feedback(
            model, batch, num_bins, max_len, max_iterations=256)

    # Evaluate solver result
    try:
        doc, fc_sketch, solver_history_final, mapping = sketch_to_freecad(
            original_sketch, solver_sample[0], num_bins, CONSTRAINT_TOKEN_LEN)
        solver_status = get_solver_status(fc_sketch, mapping)
        FreeCAD.closeDocument(doc.Name)
    except FreeCADConversionError:
        result['error'] = 'Solver conversion failed'
        return result

    # Check solver criteria
    solver_ok = (
        solver_status.DoF == 2 and
        len(solver_status.MappedMalformedIndices) == 0 and
        len(solver_status.MappedConflictingIndices) == 0 and
        len(solver_status.MappedRedundantIndices) == 0
    )
    result['solver'] = {
        'DoF': solver_status.DoF,
        'malformed': len(solver_status.MappedMalformedIndices),
        'conflicting': len(solver_status.MappedConflictingIndices),
        'redundant': len(solver_status.MappedRedundantIndices),
        'status_ok': solver_ok,
    }

    if not solver_ok:
        return result

    # Run solver geometry validation
    try:
        sketch = build_sketch_with_original_entities(
            original_sketch, solver_sample[0].cpu().numpy(),
            num_bins, CONSTRAINT_TOKEN_LEN)
        geo_results, fc_sketch, _, _ = validate_constraint_geometry_impact(
            sketch, threshold_mm=geo_threshold_mm, verbose=False)
        FreeCAD.closeDocument(fc_sketch.Document.Name)

        max_dev = max((r.max_deviation for r in geo_results), default=0)
        solver_geo_ok = max_dev < geo_threshold_mm
        result['solver']['max_dev'] = max_dev
        result['solver']['geo_ok'] = solver_geo_ok
    except Exception as e:
        result['solver']['geo_ok'] = False
        result['solver']['geo_error'] = str(e)[:50]
        return result

    if not solver_geo_ok:
        return result

    # All criteria passed! Save files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save GT refined history
    gt_path = output_dir / f"history_gt_ar_refined_{idx}.py"
    with open(gt_path, "w") as f:
        f.write("\n".join(gt_refined_history))

    # Save solver history
    solver_path = output_dir / f"history_pred_with_solver_{idx}.py"
    with open(solver_path, "w") as f:
        f.write("\n".join(solver_history_final))

    # Save GT SVG
    try:
        # Load original sequence to render
        sequences = np.load(config.data.sequence_path, allow_pickle=True)
        import json
        with open(config.data.fc_clusters_path, 'r') as f:
            fc_clusters = json.load(f)
        test_indices = fc_clusters['selected_for_test']
        seq_idx = test_indices[idx]
        seq = sequences[seq_idx]
        original_sketch = datalib.sketch_from_sequence(seq)

        fig = datalib.render_sketch(original_sketch)
        svg_path = output_dir / f"gt_sketch_{idx}.svg"
        fig.savefig(svg_path)
        plt.close(fig)
    except Exception as e:
        # Fallback: render reconstructed sketch if original fails
        try:
            fig = datalib.render_sketch(sketch)
            svg_path = output_dir / f"gt_sketch_{idx}.svg"
            fig.savefig(svg_path)
            plt.close(fig)
        except:
            pass  # Skip SVG if all fails

    result['saved'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description="Save FC GT and predictions that pass all validation criteria")
    parser.add_argument('checkpoint', type=str, help="Path to AR model checkpoint")
    parser.add_argument('--data_num', type=int, default=100, help="Number of FC samples to process")
    parser.add_argument('--geo_threshold', type=float, default=50.0, help="Geometry validation threshold in mm")
    parser.add_argument('--output_dir', type=str, default='scripts/prediction_fc_search', help="Output directory")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load model
    model = ConstraintARModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device, weights_only=False)
    model.to(args.device)
    model.eval()

    config = model.hparams
    config.data.sequence_path = "data_processed/sg_filtered_unique.npy"
    config.data.fc_clusters_path = "scripts/sketchgraphs_fc_search/sketchgraphs_fc_clustered.json"
    config.data.primitive_noise.enabled = False
    config.data.batch_size = 1
    config.data.num_data_workers = 0

    datamodule = ConstraintDataModule(config.data)
    datamodule.setup("test")

    # Load original sequences for non-quantized dimension calculation
    seq_data = flat_array.load_dictionary_flat(config.data.sequence_path)
    sequences = seq_data['sequences']

    # Use test dataloader - may be a single loader or a list [normal, fc]
    test_loaders = datamodule.test_dataloader()
    if isinstance(test_loaders, list):
        # Use the second (fc test) dataloader
        test_loader = test_loaders[1]
        print(f"Test dataset size: {len(test_loader.dataset)} (using normal test)")
    else:
        test_loader = test_loaders
        print(f"Test dataset size: {len(test_loader.dataset)}")

    # Get single sample from test dataloader
    fc_dataset = test_loader.dataset
    assert fc_dataset is not None, "FC test dataset not available"
    n_samples = min(args.data_num, len(fc_dataset))

    print(f"Processing {n_samples} FC samples...")
    print(f"Geometry threshold: {args.geo_threshold}mm")
    print(f"Output directory: {output_dir}")
    print()

    saved_count = 0
    for idx in range(n_samples):
        # Get original sequence via data_index
        data_item = fc_dataset[idx]
        data_index = data_item['data_index'].item()
        original_seq = sequences[data_index]
        original_sketch = datalib.sketch_from_sequence(original_seq)
        sketch_utils.normalize_sketch(original_sketch)

        result = evaluate_and_save(
            model, datamodule, idx, config, output_dir, original_sketch,
            geo_threshold_mm=args.geo_threshold, seed=args.seed)

        status = "SAVED" if result['saved'] else "SKIP"

        gt_info = ""
        if result['gt_refined']:
            gt = result['gt_refined']
            gt_info = f"GT:DoF={gt['DoF']}"
            if 'max_dev' in gt:
                gt_info += f",dev={gt['max_dev']:.0f}mm"

        solver_info = ""
        if result['solver']:
            s = result['solver']
            solver_info = f" | Solver:DoF={s['DoF']}"
            if 'max_dev' in s:
                solver_info += f",dev={s['max_dev']:.0f}mm"

        print(f"[{idx:3d}] {status} - {gt_info}{solver_info}")

        if result['saved']:
            saved_count += 1

    print()
    print(f"Done! Saved {saved_count}/{n_samples} samples to {output_dir}")


if __name__ == '__main__':
    main()
