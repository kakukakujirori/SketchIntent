"""Generate pseudo-relational constraint dataset.

Applies a rel_only RLOO model to generate relational constraints for every
sketch in an input npy file. Predicted relational constraints serve as the
pseudo-GT preservation set R_G for DFeasReward in rloo_dfeas training.

By default, solver validation is enabled (--solver_validate): only samples
whose predicted relational constraints are solver-Converged with no
malformed/conflicting constraints are saved. Use --no_solver_validate to
skip this and rely on tokenizer-level checking only (faster but less clean).

Typical usage (10 % subset + solver validation, ~4-6 h on cuda:1):
    python scripts/generate_pseudo_rel_dataset.py \
        --checkpoint logs/train/runs/2026-05-17_17-01-17/checkpoints/epoch_1_step_52700.ckpt \
        --input_npy  data_processed/sg_filtered_unique_valid.npy \
        --output_npy data_processed/sg_pseudo_rel_167k.npy \
        --limit 167000 --batch_size 32

    # Smoke test (100 samples):
    python scripts/generate_pseudo_rel_dataset.py \
        --checkpoint logs/train/runs/2026-05-17_17-01-17/checkpoints/epoch_1_step_52700.ckpt \
        --input_npy  data_processed/sg_filtered_unique_valid.npy \
        --output_npy /tmp/sg_pseudo_rel_smoke.npy \
        --limit 100 --batch_size 32
"""
import argparse
import sys
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from src.data.components.constraint_dataset import ConstraintDataset
from src.data.components.primitive_noise import PrimitiveNoise
from src.data.constraint_datamodule import sketch_collate_fn
from src.models.components.solver_feedback import SolverFeedback
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_constraints,
    complete_constraint_values,
    remove_dimensional_constraints,
)
from src.test import generate_with_partial_sequence


def _offsets_from_counts(counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def build_output_npy(
    original_data: dict,
    valid_indices: list[int],
    new_sequences: list,
) -> np.ndarray:
    """Pack new sequences + preserved metadata into a dictionary flat array."""
    packed_seqs = flat_array.save_list_flat(new_sequences)

    out: dict = {'sequences': packed_seqs}
    for key, value in original_data.items():
        if key == 'sequences':
            continue
        if isinstance(value, np.ndarray):
            out[key] = value[valid_indices]
        elif hasattr(value, '__getitem__'):
            out[key] = np.array([value[i] for i in valid_indices])

    return flat_array.pack_dictionary_flat(out)


@torch.no_grad()
def generate_pseudo_rel(
    module: L.LightningModule,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    solver_validate: bool = True,
    limit: int | None = None,
) -> tuple[list[int], list, dict]:
    """Run model inference and collect valid pseudo-relational sketch sequences.

    Validation steps (always):
      1. Detokenization must succeed.
      2. At least one relational constraint must be predicted.

    Additional validation when solver_validate=True:
      3. FreeCAD solver must report Converged with no malformed or conflicting
         constraints. Redundant constraints are tolerated (they are harmless as
         extra rows in J_R and do not cause incorrect d_feas signals).
    """
    module.eval().to(device)
    tokenizer = SketchTokenizer(module.hparams.context)
    sf = SolverFeedback(module.hparams.context) if solver_validate else None

    valid_indices: list[int] = []
    new_sequences: list = []
    stats: dict = {
        'total': 0,
        'valid': 0,
        'no_rel': 0,
        'solver_reject': 0,
        'error': 0,
    }

    for batch in tqdm(loader, desc='Generating pseudo-rel constraints'):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        B, L = batch['p_val'].shape

        start_seqs = torch.full((B, L), ConstraintToken.Pad, dtype=torch.long, device=device)
        start_seqs[:, 0] = ConstraintToken.Start

        generated = generate_with_partial_sequence(
            module,
            {
                'p_val': batch['p_val'],
                'p_attr': batch['p_attr'],
                'p_pos': batch['p_pos'],
                'c_val': start_seqs,
            },
            device,
        )

        for i in range(B):
            stats['total'] += 1
            if limit is not None and stats['total'] > limit:
                return valid_indices, new_sequences, stats

            prim_tokens = batch['p_val'][i].cpu().numpy()
            cons_tokens = generated[i].cpu().numpy()
            original_sketch = batch['sketch'][i]  # datalib.Sketch

            try:
                # Step 1-2: detokenize + filter to relational only
                sg_sketch, _, _ = tokenizer.detokenize(
                    {'p_val': prim_tokens, 'c_val': cons_tokens},
                    allow_invalid_constraints=True,
                    reference_sketch=original_sketch,
                )
                sg_sketch = SolverFeedback.overwrite_primitives_with_original(
                    sg_sketch, original_sketch
                )
                sg_sketch, _ = complete_constraint_values(sg_sketch, skip_invalid_constraints=True)
                rel_sketch = remove_dimensional_constraints(sg_sketch)
                # Cleanse relational constraints: fixes Midpoint/Mirror ref ordering,
                # removes self-referential or geometrically inconsistent constraints.
                rel_sketch = cleanse_sketch_constraints(rel_sketch, delete_invalid=True)

                if len(rel_sketch.constraints) == 0:
                    stats['no_rel'] += 1
                    continue

                # Step 3 (optional): FreeCAD solver validation
                if sf is not None:
                    status = sf.get_solver_status(rel_sketch)
                    if status.Status != 'Converged':
                        stats['solver_reject'] += 1
                        continue
                    if status.MalformedConstraints or status.ConflictingConstraints:
                        stats['solver_reject'] += 1
                        continue
                    if status.DistortingConstraints:
                        stats['solver_reject'] += 1
                        continue
                    # Redundant constraints are tolerated

                new_seq = datalib.sketch_to_sequence(rel_sketch)
                valid_indices.append(batch['sketch_index'][i].item())
                new_sequences.append(new_seq)
                stats['valid'] += 1

            except Exception:
                stats['error'] += 1

    return valid_indices, new_sequences, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True,
                        help='Path to rel_only RLOO checkpoint (.ckpt)')
    parser.add_argument('--input_npy', required=True,
                        help='Input npy (sg_filtered_unique_valid.npy)')
    parser.add_argument('--output_npy', required=True,
                        help='Output npy path')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers for data loading')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after this many input samples (for subsets / smoke tests)')
    parser.add_argument('--no_solver_validate', action='store_true',
                        help='Skip FreeCAD solver validation (faster but allows conflicting pseudo-GT)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    solver_validate = not args.no_solver_validate

    L.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    # Load model from checkpoint (uses .hydra/config.yaml next to checkpoints/)
    config_file = Path(args.checkpoint).parent.parent / '.hydra' / 'config.yaml'
    print(f'Loading config from {config_file}')
    cfg = OmegaConf.load(config_file)
    module = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    module.load_state_dict(ckpt.get('state_dict', ckpt))
    context = module.hparams.context
    print(f'Model loaded: max_token_length={context.max_token_length}')
    print(f'Solver validation: {"ON" if solver_validate else "OFF (--no_solver_validate)"}')

    # Dataset: relational_only=True matches rel_only model's training setup
    dataset = ConstraintDataset(
        args.input_npy,
        context=context,
        primitive_noise=PrimitiveNoise(std=0.0, max_diff=0.0),
        relational_only=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=sketch_collate_fn,
    )
    original_data = flat_array.load_dictionary_flat(args.input_npy)
    n_input = args.limit if args.limit else len(dataset)
    print(f'Dataset: {len(dataset)} sketches total, processing up to {n_input}')

    # Generate
    valid_indices, new_sequences, stats = generate_pseudo_rel(
        module, loader, device,
        solver_validate=solver_validate,
        limit=args.limit,
    )

    total = max(stats['total'], 1)
    print('\n=== Summary ===')
    print(f'Total processed:         {stats["total"]:>8}')
    print(f'Valid (saved):           {stats["valid"]:>8}  ({100*stats["valid"]/total:.1f}%)')
    print(f'Excluded (no rel):       {stats["no_rel"]:>8}  ({100*stats["no_rel"]/total:.1f}%)')
    if solver_validate:
        print(f'Excluded (solver):       {stats["solver_reject"]:>8}  ({100*stats["solver_reject"]/total:.1f}%)')
    print(f'Excluded (error):        {stats["error"]:>8}  ({100*stats["error"]/total:.1f}%)')

    if not valid_indices:
        print('ERROR: No valid samples generated. Exiting without saving.', file=sys.stderr)
        sys.exit(1)

    print(f'\nPacking {len(valid_indices)} sequences...')
    output_npy = build_output_npy(original_data, valid_indices, new_sequences)

    print(f'Saving to {args.output_npy} ...')
    np.save(args.output_npy, output_npy, allow_pickle=False)
    print('Done.')


if __name__ == '__main__':
    main()
