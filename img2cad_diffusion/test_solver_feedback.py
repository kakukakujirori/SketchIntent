"""
Detailed inference script to test 'tau_leaping+solver_feedback' transparency.
Shows step-by-step intermediate states, removed constraints, and before/after comparison.
"""
import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch

from img2cad_diffusion.constraint_data import ConstraintDataModule
from img2cad_diffusion.constraint_models import ConstraintTokenInput, ConstraintMDDModule
from img2cad_diffusion.solver_feedback import get_solver_status, tokens_to_freecad_sketch, HAS_FREECAD
from img2cad_diffusion.tokenization import ConstraintToken, CONSTRAINT_TOKEN_LEN

if HAS_FREECAD:
    import FreeCAD

def tokens_to_readable(tokens: torch.Tensor, vocab_size: int = 64) -> list[str]:
    """Convert token tensor to human-readable list."""
    result = []
    for t in tokens.cpu().numpy():
        if t == int(ConstraintToken.Start):
            result.append("[START]")
        elif t == int(ConstraintToken.Stop):
            result.append("[STOP]")
        elif t == int(ConstraintToken.Pad):
            result.append("[PAD]")
        elif t == int(ConstraintToken.Mask):
            result.append("[MASK]")
        elif t == int(ConstraintToken.InnerPad):
            result.append("[IPAD]")
        elif t < len(ConstraintToken):
            result.append(ConstraintToken(t).name)
        else:
            # Reference token
            result.append(f"R{t - len(ConstraintToken)}")
    return result


def print_constraints_readable(tokens: torch.Tensor, prefix: str = ""):
    """Print constraint tokens in block format."""
    readable = tokens_to_readable(tokens)

    # Structure: [START, C0(4 tokens), C1(4 tokens), ...]
    print(f"{prefix}Prefix: {readable[:1]}")  # START

    idx = 1
    constraint_idx = 0
    while idx < len(readable):
        block = readable[idx:idx+4]
        if not block or block[0] in ["[PAD]", "[STOP]"]:
            # Print remaining
            remaining = readable[idx:]
            if len(remaining) > 10:
                print(f"{prefix}  ... {remaining[:5]} ... ({len(remaining)} tokens)")
            else:
                print(f"{prefix}  {remaining}")
            break
        print(f"{prefix}  c{constraint_idx}: {block}")
        constraint_idx += 1
        idx += 4


def evaluate(
    model_module: ConstraintMDDModule,
    batch: ConstraintTokenInput,
    seed: int = 42,
):
    # Show ground truth
    gt_tokens = batch['c_val'][0]
    print("\n[GROUND TRUTH CONSTRAINTS]")
    print_constraints_readable(gt_tokens, prefix="  ")

    # Evaluate baseline with FreeCAD
    doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
        batch['val'][0],
        gt_tokens,
        config.data.num_position_bins,
        token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
    )
    gt_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    print(f"\n  Status: {gt_status.Status}")
    print(f"  DoF: {gt_status.DoF}")
    print(f"  Malformed: {gt_status.MalformedConstraints}")
    print(f"  Conflicting: {gt_status.ConflictingConstraints}")
    print(f"  Redundant: {gt_status.RedundantConstraints}")
    print(f"  Mapped Conflicts: {gt_status.MappedConflictingIndices}")
    print(f"  Mapped Redundant: {gt_status.MappedRedundantIndices}")

    # =========================================================================
    # RUN 1: WITHOUT SOLVER FEEDBACK (baseline)
    # =========================================================================
    print(f"\n{'='*80}")
    print("RUN 1: WITHOUT SOLVER FEEDBACK (baseline)")
    print(f"{'='*80}")

    model_module.ProbPath.method = 'tau_leaping'  # No solver feedback

    torch.manual_seed(seed)
    with torch.no_grad():
        baseline_sample, baseline_intermediates = model_module.ProbPath.generate_sample(
            batch.copy(),
            num_steps=64,
            start_tokens=[int(ConstraintToken.Start)],
            constraint_blockwise_masking=True
        )

    print("\n[BASELINE FINAL RESULT]")
    print_constraints_readable(baseline_sample[0], prefix="  ")

    # Evaluate baseline with FreeCAD
    doc, fc_sketch, history_commands,mapping = tokens_to_freecad_sketch(
        batch['val'][0],
        baseline_sample[0],
        config.data.num_position_bins,
        token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
    )
    baseline_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    with open("history_without_solver.py", "w") as f:
        f.write("\n".join(history_commands))

    print(f"\n  Status: {baseline_status.Status}")
    print(f"  DoF: {baseline_status.DoF}")
    print(f"  Malformed: {baseline_status.MalformedConstraints}")
    print(f"  Conflicting: {baseline_status.ConflictingConstraints}")
    print(f"  Redundant: {baseline_status.RedundantConstraints}")
    print(f"  Mapped Conflicts: {baseline_status.MappedConflictingIndices}")
    print(f"  Mapped Redundant: {baseline_status.MappedRedundantIndices}")

    # =========================================================================
    # RUN 2: WITH SOLVER FEEDBACK
    # =========================================================================
    print(f"\n{'='*80}")
    print("RUN 2: WITH SOLVER FEEDBACK")
    print(f"{'='*80}")

    model_module.ProbPath.method = 'tau_leaping+solver_feedback'
    model_module.ProbPath.noise_removal = False

    # === Wrap _apply_solver_feedback for invalid constraint masking ===
    original_apply = model_module.ProbPath._apply_solver_feedback
    feedback_logs = []

    def logged_apply_solver_feedback(
        batch,
        p_x0,
        num_bins,
        max_solver_iters=5,
        scoot_valid_constraints=False,
        strip_first_stop_token=False,
    ):
        from img2cad_diffusion.convert_sketch_to_freecad import FreeCADConversionError
        from img2cad_diffusion.detokenization import reconstruct_constraints, reconstruct_sketch
        from img2cad_diffusion.solver_feedback import SolverStatus, convert_sketch_to_freecad

        x = batch['c_val'].clone()
        B, L = x.shape

        for b in range(B):
            prim_tokens = batch['val'][b]
            sample_x = x[b]
            sample_x_init = sample_x.clone()

            # STEP 1: Mask syntax-error constraints
            valid_constraints, invalid_constraints = reconstruct_constraints(
                sample_x.cpu().numpy(),
                token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
            )
            for constraint_type, refs, (start, end) in invalid_constraints:
                sample_x[start:end] = model_module.ProbPath.mask_index

            sample_x_after_syntax_filtering = sample_x.clone()

            # STEP2: MASK solver-alerting constraints
            sketch = reconstruct_sketch(
                prim_tokens.cpu().numpy(),
                sample_x.cpu().numpy(),
                num_bins,
                token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
            )

            status = SolverStatus()
            removed_constraint_ids: list[int] = []

            for _ in range(max_solver_iters):
                try:
                    doc, fc_sketch, _, fc_mapping = convert_sketch_to_freecad(sketch, doc_name="SolverEval")
                except FreeCADConversionError as e:
                    # If a constraint caused the error, remove it and retry
                    if e.phase == "constraint":
                        cid = e.constraint_id
                        if cid in sketch.constraints:
                            del sketch.constraints[cid]
                            removed_constraint_ids.append(int(cid[1:]))
                            continue  # Retry conversion
                    # If primitive error or unrecoverable, skip this sample
                    print(f"[_apply_solver_feedback] Error: {e}")
                    break
                except Exception as e:
                    print(f"[_apply_solver_feedback] Unexpected error: {e}")
                    break

                status = get_solver_status(fc_sketch, fc_mapping)
                FreeCAD.closeDocument(doc.Name)

                # Get problem constraints
                problem_indices = status.MappedConflictingIndices + status.MappedRedundantIndices + status.MappedMalformedIndices
                if not problem_indices:
                    break

                # Find least confident constraint to mask
                min_conf = float('inf')
                target_idx = -1

                for constraint_idx in problem_indices:
                    constraint_type, constraint_refs, (start_pos, end_pos) = valid_constraints[constraint_idx]

                    # Calculate confidence as product of p(x0|xt)
                    chunk_probs = p_x0[b, start_pos:end_pos, :]
                    chunk_tokens = sample_x[start_pos:end_pos]
                    token_probs = chunk_probs.gather(dim=-1, index=chunk_tokens.unsqueeze(1)).squeeze(1)
                    score = token_probs.prod().item()

                    if score < min_conf:
                        min_conf = score
                        target_idx = constraint_idx

                # Remove constraint from Sketch
                del sketch.constraints[f"c{target_idx}"]
                removed_constraint_ids.append(target_idx)

            for cid in removed_constraint_ids:
                _, _, (start_pos, end_pos) = valid_constraints[cid]
                sample_x[start_pos:end_pos] = model_module.ProbPath.mask_index

            sample_x_after_solver_filtering = sample_x.clone()

            # Log
            feedback_logs.append({
                'syntax_invalid': [(ct.name, refs, (s, e)) for ct, refs, (s, e) in invalid_constraints],
                'syntax_valid': [(ct.name, refs, (s, e)) for ct, refs, (s, e) in valid_constraints],
                'solver_invalid': [(valid_constraints[cid][0].name, valid_constraints[cid][1], valid_constraints[cid][2]) for cid in removed_constraint_ids],
                'input': tokens_to_readable(sample_x_init),
                'after_syntax_filtering': tokens_to_readable(sample_x_after_syntax_filtering),
                'after_solver_filtering': tokens_to_readable(sample_x_after_solver_filtering),
            })

            x[b] = sample_x

        batch['c_val'] = x
        return batch

    model_module.ProbPath._apply_solver_feedback = logged_apply_solver_feedback

    torch.manual_seed(seed)
    with torch.no_grad():
        solver_sample, solver_intermediates = model_module.ProbPath.generate_sample(
            deepcopy(batch),
            num_steps=64,
            start_tokens=[int(ConstraintToken.Start)],
            constraint_blockwise_masking=True
        )

    # Restore originals
    model_module.ProbPath._apply_solver_feedback = original_apply

    # Evaluate with FreeCAD
    doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
        batch['val'][0],
        solver_sample[0],
        config.data.num_position_bins,
        token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
    )
    solver_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    with open("history_with_solver.py", "w") as f:
        f.write("\n".join(history_commands))

    print(f"\n  Status: {solver_status.Status}")
    print(f"  DoF: {solver_status.DoF}")
    print(f"  Malformed: {solver_status.MalformedConstraints}")
    print(f"  Conflicting: {solver_status.ConflictingConstraints}")
    print(f"  Redundant: {solver_status.RedundantConstraints}")

    # Print feedback summary
    print("\n[STEP-BY-STEP SUMMARY]")
    for i, log in enumerate(feedback_logs):
        print(f"\n<<< Step {i} >>>")
        print('Init:', log['input'])
        print('Syntax Invalid:', log['syntax_invalid'])
        print('After syntax filtering:', log['after_syntax_filtering'])
        print('Solver Invalid:', log['solver_invalid'])
        print('After solver filtering:', log['after_solver_filtering'])

    print("\n[SOLVER FINAL RESULT]")
    print_constraints_readable(solver_sample[0], prefix="  ")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'GT':<15} {'Baseline':<15} {'With Solver':<15}")
    print("-" * 70)
    print(f"{'DoF':<25} {gt_status.DoF:<15} {baseline_status.DoF:<15} {solver_status.DoF:<15}")
    print(f"{'Malformed':<25} {len(gt_status.MalformedConstraints):<15} {len(baseline_status.MalformedConstraints):<15} {len(solver_status.MalformedConstraints):<15}")
    print(f"{'Conflicts':<25} {len(gt_status.ConflictingConstraints):<15} {len(baseline_status.ConflictingConstraints):<15} {len(solver_status.ConflictingConstraints):<15}")
    print(f"{'Redundant':<25} {len(gt_status.RedundantConstraints):<15} {len(baseline_status.RedundantConstraints):<15} {len(solver_status.RedundantConstraints):<15}")
    print(f"{'Status':<25} {gt_status.Status:<15} {baseline_status.Status:<15} {solver_status.Status:<15}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_module = ConstraintMDDModule.load_from_checkpoint(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model_module.to(args.device)
    model_module.eval()

    config = model_module.hparams
    config.data.primitive_noise.enabled = False
    config.data.batch_size = 1
    config.data.num_data_workers = 0
    datamodule = ConstraintDataModule(config.data)
    datamodule.setup()

    dataset = datamodule._dataset
    assert dataset is not None

    data_item = dataset[args.index]
    batch = {}
    for k, v in data_item.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(args.device)
        else:
            batch[k] = [v]

    print(f"\n{'='*80}")
    print(f"DATA SAMPLE: Index {args.index}")
    print(f"{'='*80}")

    evaluate(model_module, batch, args.seed)
