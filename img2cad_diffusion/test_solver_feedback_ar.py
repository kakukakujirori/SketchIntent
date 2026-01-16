"""
Solver-guided autoregressive constraint generation.
Generates constraints one block (4 tokens) at a time with FreeCAD solver feedback.
"""
import argparse
import json
from dataclasses import dataclass, field, asdict
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from img2cad_diffusion.constraint_data import ConstraintDataModule
from img2cad_diffusion.constraint_models_ar import ConstraintARModule
from img2cad_diffusion.solver_feedback import (
    get_solver_status,
    tokens_to_freecad_sketch,
    validate_constraint_geometry_impact,
    SolverStatus,
    HAS_FREECAD,
)
from img2cad_diffusion.detokenization import reconstruct_constraints, reconstruct_sketch
from img2cad_diffusion.tokenization import CONSTRAINT_TOKEN_LEN
from img2cad_diffusion.tokens import ConstraintToken

if HAS_FREECAD:
    import FreeCAD


@dataclass
class IntermediateState:
    """Stores intermediate state for debugging."""
    step: int
    constraint_block_idx: int  # Which constraint block just finished
    tokens: list[str]  # Human-readable tokens
    solver_status: dict | None  # Serializable solver status
    action: str  # "continue", "stop_fc", "stop_max_iter", "remove_constraints", "force_continue"
    removed_constraints: list[tuple[str, list[int]]] = field(default_factory=list)
    dof: int = -1


def tokens_to_readable(tokens: torch.Tensor) -> list[str]:
    """Convert token tensor to human-readable list."""
    result = []
    for t in tokens.cpu().numpy():
        t = int(t)
        if t == ConstraintToken.Start:
            result.append("[START]")
        elif t == ConstraintToken.Stop:
            result.append("[STOP]")
        elif t == ConstraintToken.Pad:
            result.append("[PAD]")
        elif t == ConstraintToken.Mask:
            result.append("[MASK]")
        elif t == ConstraintToken.InnerPad:
            result.append("[IPAD]")
        elif t < len(ConstraintToken):
            result.append(ConstraintToken(t).name)
        else:
            result.append(f"R{t - len(ConstraintToken)}")
    return result


def print_constraints_readable(tokens: torch.Tensor, prefix: str = ""):
    """Print constraint tokens in block format."""
    readable = tokens_to_readable(tokens)
    print(f"{prefix}Tokens: {readable[:1]}", end="")  # START

    idx = 1
    constraint_idx = 0
    while idx < len(readable):
        block = readable[idx:idx+4]
        if not block or block[0] in ["[PAD]", "[STOP]"]:
            remaining = readable[idx:]
            if len(remaining) > 10:
                print(f" ... ({len(remaining)} tail tokens)")
            else:
                print(f" {remaining}")
            break
        print(f" c{constraint_idx}:{block}", end="")
        constraint_idx += 1
        idx += 4
    else:
        print()


def repack_constraints(
    valid_constraints: list[tuple[ConstraintToken, list[int], tuple[int, int]]],
    remove_indices: set[int],
    max_len: int,
) -> torch.Tensor:
    """Rebuild token sequence after removing some constraints.

    Args:
        valid_constraints: List of (constraint_type, refs, (start, end))
        remove_indices: Set of constraint indices to remove
        max_len: Maximum token length

    Returns:
        New token sequence with START and remaining constraints
    """
    new_tokens = torch.full((max_len,), ConstraintToken.Pad, dtype=torch.long)
    new_tokens[0] = ConstraintToken.Start

    pos = 1
    for idx, (ctype, refs, _) in enumerate(valid_constraints):
        if idx in remove_indices:
            continue

        # Write constraint type
        new_tokens[pos] = ctype.value
        pos += 1

        # Write references (up to 3, pad with InnerPad)
        for i in range(CONSTRAINT_TOKEN_LEN - 1):
            if i < len(refs):
                new_tokens[pos] = refs[i] + len(ConstraintToken)
            else:
                new_tokens[pos] = ConstraintToken.InnerPad
            pos += 1

    return new_tokens


def cleanup_constraints(
    prim_tokens: torch.Tensor,
    cons_tokens: torch.Tensor,
    num_bins: int,
    max_len: int,
    max_cleanup_iters: int = 10,
) -> tuple[torch.Tensor, SolverStatus | None, list[str] | None]:
    """Remove malformed/conflicting/redundant constraints iteratively.

    Args:
        prim_tokens: Primitive tokens [L_prim]
        cons_tokens: Constraint tokens [L_cons]
        num_bins: Number of quantization bins
        max_len: Maximum token length for repacking
        max_cleanup_iters: Maximum cleanup iterations

    Returns:
        (cleaned_tokens, final_status, history_commands)
    """
    current_tokens = cons_tokens.clone()
    final_status = None
    history_commands = None
    max_conflict_retries = 3  # Retry if unexpected conflicting constraints appear

    for iter_idx in range(max_cleanup_iters):
        # Retry loop for non-deterministic FreeCAD solver behavior
        for retry in range(max_conflict_retries):
            try:
                doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
                    prim_tokens, current_tokens, num_bins, CONSTRAINT_TOKEN_LEN)
                status = get_solver_status(fc_sketch, mapping)
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                break

            # If no conflicting (expected for GT), proceed
            if not status.MappedConflictingIndices:
                break
            # Conflicting found - retry with fresh FreeCAD document

        final_status = status

        # Prioritize: malformed > conflicting > redundant
        # Remove all malformed/conflicting at once (they must go)
        # Remove only ONE redundant at a time
        if status.MappedMalformedIndices or status.MappedConflictingIndices:
            problem_indices = set(
                status.MappedMalformedIndices +
                status.MappedConflictingIndices
            )
        elif status.MappedRedundantIndices:
            # Remove only the LAST redundant constraint (highest index)
            problem_indices = {max(status.MappedRedundantIndices)}
        else:
            # No problems
            break

        # Remove problematic constraints
        valid_constraints, _ = reconstruct_constraints(
            current_tokens.cpu().numpy(), token_len_per_constraint=CONSTRAINT_TOKEN_LEN)

        new_tokens = repack_constraints(valid_constraints, problem_indices, max_len)
        current_tokens = new_tokens.to(cons_tokens.device)

    return current_tokens, final_status, history_commands


def generate_sample_with_solver_feedback(
    model: ConstraintARModule,
    batch: dict,
    num_bins: int,
    max_len: int,
    max_iterations: int = 256,
    fully_constrained_dof: int = 2,
) -> tuple[torch.Tensor, list[IntermediateState], list[str] | None]:
    """Generate constraints with solver feedback.

    Args:
        model: The AR model
        batch: Input batch (batch_size must be 1)
        num_bins: Number of quantization bins
        max_len: Maximum token sequence length
        max_iterations: Maximum total generation iterations (prevent infinite loops)
        fully_constrained_dof: DoF threshold for fully constrained (default 2)

    Returns:
        (generated_tokens, intermediate_states, history_commands)
    """
    device = model.device

    # Initialize generated sequence
    generated = torch.full((1, max_len), ConstraintToken.Pad, device=device, dtype=torch.long)
    generated[0, 0] = ConstraintToken.Start

    intermediate_states: list[IntermediateState] = []
    history_commands: list[str] | None = None

    pos = 1  # Current position (after START)
    iteration = 0
    constraint_block_idx = 0
    force_continue = False  # If True, prevent STOP token

    while iteration < max_iterations and pos < max_len - 1:
        # Generate one block of 4 tokens
        for block_pos in range(CONSTRAINT_TOKEN_LEN):
            if pos >= max_len - 1:
                break

            # Create batch with current generated tokens
            gen_batch = {
                'val': batch['val'],
                'coord': batch['coord'],
                'pos': batch['pos'],
                'c_val': generated,
            }

            # Forward pass
            output = model.model(gen_batch)  # [1, L, V]

            # Get logits for current position
            next_token_logits = output[0, pos - 1, :].clone()  # [V]

            # Always disallow MASK token
            next_token_logits[ConstraintToken.Mask] = -torch.inf

            # If force_continue is True, disallow STOP token
            if force_continue:
                next_token_logits[ConstraintToken.Stop] = -torch.inf

            # Sample next token
            next_token = torch.distributions.Categorical(logits=next_token_logits).sample()
            generated[0, pos] = next_token
            pos += 1
            iteration += 1

            # Check if STOP was generated
            if next_token == ConstraintToken.Stop:
                # Record final state
                intermediate_states.append(IntermediateState(
                    step=iteration,
                    constraint_block_idx=constraint_block_idx,
                    tokens=tokens_to_readable(generated[0]),
                    solver_status=None,
                    action="stop_generated",
                    dof=-1,
                ))
                # Build final sketch for history
                try:
                    doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
                        batch['val'][0], generated[0], num_bins, CONSTRAINT_TOKEN_LEN)
                    FreeCAD.closeDocument(doc.Name)
                except Exception:
                    pass
                return generated, intermediate_states, history_commands

        # After completing a 4-token block, check with solver
        constraint_block_idx += 1
        force_continue = False  # Reset for next block

        # Get solver status
        try:
            doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
                batch['val'][0], generated[0], num_bins, CONSTRAINT_TOKEN_LEN)
            status = get_solver_status(fc_sketch, mapping)
            FreeCAD.closeDocument(doc.Name)
        except Exception as e:
            # Conversion failed, record and continue
            intermediate_states.append(IntermediateState(
                step=iteration,
                constraint_block_idx=constraint_block_idx,
                tokens=tokens_to_readable(generated[0]),
                solver_status={"error": str(e)},
                action="continue_after_error",
                dof=-1,
            ))
            continue

        # Check for problematic constraints first
        has_problems = (
            status.MappedMalformedIndices or
            status.MappedConflictingIndices or
            status.MappedRedundantIndices
        )

        # Check DoF - only stop if fully constrained AND no problems
        if status.DoF <= fully_constrained_dof and not has_problems:
            # Fully constrained with no issues! Stop generation
            intermediate_states.append(IntermediateState(
                step=iteration,
                constraint_block_idx=constraint_block_idx,
                tokens=tokens_to_readable(generated[0]),
                solver_status={"DoF": status.DoF, "Status": status.Status},
                action="stop_fc",
                dof=status.DoF,
            ))
            return generated, intermediate_states, history_commands

        # Handle problematic constraints
        if has_problems:
            problem_indices = set(
                status.MappedMalformedIndices +
                status.MappedConflictingIndices +
                status.MappedRedundantIndices
            )

            # Get current valid constraints
            valid_constraints, _ = reconstruct_constraints(
                generated[0].cpu().numpy(), token_len_per_constraint=CONSTRAINT_TOKEN_LEN)

            # Record removed constraints for logging
            removed = []
            for idx in problem_indices:
                if idx < len(valid_constraints):
                    ctype, refs, _ = valid_constraints[idx]
                    removed.append((ctype.name, refs))

            # Repack constraints without problematic ones
            new_generated = repack_constraints(valid_constraints, problem_indices, max_len)
            new_generated = new_generated.to(device)

            # Find new position (after last valid constraint)
            remaining_count = len(valid_constraints) - len(problem_indices)
            pos = 1 + remaining_count * CONSTRAINT_TOKEN_LEN

            # Update generated
            generated[0] = new_generated

            intermediate_states.append(IntermediateState(
                step=iteration,
                constraint_block_idx=constraint_block_idx,
                tokens=tokens_to_readable(generated[0]),
                solver_status={
                    "DoF": status.DoF,
                    "Malformed": status.MappedMalformedIndices,
                    "Conflicting": status.MappedConflictingIndices,
                    "Redundant": status.MappedRedundantIndices,
                },
                action="remove_constraints",
                removed_constraints=removed,
                dof=status.DoF,
            ))

            # Force continue since we removed constraints
            force_continue = True
        else:
            # No problems, but underconstrained (DoF > 2)
            # Force continue to add more constraints
            force_continue = True
            intermediate_states.append(IntermediateState(
                step=iteration,
                constraint_block_idx=constraint_block_idx,
                tokens=tokens_to_readable(generated[0]),
                solver_status={"DoF": status.DoF, "Status": status.Status},
                action="force_continue",
                dof=status.DoF,
            ))

    # Max iterations reached - record before final cleanup
    intermediate_states.append(IntermediateState(
        step=iteration,
        constraint_block_idx=constraint_block_idx,
        tokens=tokens_to_readable(generated[0]),
        solver_status=None,
        action="stop_max_iter",
        dof=-1,
    ))

    # ===========================================================================
    # FINAL CLEANUP PASS: Remove any remaining malformed/conflicting/redundant
    # ===========================================================================
    max_cleanup_iters = 10
    for cleanup_iter in range(max_cleanup_iters):
        try:
            doc, fc_sketch, history_commands, mapping = tokens_to_freecad_sketch(
                batch['val'][0], generated[0], num_bins, CONSTRAINT_TOKEN_LEN)
            status = get_solver_status(fc_sketch, mapping)
            FreeCAD.closeDocument(doc.Name)
        except Exception as e:
            intermediate_states.append(IntermediateState(
                step=iteration,
                constraint_block_idx=constraint_block_idx,
                tokens=tokens_to_readable(generated[0]),
                solver_status={"error": str(e)},
                action="cleanup_error",
                dof=-1,
            ))
            break

        problem_indices = set(
            status.MappedMalformedIndices +
            status.MappedConflictingIndices +
            status.MappedRedundantIndices
        )

        if not problem_indices:
            # Clean! Record and exit
            intermediate_states.append(IntermediateState(
                step=iteration,
                constraint_block_idx=constraint_block_idx,
                tokens=tokens_to_readable(generated[0]),
                solver_status={"DoF": status.DoF, "Status": status.Status},
                action="cleanup_done",
                dof=status.DoF,
            ))
            break

        # Remove problematic constraints
        valid_constraints, _ = reconstruct_constraints(
            generated[0].cpu().numpy(), token_len_per_constraint=CONSTRAINT_TOKEN_LEN)

        removed = []
        for idx in problem_indices:
            if idx < len(valid_constraints):
                ctype, refs, _ = valid_constraints[idx]
                removed.append((ctype.name, refs))

        new_generated = repack_constraints(valid_constraints, problem_indices, max_len)
        generated[0] = new_generated.to(device)

        intermediate_states.append(IntermediateState(
            step=iteration,
            constraint_block_idx=constraint_block_idx,
            tokens=tokens_to_readable(generated[0]),
            solver_status={
                "DoF": status.DoF,
                "Malformed": status.MappedMalformedIndices,
                "Conflicting": status.MappedConflictingIndices,
                "Redundant": status.MappedRedundantIndices,
            },
            action=f"cleanup_remove_{cleanup_iter+1}",
            removed_constraints=removed,
            dof=status.DoF,
        ))

    return generated, intermediate_states, history_commands


def evaluate(
    model: ConstraintARModule,
    batch: dict,
    config,
    seed: int = 42,
    geo_threshold_mm: float = 50.0,
):
    """Evaluate solver-guided generation vs baseline."""
    from img2cad_diffusion.convert_sketch_to_freecad import FreeCADConversionError

    num_bins = config.data.num_position_bins
    max_len = config.data.max_token_length
    device = model.device

    # Geometry validation results for each pattern
    geo_results = {
        'gt_raw': {'pass': None, 'max_dev': None},
        'gt_refined': {'pass': None, 'max_dev': None},
        'baseline': {'pass': None, 'max_dev': None},
        'solver': {'pass': None, 'max_dev': None},
    }

    def run_geo_validation(prim_tokens, cons_tokens, key):
        """Run geometry validation and store results."""
        try:
            sketch = reconstruct_sketch(
                prim_tokens.cpu().numpy(), cons_tokens.cpu().numpy(),
                num_bins, CONSTRAINT_TOKEN_LEN)
            results, fc_sketch, _, _ = validate_constraint_geometry_impact(
                sketch, threshold_mm=geo_threshold_mm, verbose=False)
            FreeCAD.closeDocument(fc_sketch.Document.Name)

            max_dev = max((r.max_deviation for r in results), default=0)
            passed = all(r.is_preserved for r in results)
            geo_results[key] = {'pass': passed, 'max_dev': max_dev}
        except Exception as e:
            geo_results[key] = {'pass': False, 'max_dev': -1, 'error': str(e)[:30]}

    # Show ground truth
    gt_tokens = batch['c_val'][0]
    print("\n[GROUND TRUTH CONSTRAINTS (raw)]")
    print_constraints_readable(gt_tokens, prefix="  ")

    # Evaluate GT with FreeCAD
    gt_status = None
    gt_refined_tokens = None
    gt_refined_status = None
    try:
        doc, fc_sketch, gt_history, mapping = tokens_to_freecad_sketch(
            batch['val'][0], gt_tokens, num_bins, CONSTRAINT_TOKEN_LEN)
        gt_status = get_solver_status(fc_sketch, mapping)
        FreeCAD.closeDocument(doc.Name)

        with open("history_gt_ar.py", "w") as f:
            f.write("\n".join(gt_history))
        print("  Saved: history_gt_ar.py")

        print(f"\n  DoF: {gt_status.DoF}, Status: {gt_status.Status}")
        print(f"  Malformed: {gt_status.MappedMalformedIndices}")
        print(f"  Conflicting: {gt_status.MappedConflictingIndices}")
        print(f"  Redundant: {gt_status.MappedRedundantIndices}")

        # Cleanup GT if needed
        problem_count = (len(gt_status.MappedMalformedIndices) +
                        len(gt_status.MappedConflictingIndices) +
                        len(gt_status.MappedRedundantIndices))
        if problem_count > 0:
            print("\n[GROUND TRUTH CONSTRAINTS (refined)]")
            gt_refined_tokens, gt_refined_status, gt_refined_history = cleanup_constraints(
                batch['val'][0], gt_tokens, num_bins, max_len)
            print_constraints_readable(gt_refined_tokens, prefix="  ")

            if gt_refined_history:
                with open("history_gt_refined_ar.py", "w") as f:
                    f.write("\n".join(gt_refined_history))
                print("  Saved: history_gt_refined_ar.py")

            if gt_refined_status:
                print(f"\n  DoF: {gt_refined_status.DoF}, Status: {gt_refined_status.Status}")
                print(f"  Malformed: {gt_refined_status.MappedMalformedIndices}")
                print(f"  Conflicting: {gt_refined_status.MappedConflictingIndices}")
                print(f"  Redundant: {gt_refined_status.MappedRedundantIndices}")

    except FreeCADConversionError as e:
        print(f"\n  ERROR: Failed to convert GT to FreeCAD: {e}")
        print("  Skipping GT evaluation.")

    # Run geometry validation for GT
    if gt_status:
        run_geo_validation(batch['val'][0], gt_tokens, 'gt_raw')
    if gt_refined_tokens is not None:
        run_geo_validation(batch['val'][0], gt_refined_tokens, 'gt_refined')
    elif gt_status:
        geo_results['gt_refined'] = geo_results['gt_raw'].copy()

    # =========================================================================
    # RUN 1: WITHOUT SOLVER FEEDBACK (baseline)
    # =========================================================================
    print(f"\n{'='*80}")
    print("RUN 1: WITHOUT SOLVER FEEDBACK (baseline)")
    print(f"{'='*80}")

    torch.manual_seed(seed)
    with torch.no_grad():
        baseline_sample = model.generate_sample(batch, max_len, dtype=torch.long)

    print("\n[BASELINE FINAL RESULT]")
    print_constraints_readable(baseline_sample[0], prefix="  ")

    # Evaluate baseline
    doc, fc_sketch, baseline_history, mapping = tokens_to_freecad_sketch(
        batch['val'][0], baseline_sample[0], num_bins, CONSTRAINT_TOKEN_LEN)
    baseline_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    with open("history_without_solver_ar.py", "w") as f:
        f.write("\n".join(baseline_history))
    print("  Saved: history_without_solver_ar.py")

    print(f"\n  DoF: {baseline_status.DoF}, Status: {baseline_status.Status}")
    print(f"  Malformed: {baseline_status.MappedMalformedIndices}")
    print(f"  Conflicting: {baseline_status.MappedConflictingIndices}")
    print(f"  Redundant: {baseline_status.MappedRedundantIndices}")

    # Run geometry validation for baseline
    run_geo_validation(batch['val'][0], baseline_sample[0], 'baseline')

    # =========================================================================
    # RUN 2: WITH SOLVER FEEDBACK
    # =========================================================================
    print(f"\n{'='*80}")
    print("RUN 2: WITH SOLVER FEEDBACK")
    print(f"{'='*80}")

    torch.manual_seed(seed)
    with torch.no_grad():
        solver_sample, intermediates, solver_history = generate_sample_with_solver_feedback(
            model, batch, num_bins, max_len, max_iterations=256)

    print("\n[STEP-BY-STEP SUMMARY]")
    for state in intermediates:
        print(f"\n  Step {state.step} (block {state.constraint_block_idx}): action={state.action}, DoF={state.dof}")
        if state.removed_constraints:
            print(f"    Removed: {state.removed_constraints}")
        if state.solver_status:
            print(f"    Status: {state.solver_status}")

    print("\n[SOLVER-GUIDED FINAL RESULT]")
    print_constraints_readable(solver_sample[0], prefix="  ")

    # Evaluate solver-guided result
    doc, fc_sketch, solver_history_final, mapping = tokens_to_freecad_sketch(
        batch['val'][0], solver_sample[0], num_bins, CONSTRAINT_TOKEN_LEN)
    solver_status = get_solver_status(fc_sketch, mapping)
    FreeCAD.closeDocument(doc.Name)

    with open("history_with_solver_ar.py", "w") as f:
        f.write("\n".join(solver_history_final))
    print("  Saved: history_with_solver_ar.py")

    print(f"\n  DoF: {solver_status.DoF}, Status: {solver_status.Status}")
    print(f"  Malformed: {solver_status.MappedMalformedIndices}")
    print(f"  Conflicting: {solver_status.MappedConflictingIndices}")
    print(f"  Redundant: {solver_status.MappedRedundantIndices}")

    # Run geometry validation for solver-guided result
    run_geo_validation(batch['val'][0], solver_sample[0], 'solver')

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'GT (raw)':<12} {'GT (refined)':<12} {'Baseline':<12} {'With Solver':<12}")
    print("-" * 68)

    # GT raw values
    gt_raw_dof = gt_status.DoF if gt_status else "N/A"
    gt_raw_malformed = len(gt_status.MappedMalformedIndices) if gt_status else "N/A"
    gt_raw_conflicting = len(gt_status.MappedConflictingIndices) if gt_status else "N/A"
    gt_raw_redundant = len(gt_status.MappedRedundantIndices) if gt_status else "N/A"
    gt_raw_stat = gt_status.Status if gt_status else "N/A"

    # GT refined values (use raw if no refinement needed)
    gt_final_status = gt_refined_status if gt_refined_status else gt_status
    gt_ref_dof = gt_final_status.DoF if gt_final_status else "N/A"
    gt_ref_malformed = len(gt_final_status.MappedMalformedIndices) if gt_final_status else "N/A"
    gt_ref_conflicting = len(gt_final_status.MappedConflictingIndices) if gt_final_status else "N/A"
    gt_ref_redundant = len(gt_final_status.MappedRedundantIndices) if gt_final_status else "N/A"
    gt_ref_stat = gt_final_status.Status if gt_final_status else "N/A"

    print(f"{'DoF':<20} {gt_raw_dof:<12} {gt_ref_dof:<12} {baseline_status.DoF:<12} {solver_status.DoF:<12}")
    print(f"{'Malformed':<20} {gt_raw_malformed:<12} {gt_ref_malformed:<12} {len(baseline_status.MappedMalformedIndices):<12} {len(solver_status.MappedMalformedIndices):<12}")
    print(f"{'Conflicting':<20} {gt_raw_conflicting:<12} {gt_ref_conflicting:<12} {len(baseline_status.MappedConflictingIndices):<12} {len(solver_status.MappedConflictingIndices):<12}")
    print(f"{'Redundant':<20} {gt_raw_redundant:<12} {gt_ref_redundant:<12} {len(baseline_status.MappedRedundantIndices):<12} {len(solver_status.MappedRedundantIndices):<12}")
    print(f"{'Status':<20} {gt_raw_stat:<12} {gt_ref_stat:<12} {baseline_status.Status:<12} {solver_status.Status:<12}")

    # Geometry validation row
    def geo_str(key):
        r = geo_results[key]
        if r['pass'] is None:
            return 'N/A'
        elif r['max_dev'] < 0:
            return 'ERROR'
        else:
            status = 'PASS' if r['pass'] else 'FAIL'
            return f"{status}/{r['max_dev']:.0f}mm"

    print(f"{'Geometry (<' + str(int(geo_threshold_mm)) + 'mm)':<20} {geo_str('gt_raw'):<12} {geo_str('gt_refined'):<12} {geo_str('baseline'):<12} {geo_str('solver'):<12}")

    # Save intermediate states to JSON for detailed analysis
    with open("intermediate_states_ar.json", "w") as f:
        json.dump([asdict(s) for s in intermediates], f, indent=2)
    print("\nSaved intermediate states to: intermediate_states_ar.json")


if __name__ == '__main__':
    if not HAS_FREECAD:
        print("ERROR: FreeCAD is required for solver feedback. Exiting.")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help="Path to AR model checkpoint")
    parser.add_argument('--index', type=int, default=0, help="Data sample index to process")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

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
    test_dataset = test_loader.dataset
    data_item = test_dataset[args.index]
    batch = {}
    for k, v in data_item.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(args.device)
        else:
            batch[k] = [v]

    print(f"\n{'='*80}")
    print(f"DATA SAMPLE: test index {args.index}")
    print(f"{'='*80}")

    evaluate(model, batch, config, args.seed)
