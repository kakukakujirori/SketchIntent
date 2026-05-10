"""
Test-time inference with solver intervention for CAD constraint inference model.

Algorithm:
1. Load model from checkpoint and input data (.npy)
2. Generate full constraint token sequence autoregressively
3. Run CAD solver to detect malformed / conflicting / redundant constraints
4. Remove invalid constraint blocks, left-justify valid ones
5. Regenerate tokens for the remaining (Pad) positions autoregressively
6. Repeat steps 3-5 up to N times
7. Decode final sequence to FreeCAD commands
8. Log all intermediate results

Usage:
    python src/test.py --ckpt logs/train/runs/2026-03-12_20-30-34/checkpoints/epoch_2999.ckpt \
                       --data data_ryotaro_prim5.npy --n_iter 5 --batch_size 8
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.constraint_dataset import ConstraintDataset
from src.data.constraint_datamodule import sketch_collate_fn
from src.data.components.primitive_noise import PrimitiveNoise
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.sketch_handlers.constants import ConstraintToken, CONSTRAINT_VAL_OFFSET
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_freecad_commands
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_primitives, cleanse_sketch_constraints,
    complete_constraint_values, normalize_sketch)


def generate_with_partial_sequence(
    model: L.LightningModule,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Regenerate tokens autoregressively, keeping already-filled positions fixed.

    Args:
        model: The autoregressive model.
        batch: Batch dict with p_val, p_attr, p_pos, partial c_val.
        device: Device.

    Returns:
        generated: Full generated sequence [B, L].
    """
    start_sequence = batch['c_val']
    batch_size, max_len = start_sequence.shape
    generated = start_sequence.clone()

    assert not (generated == ConstraintToken.Stop).any(), \
        "Precondition violated: Stop token found in input. Caller must strip Stop before calling this function."
    assert (generated == ConstraintToken.Pad).any(dim=1).all(), \
        "Precondition violated: every sample must have at least one Pad token. " \
        "Caller must exclude fully-packed samples before calling this function."

    # Find the first Pad position per sample to start generation from
    pos = (generated == ConstraintToken.Pad).to(torch.long).argmax(dim=1)  # [B]
    assert pos.min() >= 1, f"Precondition violated: sequence should start from the [START] token." \

    for cnt in range(max_len - pos.min()):
        # Check if any sample needs generation
        pos_mask = torch.arange(max_len, device=device).unsqueeze(0) < pos.unsqueeze(1)  # [B, max_len]
        already_stopped = ((generated == ConstraintToken.Stop) & pos_mask).any(dim=1)    # [B]
        already_maxed = pos >= max_len                                                   # [B]
        need_generation = ~(already_stopped | already_maxed)                             # [B]
        if not need_generation.any():
            break

        # Build batch and forward for the can-generate sub-batch
        gen_batch = {
            'p_val': batch['p_val'][need_generation],
            'p_attr': batch['p_attr'][need_generation],
            'p_pos': batch['p_pos'][need_generation],
            'c_val': generated[need_generation],
        }
        output = model.forward(gen_batch)  # [num_need_generation, L, V]

        # Generate next token for the sub-batch (NOTE: generated token is for pos-1)
        generated_pos = pos[need_generation] - 1  # [num_need_generation]
        batch_idx = torch.arange(generated_pos.shape[0], device=device)
        next_token_logits = output[batch_idx, generated_pos, :]  # [num_need_generation, V]

        # logit masking (NOTE: For the first generation, forbid the generation of STOP)
        next_token_logits[:, ConstraintToken.Mask] = -torch.inf
        if cnt == 0:
            next_token_logits[:, ConstraintToken.Stop] = -torch.inf

        dist = torch.distributions.Categorical(logits=next_token_logits)
        next_token = dist.sample()

        # Update generated tokens back in the full tensor (clamp for safety)
        gen_idx = need_generation.nonzero(as_tuple=True)[0]  # [num_need_generation]
        generated[gen_idx, generated_pos + 1] = next_token

        # Advance pos for samples that can still generate
        pos[need_generation] += 1

    # Post-processing pass: ensure all tokens after the first Stop are strictly Pad
    is_stop_mask = (generated == ConstraintToken.Stop)
    if is_stop_mask.any():
        # argmax gives the first index where Stop appears
        first_stop_indices = is_stop_mask.to(torch.long).argmax(dim=1)
        has_stop = is_stop_mask.any(dim=1)
        # Create a boolean mask of shape [B, L] where True means "beyond the first Stop"
        sequence_indices = torch.arange(max_len, device=device).unsqueeze(0)
        after_stop_mask = has_stop.unsqueeze(1) & (sequence_indices > first_stop_indices.unsqueeze(1))
        generated[after_stop_mask] = ConstraintToken.Pad

    return generated


def remove_invalid_tokens(
    tokens: np.ndarray,
    invalid_positions: set[int],
    align_constraint_token_length_to: int | None,
) -> np.ndarray:
    """Remove invalid tokens and STOP tokens, and scoot the rest to the left."""
    assert len(tokens.shape) == 1

    # Since the sample is invalid, we must strip the Stop token for re-generation
    valid_mask = np.ones(len(tokens), dtype=bool)
    valid_mask[tokens == ConstraintToken.Stop] = False

    if align_constraint_token_length_to is not None and align_constraint_token_length_to > 0:
        # Fixed-block mode: remove entire blocks
        for pos in sorted(invalid_positions):
            valid_mask[pos : pos + align_constraint_token_length_to] = False
    else:
        # Variable-length mode
        for pos in invalid_positions:
            offset = 0
            while pos + offset < len(tokens) and tokens[pos + offset] >= CONSTRAINT_VAL_OFFSET:
                offset += 1
            valid_mask[pos : pos + offset + 1] = False

    valid_tokens = tokens[valid_mask].copy()
    result = np.full_like(tokens, ConstraintToken.Pad)
    result[:len(valid_tokens)] = valid_tokens
    return result


def finalize_constraint_tokens(tokens: np.ndarray) -> np.ndarray:
    """Restore a Stop token before final decoding if the retry loop ended on Pads."""
    result = tokens.copy()
    if (result == ConstraintToken.Stop).any():
        return result

    pad_positions = np.flatnonzero(result == ConstraintToken.Pad)
    if len(pad_positions) > 0:
        result[pad_positions[0]] = ConstraintToken.Stop
    return result


def batch_item(batch: dict, key: str, index: int, default=None):
    if key not in batch:
        return default

    value = batch[key]
    if isinstance(value, torch.Tensor):
        item = value[index]
        if item.numel() == 1:
            return item.item()
        return item.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        item = value[index]
        return item.item() if hasattr(item, "item") else item
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def sample_metadata(batch: dict, batch_idx: int, batch_size: int, sample_idx: int) -> dict:
    dataset_index = batch_item(batch, "sketch_index", sample_idx, batch_idx * batch_size + sample_idx)
    source_file = batch_item(batch, "source_file", sample_idx)
    sketch_id = batch_item(batch, "sketch_id", sample_idx)
    return {
        "dataset_index": int(dataset_index) if isinstance(dataset_index, (int, np.integer)) else dataset_index,
        "source_file": str(source_file) if source_file is not None else None,
        "sketch_id": str(sketch_id) if sketch_id is not None else None,
    }


def sample_display_name(metadata: dict) -> str:
    if metadata.get("source_file"):
        return Path(metadata["source_file"]).name
    if metadata.get("sketch_id"):
        return metadata["sketch_id"]
    return f"idx={metadata['dataset_index']}"


def data_directory(data_path: str) -> Path:
    path = Path(data_path).expanduser()
    return path if path.is_dir() else path.parent


def result_script_name(result: dict) -> str:
    if result.get("source_file"):
        return Path(result["source_file"]).name
    return f"{result['sample_idx']:04d}.py"


def load_source_primitives_sketch(source_path: str, solver_feedback: SolverFeedback):
    with open(source_path, "r") as f:
        script_lines = f.read().splitlines()

    primitive_only_lines = []
    for line in script_lines:
        stripped = line.strip()
        if "fc_sketch.addConstraint(" in stripped:
            continue
        if stripped in {"doc.recompute()", "FreeCAD.ActiveDocument.recompute()"}:
            continue
        primitive_only_lines.append(line)

    doc = None
    try:
        doc, fc_sketch = load_freecad_commands(primitive_only_lines)
        return solver_feedback.sketch_freecad_converter.from_freecad(fc_sketch)
    finally:
        FreeCAD, _, _ = ensure_freecad_imported()
        FreeCAD.closeDocument(doc.Name) if doc else None


def output_primitive_source_sketch(
    batch: dict,
    sample_idx: int,
    metadata: dict,
    solver_feedback: SolverFeedback,
    source_sketch_cache: dict[str, object],
):
    source_file = metadata.get("source_file")
    if not source_file:
        return batch['sketch'][sample_idx]

    source_path = str(Path(source_file).expanduser())
    if source_path not in source_sketch_cache:
        raw_sketch = load_source_primitives_sketch(source_path, solver_feedback)
        raw_sketch = cleanse_sketch_primitives(raw_sketch)
        source_sketch_cache[source_path] = raw_sketch
    return source_sketch_cache[source_path]


@torch.no_grad()
def run_test_inference(
    module: L.LightningModule,
    loader: torch.utils.data.DataLoader,
    n_iter: int = 5,
    fully_constrained_dof: int = 2,
    solver_check_one_by_one: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """Run test inference with solver intervention.

    Args:
        module: The trained LightningModule model.
        loader: DataLoader for the dataset.
        n_iter: Maximum number of solver intervention iterations.
        fully_constrained_dof: DoF threshold for "fully constrained".
        solver_check_one_by_one: If True, solver checks constraints one by one.
        device: Device to run on.

    Returns:
        List of results dicts.
    """
    context = module.hparams.context
    module.eval().to(device)
    solver_feedback = SolverFeedback(context)

    results = []
    source_sketch_cache = {}

    for batch_idx, batch in enumerate(loader):

        # Prepare batch
        batch = {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
        batch_size, max_len = batch['p_val'].shape

        start_seq = np.full((max_len,), fill_value=ConstraintToken.Pad, dtype=np.int64)
        start_seq[0] = ConstraintToken.Start
        current_tokens = [start_seq.copy() for _ in range(batch_size)]

        iteration_results = [[] for _ in range(batch_size)]
        is_valid = [False] * batch_size

        for iter_idx in range(n_iter):
            # Generate tokens for only the incomplete samples
            incomplete = [i for i in range(batch_size) if not is_valid[i]]
            if not incomplete:
                break

            start_seqs = torch.stack([
                torch.tensor(current_tokens[i], dtype=torch.long, device=device)
                for i in incomplete
            ])
            gen_batch = {
                'p_val': batch['p_val'][incomplete],
                'p_attr': batch['p_attr'][incomplete],
                'p_pos': batch['p_pos'][incomplete],
                'c_val': start_seqs,
            }
            generated_part = generate_with_partial_sequence(module, gen_batch, device)

            # Evaluate generated samples with solver
            for local_idx, global_idx in enumerate(incomplete):
                metadata = sample_metadata(batch, batch_idx, batch_size, global_idx)
                display_name = sample_display_name(metadata)
                prim_tokens = batch['p_val'][global_idx].cpu().numpy()
                cons_tokens = generated_part[local_idx].cpu().numpy()

                status, sg_sketch, constraint_to_token_start, _ = \
                    solver_feedback.evaluate_constraints(
                        prim_tokens, cons_tokens,
                        batch['sketch'][global_idx],
                        check_one_by_one=solver_check_one_by_one,
                    )

                # Record iteration result
                result_entry = {
                    'tokens': cons_tokens,
                    'status': status.Status,
                    'dof': status.DoF,
                    'malformed': status.MalformedConstraints,
                    'conflicting': status.ConflictingConstraints,
                    'redundant': status.RedundantConstraints,
                }
                iteration_results[global_idx].append(result_entry)
                if batch_size == 1:  # DEBUG
                    print("\n[run_test_inference]", f"(Iter: {iter_idx})", visualize_constraint_sequence(cons_tokens))

                # Check validity
                if (status.Status == 'Converged'
                    and not status.MalformedConstraints
                    and not status.ConflictingConstraints
                    and not status.RedundantConstraints):
                    fully_constrained = status.DoF <= fully_constrained_dof
                    max_len_reached = not (cons_tokens == ConstraintToken.Pad).any()
                    if fully_constrained or max_len_reached:
                        is_valid[global_idx] = True
                        current_tokens[global_idx] = cons_tokens
                        print(f"  Batch {batch_idx}, sample {global_idx} [{display_name}]: "
                              f"SOLVED at iter {iter_idx} (DoF={status.DoF})")
                        continue

                # Collect invalid positions from all error types
                invalid_positions = set()
                problematic_constraints = status.MalformedConstraints + status.ConflictingConstraints + status.RedundantConstraints
                if status.Status == "Failed" and len(problematic_constraints) == 0:
                    problematic_constraints = sg_sketch.constraints.keys()
                for c_id in problematic_constraints:
                    start = constraint_to_token_start.get(c_id)
                    if start is not None:
                        invalid_positions.add(start)

                # Remove invalid tokens
                current_tokens[global_idx] = remove_invalid_tokens(
                    cons_tokens,
                    invalid_positions,
                    context.align_constraint_token_length_to,
                )

                print(f"  Batch {batch_idx}, sample {global_idx} [{display_name}]: "
                      f"iter {iter_idx}: {status.Status} (DoF={status.DoF}), "
                      f"removed {len(invalid_positions)} blocks, regenerating...")

        # Store final results
        for global_idx in range(batch_size):
            metadata = sample_metadata(batch, batch_idx, batch_size, global_idx)
            # Convert final tokens to FreeCAD commands
            prim_tokens = batch['p_val'][global_idx].cpu().numpy()
            cons_tokens = finalize_constraint_tokens(current_tokens[global_idx])
            final_decode_error = None
            final_commands = []

            # Tokens -> SketchGraphs -> FreeCAD commands
            try:
                sg_sketch, _, _ = solver_feedback.sketch_tokenizer.detokenize(
                    {"p_val": prim_tokens, "c_val": cons_tokens},
                    allow_invalid_constraints=True,
                )
                if is_valid[global_idx]:
                    primitive_source_sketch = output_primitive_source_sketch(
                        batch,
                        global_idx,
                        metadata,
                        solver_feedback,
                        source_sketch_cache,
                    )
                    sg_sketch = solver_feedback.overwrite_primitives_with_original(
                        sg_sketch,
                        primitive_source_sketch,
                    )
                    sg_sketch, _ = complete_constraint_values(sg_sketch, skip_invalid_constraints=True)
                    final_commands = solver_feedback.sketch_freecad_converter.to_freecad_commands(sg_sketch)
            except Exception as e:
                final_decode_error = repr(e)

            results.append({
                'sample_idx': batch_idx * batch_size + global_idx,
                **metadata,
                'iterations': iteration_results[global_idx] or [],
                'final_tokens': cons_tokens.tolist(),
                'final_decode_error': final_decode_error,
                'solved': is_valid[global_idx],
                'final_commands': final_commands,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to input data (.npy or directory/file of .py)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--n_iter", type=int, default=5,
                        help="Max solver intervention iterations")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation")
    parser.add_argument("--fully_constrained_dof", type=int, default=2,
                        help="DoF threshold for fully constrained")
    parser.add_argument("--solver_check_one_by_one", action='store_true',
                        help="If True, solver checks constraints one by one")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    input_data_dir = data_directory(args.data).resolve()
    output_dir = Path(args.outdir).expanduser().resolve()
    if output_dir == input_data_dir:
        print(
            f"Error: --outdir must differ from the input data directory: {input_data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pre-processing if input is python data
    input_is_python_data = args.data.endswith(".py") or os.path.isdir(args.data)
    if input_is_python_data:
        input_data_dir = args.data
        temp_dir_obj = None

        # If single .py file, wrap it in a temporary directory
        if os.path.isfile(args.data) and args.data.endswith(".py"):
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_py_")
            input_data_dir = temp_dir_obj.name
            shutil.copy(args.data, input_data_dir)
            print(f"Wrapped single python file in {input_data_dir}")

        print(f"Pre-processing python data from {input_data_dir}...")
        tmp_npy = "/tmp/test_data.npy"
        try:
            # 1. Sieve
            subprocess.run([sys.executable, "scripts/qwen3_sieve_generated_data.py"], check=False)
            # 2. Convert to npy
            subprocess.run([sys.executable, "scripts/convert_freecad_to_seq.py",
                            "--data_dir", input_data_dir,
                            "--out_path", tmp_npy], check=True)
            # 3. Cleanse
            subprocess.run([sys.executable, "scripts/cleanse_seq_for_freecad.py", tmp_npy], check=True)

            # Update args.data to the cleansed npy
            args.data = tmp_npy.replace(".npy", "_unique_valid.npy")
            print(f"Pre-processing finished. Using: {args.data}")

        except subprocess.CalledProcessError as e:
            print(f"Error during pre-processing: {e}")
            sys.exit(1)
        finally:
            if temp_dir_obj:
                temp_dir_obj.cleanup()

    device = torch.device(args.device)

    # Load model and context
    config_file = os.path.join(os.path.dirname(args.ckpt).replace("checkpoints", ".hydra"), "config.yaml")
    cfg = OmegaConf.load(config_file)
    module = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    module.load_state_dict(state_dict)
    context = module.hparams.context

    # Load data
    dataset = ConstraintDataset(args.data, context=context, primitive_noise=PrimitiveNoise(std=0.0, max_diff=0.0))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sketch_collate_fn)

    # Run inference
    print(f"\nRunning test inference with {args.n_iter} iterations...")
    results = run_test_inference(
        module=module,
        loader=loader,
        n_iter=args.n_iter,
        fully_constrained_dof=args.fully_constrained_dof,
        solver_check_one_by_one=args.solver_check_one_by_one,
        device=device,
    )

    # Summary
    solved_count = sum(1 for r in results if r['solved'])
    print(f"\n=== Summary ===")
    print(f"Total samples: {len(results)}")
    print(f"Solved: {solved_count} ({100*solved_count/len(results):.1f}%)")
    print(f"Unsolved: {len(results) - solved_count}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = []
    for r in results:
        r_clean = {
            'sample_idx': r['sample_idx'],
            'dataset_index': r.get('dataset_index'),
            'source_file': r.get('source_file'),
            'sketch_id': r.get('sketch_id'),
            'solved': r['solved'],
            'final_decode_error': r.get('final_decode_error'),
            'final_commands': r['final_commands'],
        }
        if isinstance(r.get('final_tokens'), np.ndarray):
            r_clean['final_tokens'] = r['final_tokens'].tolist()
        else:
            r_clean['final_tokens'] = r.get('final_tokens')
        r_clean['iterations'] = [
            {
                'status': it['status'],
                'dof': it['dof'],
                'malformed': it['malformed'],
                'conflicting': it['conflicting'],
                'redundant': it['redundant'],
            }
            for it in r['iterations']
        ]
        results_serializable.append(r_clean)

    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save FreeCAD command scripts
    for r in results:
        if r['final_commands']:
            fc_path = output_dir / result_script_name(r)
            with open(fc_path, "w") as f:
                f.write("# Sample {:04d} | Solved: {}\n".format(r['sample_idx'], r['solved']))
                if r.get('source_file'):
                    f.write("# Source: {}\n".format(r['source_file']))
                for cmd in r['final_commands']:
                    f.write(cmd + "\n")
    print(f"FreeCAD scripts saved to {output_dir}/*.py")


if __name__ == "__main__":
    main()
