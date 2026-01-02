import json

import torch
from tqdm import tqdm

from img2cad_diffusion.constraint_data import ConstraintDataset, PrimitiveNoiseConfig
from img2cad_diffusion.solver_feedback import evaluate_constraints_batch, HAS_FREECAD


def test_fc_data_is_fc(
    sequence_file: str,
    fc_json_path: str,
    test_sample_num: int = 1000,
):
    """Test that FC data from JSON is fully constrained when directly converted to FreeCAD.

    This test uses convert_sequence_to_freecad (same as used for FC discovery),
    NOT convert_sketch_to_freecad, to ensure consistency.
    """
    if not HAS_FREECAD:
        raise ValueError("This test requires FreeCAD to be installed.")

    import sys
    import os
    # Add scripts directory for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

    from sketchgraphs.data import flat_array
    from find_fully_constrained_in_vitruvion import process_sequence

    # Load sequence data
    seq_data = flat_array.load_dictionary_flat(sequence_file)

    # Load FC indices
    with open(fc_json_path, 'r') as f:
        data = json.load(f)
    fc_indices = data.get('selected_for_validation', []) + data.get('selected_for_test', [])

    # Test
    total_results = {
        "fc": [],
        "under": [],
        "over": [],
        "redundant": [],
        "conversion_fail": [],
    }

    for i, idx in enumerate(tqdm(fc_indices[:test_sample_num], desc="Direct FC Test")):
        try:
            seq = seq_data['sequences'][idx]
            # Use the same converter as FC discovery
            status = process_sequence(seq, output_py_path=None, debug=False)

            dof = status['DoF']
            conflict = len(status['ConflictingConstraints'])
            redundant = len(status['RedundantConstraints'])

            # FC discovery uses DoF == 2 only (does NOT check conflict/redundant)
            # So we use the same criteria here
            if dof == 2:
                total_results["fc"].append(i)
            elif dof > 2:
                total_results["under"].append(i)
            else:
                total_results["over"].append(i)

            # Also track redundant/conflict for informational purposes
            if redundant > 0:
                total_results["redundant"].append(i)
            if conflict > 0:
                # Note: conflict may overlap with over (dof < 2)
                pass

        except Exception as e:
            print(f"[{i}] idx={idx}: Conversion failed: {e}")
            total_results["conversion_fail"].append(i)

    print(f"\nDirect FC Test Results: {total_results}")
    assert len(total_results["fc"]) == test_sample_num, f"Not all samples are FC (DoF==2): {total_results}"


def test_fc_data_remains_fc_through_tokenization(
    sequence_file: str,
    fc_json_path: str,
    num_position_bins: int = 64,
    max_token_length: int = 130,
    test_sample_num: int = 1000,
):
    """Test that FC data remains fully constrained (DoF==2) after tokenization/detokenization.

    Uses the same criteria as FC discovery: DoF == 2 only.
    """
    if not HAS_FREECAD:
        raise ValueError(f"This test requires FreeCAD to be installed.")

    import FreeCAD
    from img2cad_diffusion.solver_feedback import tokens_to_freecad_sketch, get_solver_status

    # Prepare data
    full_dataset = ConstraintDataset(
        sequence_file=sequence_file,
        num_bins=num_position_bins,
        max_length=max_token_length,
        primitive_noise_config=PrimitiveNoiseConfig(enabled=False),
        align_constraint_token_length=False,
        use_stop_padding=False,
        constraint_order="original",
    )

    with open(fc_json_path, 'r') as f:
        data = json.load(f)
    fc_indices = data.get('selected_for_validation', []) + data.get('selected_for_test', [])
    fc_indices = fc_indices[:test_sample_num]
    fc_dataset = torch.utils.data.Subset(full_dataset, fc_indices)
    fc_dataloader = torch.utils.data.DataLoader(fc_dataset, 1)

    # run
    total_results = {
        "fc": [],
        "under": [],
        "over": [],
        "redundant": [],
        "conversion_fail": [],
    }
    for i, batch in enumerate(tqdm(fc_dataloader, desc="Tokenization FC Test")):
        if test_sample_num <= i:
            break

        try:
            doc, fc_sketch = tokens_to_freecad_sketch(
                batch['val'][0],  # Remove batch dim
                batch['c_val'][0],
                num_position_bins,
            )

            if doc is None:
                total_results["conversion_fail"].append(i)
                continue

            status = get_solver_status(fc_sketch)
            dof = status['DoF']
            conflict = len(status['ConflictingConstraints'])
            redundant = len(status['RedundantConstraints'])

            # FC discovery uses DoF == 2 only
            if dof == 2:
                total_results["fc"].append(i)
            elif dof > 2:
                total_results["under"].append(i)
            else:
                total_results["over"].append(i)

            # Track redundant/conflict for info
            if redundant > 0:
                total_results["redundant"].append(i)

            FreeCAD.closeDocument(doc.Name)

        except Exception as e:
            print(f"[{i}] Conversion failed: {e}")
            total_results["conversion_fail"].append(i)

    print(f"\nTokenization FC Test Results: {total_results}")
    assert len(total_results["fc"]) == test_sample_num, f"Not all samples are FC (DoF==2) after tokenization: {total_results}"


if __name__ == '__main__':
    # First, test that the source FC data is actually fully constrained
    print("=" * 60)
    print("Test 1: Direct FC conversion (no tokenization)")
    print("=" * 60)
    test_fc_data_is_fc(
        sequence_file='data_processed/sg_filtered_unique.npy',
        fc_json_path='scripts/sketchgraphs_fc_search/sketchgraphs_fc_clustered.json',
        test_sample_num=10,
    )

    # Then, test that tokenization preserves FC status
    print("\n" + "=" * 60)
    print("Test 2: FC through tokenization")
    print("=" * 60)
    test_fc_data_remains_fc_through_tokenization(
        sequence_file='data_processed/sg_filtered_unique.npy',
        fc_json_path='scripts/sketchgraphs_fc_search/sketchgraphs_fc_clustered.json',
        num_position_bins=8192,
        max_token_length=8192,
        test_sample_num=10,
    )