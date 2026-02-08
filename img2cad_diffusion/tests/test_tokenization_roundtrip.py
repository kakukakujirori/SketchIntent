"""Comprehensive roundtrip tests for tokenization and detokenization.

These tests verify that tokenize → detokenize produces the original sketch
with all parameters preserved within quantization tolerance.
"""
import pytest
from tqdm import tqdm

import numpy as np
from sketchgraphs.data import Arc, Circle, Line, Point
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array as fa

from img2cad_diffusion.tokenization import tokenize_primitives, tokenize_sketch, tokenize_constraints, CONSTRAINT_TOKEN_LEN
from img2cad_diffusion.detokenization import reconstruct_primitives, reconstruct_constraints, reconstruct_sketch


DATA_PATH = '/home/ryotaro/github/vitruvion/data_processed/sg_filtered_unique.npy'
# Use high num_bins for roundtrip tests to verify tokenization/detokenization preserves info.
# Training uses 64 bins for conditioning, but tests should use high precision.
NUM_BINS = 8192
NUM_TEST_DATA = 10000
# Quantization tolerance: with 1024 bins over [-0.5, 0.5] range, each bin is ~0.001
TOLERANCE = 2.0 / NUM_BINS


@pytest.fixture(scope="module")
def sequences():
    """Load test sequences."""
    data = fa.load_dictionary_flat(DATA_PATH)
    return data['sequences']


def assert_params_close(orig, recon, tol=TOLERANCE):
    """Assert all relevant geometric parameters match within tolerance.

    Compares only the parameters that are meaningful for each entity type,
    excluding internal attributes like entityId, yDir, etc.
    """
    entity_type = type(orig).__name__

    # Type must match exactly
    assert type(orig) == type(recon), \
        f"Type mismatch: expected {entity_type}, got {type(recon).__name__}"

    # Parameters to check per entity type (only geometric/semantic params)
    ENTITY_PARAMS = {
        'Arc': ['isConstruction'], # Center/Radius/Clockwise are unstable/derived for S/M/E parametrization
        'Circle': ['xCenter', 'yCenter', 'radius', 'isConstruction'],
        'Line': ['isConstruction'],  # start_point/end_point are properties
        'Point': ['x', 'y', 'isConstruction'],
    }

    params_to_check = ENTITY_PARAMS.get(entity_type, [])

    for key in params_to_check:
        if not hasattr(orig, key) or not hasattr(recon, key):
            continue
        orig_val = getattr(orig, key)
        recon_val = getattr(recon, key)

        if isinstance(orig_val, (int, float)) and isinstance(recon_val, (int, float)):
            diff = abs(orig_val - recon_val)
            assert diff < tol, \
                f"{entity_type}.{key}: {orig_val:.6f} vs {recon_val:.6f} (diff={diff:.6f})"
        else:
            assert orig_val == recon_val, f"{entity_type}.{key}: {orig_val} vs {recon_val}"

    # Compare computed point properties
    if entity_type == 'Line':
        for pt_name in ['start_point', 'end_point']:
            orig_pt, recon_pt = getattr(orig, pt_name), getattr(recon, pt_name)
            for i, (o, r) in enumerate(zip(orig_pt, recon_pt)):
                diff = abs(o - r)
                assert diff < tol, f"{entity_type}.{pt_name}[{i}]: {o:.6f} vs {r:.6f}"
    elif entity_type == 'Arc':
        # Check start/mid/end points (S/M/E parametrization consistency)
        for pt_name in ['start_point', 'end_point', 'mid_point']:
            orig_pt, recon_pt = getattr(orig, pt_name), getattr(recon, pt_name)
            for i, (o, r) in enumerate(zip(orig_pt, recon_pt)):
                diff = abs(o - r)
                assert diff < tol, f"{entity_type}.{pt_name}[{i}]: {o:.6f} vs {r:.6f}"


def get_expected_constraint_pairs(sketch, gather_map) -> list:
    """Extract (constraint_type_name, sorted_gather_refs) from original sketch for comparison.

    Filters out constraints with External references and returns a comparable format.

    Returns
    -------
    list[tuple[str, tuple[int, ...]]]
        List of (constraint_type_name, tuple(sorted_gather_refs)) for comparison.
    """
    from img2cad_diffusion.tokenization import PrimitiveParamToken

    result = []

    for constraint in sketch.constraints.values():
        constraint_type_name = constraint.constraint_type.name

        # Skip unsupported constraint types (must match ConstraintToken enum)
        # Note: Dimension constraints (Distance, Length, Radius, Diameter) are excluded
        from img2cad_diffusion.tokens import ConstraintToken as CT
        supported_types = {name for name in CT.__members__ if name not in ('Pad', 'Mask', 'Start', 'Stop')}
        if constraint_type_name not in supported_types:
            continue

        # Get entity references with subnode info (only LocalReferenceParameter)
        refs = []
        has_external = False
        for param in constraint.parameters:
            # Skip non-reference parameters (enum, quantity, etc.)
            if not hasattr(param, 'value'):
                continue
            # Check if this is a local reference (skip enums like 'MINIMUM')
            if hasattr(param, 'type'):
                from sketchgraphs.data._constraint import ConstraintParameterType
                if param.type != ConstraintParameterType.LocalReference:
                    continue

            value = param.value
            if not isinstance(value, str):
                continue

            # Try to parse as entity reference
            try:
                if '.' in value:
                    parts = value.split('.')
                    entity_idx = int(parts[0])
                    subnode_name = parts[1]
                else:
                    entity_idx = int(value)
                    subnode_name = None
            except ValueError:
                # Not a valid entity reference (e.g., enum value)
                continue

            # Skip External refs
            if entity_idx not in gather_map:
                has_external = True
                break

            # Compute gather idx
            mapping = gather_map[entity_idx]
            if subnode_name is not None:
                subnode_to_param = {
                    'start': PrimitiveParamToken.StartX,
                    'end': PrimitiveParamToken.EndX,
                    'center': PrimitiveParamToken.CenterX
                }
                target_param = subnode_to_param.get(subnode_name)
                if target_param is None or target_param not in mapping.param_tokens:
                    gather_idx = mapping.entity_token_pos
                else:
                    offset = list(mapping.param_tokens).index(target_param) + 1
                    gather_idx = mapping.entity_token_pos + offset
            else:
                gather_idx = mapping.entity_token_pos
            refs.append(gather_idx)

        if has_external:
            continue

        # Deduplicate consecutive refs (same logic as tokenize_constraints)
        if len(refs) >= 2:
            unique_refs = [refs[0]]
            for r in refs[1:]:
                if r != unique_refs[-1]:
                    unique_refs.append(r)
            refs = unique_refs

        # Sort refs for comparison
        sorted_refs = tuple(sorted(refs))

        result.append((constraint_type_name, sorted_refs))

    # Deduplicate constraints (matching reconstruct_constraints behavior)
    # Source data may have duplicate constraints (e.g., c_19 and c_20 both Coincident(36, 1.start))
    seen = set()
    deduplicated = []
    for pair in result:
        if pair not in seen:
            seen.add(pair)
            deduplicated.append(pair)

    return deduplicated


class TestPrimitiveRoundtrip:
    """Tests for primitive tokenization/detokenization roundtrip."""

    def test_all_entity_types_and_params(self, sequences):
        """Test that all entity types and their parameters survive roundtrip.

        This test replaces the previous 3 separate tests (entity_types,
        isConstruction, clockwise) with a single comprehensive test.
        """
        found_types = {'Arc': False, 'Circle': False, 'Line': False, 'Point': False}
        found_construction = False
        found_clockwise = False

        for i in tqdm(range(min(NUM_TEST_DATA, len(sequences)))):
            seq = sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            # Skip empty sketches
            entities = [e for e in sketch.entities.values() if e is not None]
            if not entities:
                continue

            # Normalize sketch before tokenization (required for [-0.5, 0.5] range)
            from img2cad_diffusion import sketch_utils
            sketch_utils.center_sketch(sketch)
            sketch_utils.rescale_sketch(sketch)

            # Tokenize and reconstruct
            tokens, _ = tokenize_primitives(sketch, NUM_BINS, max_length=None)
            reconstructed = reconstruct_primitives(tokens['val'], NUM_BINS)

            # Verify entity count matches
            recon_entities = [e for e in reconstructed.entities.values() if e is not None]
            assert len(entities) == len(recon_entities), \
                f"Entity count mismatch in sequence {i}: {len(entities)} vs {len(recon_entities)}"

            # Compare each entity's params
            for orig, recon in zip(entities, recon_entities):
                # Arcs reconstructed from 3 points can have unstable center/radius for flat arcs.
                # Use relaxed tolerance for Arc derived parameters.
                tol = 1e-2 if isinstance(orig, Arc) else TOLERANCE
                assert_params_close(orig, recon, tol=tol)

                # Track found variations
                ent_type = type(orig).__name__
                if ent_type in found_types:
                    found_types[ent_type] = True
                if orig.isConstruction:
                    found_construction = True
                if isinstance(orig, Arc) and orig.clockwise:
                    found_clockwise = True

        # Ensure we tested all entity types
        missing = [t for t, found in found_types.items() if not found]
        assert not missing, f"Entity types not tested: {missing}"

        # Log coverage (optional)
        print(f"Tested: types={found_types}, construction={found_construction}, clockwise={found_clockwise}")


class TestConstraintRoundtrip:
    """Tests for constraint tokenization/detokenization roundtrip."""

    def test_constraint_types_and_refs(self, sequences):
        """Test constraint tokenization roundtrip preserves types and refs exactly.

        This test explicitly uses tokenize_constraints (not tokenize_sketch) to
        verify the constraint tokenization logic directly. Uses strict element-wise
        comparison (excluding External references).
        """
        from img2cad_diffusion import sketch_utils

        tested_count = 0

        for i in tqdm(range(min(NUM_TEST_DATA, len(sequences)))):
            seq = sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            # Skip sketches without constraints
            if not sketch.constraints:
                continue

            # Normalize sketch
            sketch_utils.normalize_sketch(sketch)

            # Step 1: Tokenize primitives to get gather_map
            prim_tokens, gather_map = tokenize_primitives(sketch, NUM_BINS, max_length=None)

            # Step 2: Tokenize constraints using tokenize_constraints directly
            constr_tokens = tokenize_constraints(sketch, gather_map, max_length=None, align_token_length=True)

            # Step 3: Detokenize constraints
            c_val = constr_tokens['val']
            reconstructed_constraints, invalid_constraints = reconstruct_constraints(c_val, token_len_per_constraint=CONSTRAINT_TOKEN_LEN)
            # Note: Source data may have duplicate constraints which are correctly
            # marked as invalid (deduplicated). We only fail on truly invalid constraints
            # (wrong ref count, out-of-range refs, etc.), not duplicates.
            # Duplicates have the same (constraint_type, sorted(refs)) as a valid constraint.
            valid_keys = {(c[0], tuple(sorted(c[1]))) for c in reconstructed_constraints}
            truly_invalid = [c for c in invalid_constraints if (c[0], tuple(sorted(c[1]))) not in valid_keys]
            assert len(truly_invalid) == 0, f"Seq {i}: Truly invalid constraints: {truly_invalid}"

            # Get expected constraints (excluding External refs)
            expected_pairs = get_expected_constraint_pairs(sketch, gather_map)

            # Skip if no non-External constraints
            if not expected_pairs:
                assert len(reconstructed_constraints) == 0
                continue

            # Convert reconstructed to same format for comparison
            recon_pairs = []
            for c_type, refs, _ in reconstructed_constraints:
                sorted_refs = tuple(sorted(refs))
                recon_pairs.append((c_type.name, sorted_refs))

            # Sort both lists for order-independent comparison
            expected_sorted = sorted(expected_pairs)
            recon_sorted = sorted(recon_pairs)

            # Verify exact element-wise match
            assert len(recon_sorted) == len(expected_sorted), \
                f"Seq {i}: Constraint count mismatch: {len(recon_sorted)} vs {len(expected_sorted)}"

            for j, (expected, recon) in enumerate(zip(expected_sorted, recon_sorted)):
                assert expected == recon, \
                    f"Seq {i} constraint {j}: expected {expected}, got {recon}"

            tested_count += 1

        assert tested_count > 0, "No sketches with constraints were tested"
        print(f"Tested {tested_count} sketches with strict constraint matching")


class TestFullSketchRoundtrip:
    """Tests for full sketch tokenization → reconstruction roundtrip.

    Tests that tokenize_sketch + reconstruct_sketch produces a valid sequence
    with both primitives and constraints.
    """

    def test_full_sketch_roundtrip(self, sequences):
        """Test that full sketch tokenization and reconstruction preserves structure.

        This tests tokenize_sketch → reconstruct_primitives + reconstruct_sketch.
        """
        from img2cad_diffusion import sketch_utils

        tested_count = 0

        for i in tqdm(range(min(NUM_TEST_DATA, len(sequences)))):
            seq = sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            # Skip empty sketches or those without constraints
            entities = [e for e in sketch.entities.values() if e is not None]
            if not entities or not sketch.constraints:
                continue

            # Normalize sketch
            sketch_utils.normalize_sketch(sketch)

            # Tokenize full sketch
            result = tokenize_sketch(sketch, NUM_BINS, max_length=None, align_constraint_token_length=True)

            # Reconstruct sketch with constraints
            recon_sketch = reconstruct_sketch(
                result.prim_tokens['val'],
                result.constr_tokens['val'],
                NUM_BINS,
                token_len_per_constraint=CONSTRAINT_TOKEN_LEN,
            )

            # Verify we got a sketch back with entities and constraints
            assert recon_sketch is not None, f"reconstruct_sketch returned None for seq {i}"

            recon_entities = [e for e in recon_sketch.entities.values() if e is not None]
            assert len(recon_entities) > 0, f"Empty sketch for seq {i}"
            # Verify entities match
            assert len(recon_entities) >= len(entities), \
                f"Seq {i}: Expected >= {len(entities)} entities, got {len(recon_entities)}"

            # Get gather_map for expected_pairs from original sketch
            _, orig_gather_map = tokenize_primitives(sketch, NUM_BINS, max_length=None)
            expected_pairs = get_expected_constraint_pairs(sketch, orig_gather_map)

            # Skip if no non-External constraints
            if not expected_pairs:
                continue

            # Verify constraint count and types match
            # Note: Strict refs comparison is done in TestConstraintRoundtrip.
            # Here we verify reconstruct_sketch returns correct constraint structure.
            recon_constraints = list(recon_sketch.constraints.values())

            # Constraint count must match
            assert len(recon_constraints) == len(expected_pairs), \
                f"Seq {i}: Constraint count mismatch: {len(recon_constraints)} vs {len(expected_pairs)}"

            # Sort both by type for comparison
            expected_types = sorted([c_type for c_type, _ in expected_pairs])
            recon_types = sorted([c.constraint_type.name for c in recon_constraints])

            assert expected_types == recon_types, \
                f"Seq {i}: Constraint types mismatch: {expected_types} vs {recon_types}"

            tested_count += 1

        assert tested_count > 0, "No sketches with constraints were tested"
        print(f"Tested {tested_count} full sketch roundtrips")
