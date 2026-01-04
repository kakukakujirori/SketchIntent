"""Test script for solver_feedback module."""

import os
import sys

# Add FreeCAD Libs
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import numpy as np
from sketchgraphs.data import flat_array
import sketchgraphs.data as datalib

from img2cad_diffusion.tokenization import tokenize_primitives, tokenize_constraints, CONSTRAINT_TOKEN_LEN
from img2cad_diffusion.solver_feedback import (
    tokens_to_freecad_sketch,
    get_solver_status,
    HAS_FREECAD,
)

if not HAS_FREECAD:
    print("FreeCAD not available, exiting.")
    sys.exit(1)

# Configuration
NUM_BINS = 64

def test_roundtrip_with_solver():
    """Test: original data → tokens → FreeCAD → solver status."""
    print("Loading data...")
    seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')

    for sample_idx in [0, 1, 5, 10]:
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx}")
        print('='*60)

        seq = seq_data['sequences'][sample_idx]
        sketch = datalib.sketch_from_sequence(seq)

        # Tokenize
        prim_token_dict, gather_map = tokenize_primitives(sketch, NUM_BINS, max_length=None)
        prim_tokens = prim_token_dict['val']
        cons_token_dict = tokenize_constraints(sketch, gather_map, max_length=None, align_token_length=True, use_stop_padding=True)
        cons_tokens = cons_token_dict['val']

        print(f"Primitive tokens: {prim_tokens.shape}")
        print(f"Constraint tokens: {cons_tokens.shape}")

        # Convert to FreeCAD
        doc, fc_sketch, _, _ = tokens_to_freecad_sketch(prim_tokens, cons_tokens, NUM_BINS, CONSTRAINT_TOKEN_LEN)

        # Get solver status
        status = get_solver_status(fc_sketch)
        print(f"Solver status: {status}")

        # Close document
        import FreeCAD
        FreeCAD.closeDocument(doc.Name)

        print("  SUCCESS")


if __name__ == '__main__':
    test_roundtrip_with_solver()
