import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sketchgraphs.data as datalib
from src.sketch_handlers.sketch_utils import complete_constraint_values, normalize_sketch
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from sketchgraphs.data import flat_array
from sketchgraphs.data.sequence import sketch_from_sequence, NodeOp, EdgeOp


def check_entity_match(ent1: datalib.Entity, ent2: datalib.Entity) -> list[tuple[str, float]]:
    """Checks if two entities are geometrically close within tolerance."""
    def p_dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    if isinstance(ent1, datalib.Point) and isinstance(ent2, datalib.Point):
        return [("Point.Start", p_dist((ent1.x, ent1.y), (ent2.x, ent2.y)))]

    elif isinstance(ent1, datalib.Line) and isinstance(ent2, datalib.Line):
        start_d = p_dist(ent1.start_point, ent2.start_point)
        end_d = p_dist(ent1.end_point, ent2.end_point)
        return [("Line.Start", start_d), ("Line.End", end_d)]

    elif isinstance(ent1, datalib.Circle) and isinstance(ent2, datalib.Circle):
        center_d = p_dist(ent1.center_point, ent2.center_point)
        radius_d = abs(ent1.radius - ent2.radius)
        return [("Circle.Center", center_d), ("Circle.Radius", radius_d)]

    elif isinstance(ent1, datalib.Arc) and isinstance(ent2, datalib.Arc):
        # Check start, end, mid, center, radius
        # Note: Center/Radius might be derived differently, but should be close
        start_d = p_dist(ent1.start_point, ent2.start_point)
        end_d = p_dist(ent1.end_point, ent2.end_point)
        mid_d = p_dist(ent1.mid_point, ent2.mid_point)

        # Center/Radius check (optional but good)
        # center_d = p_dist(ent1.center_point, ent2.center_point)
        # radius_d = abs(ent1.radius - ent2.radius)

        return [("Arc.Start", start_d), ("Arc.End", end_d), ("Arc.Mid", mid_d)]

    else:
        raise ValueError(f"Comparing different types: {type(ent1)=}, {type(ent2)=}")


def verify_primitives_tokenization(data_npy: str, limit: int = 1000):
    print("Loading SketchGraphs Dataset...", flush=True)
    try:
        seq_data = flat_array.load_dictionary_flat(data_npy)
        sequences = seq_data['sequences']
        print(f"Loaded {len(sequences)} sequences.", flush=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}", flush=True)
        return False

    context = SketchContext(align_constraint_token_length_to=4)
    tokenizer = SketchTokenizer(context)

    total_processed = 0
    total_passed = 0

    # Limit to specified number
    num_to_test = min(len(sequences), limit)

    for i in range(num_to_test):
        seq = sequences[i]

        # 1. Convert sequence to Sketch
        try:
            # Reconstruct Sketch
            sketch_orig = sketch_from_sequence(seq)
            normalize_sketch(sketch_orig)
            sketch_orig, _ = complete_constraint_values(sketch_orig)

            # 2. Tokenize / Detokenize
            token_data = tokenizer.tokenize(sketch_orig)
            sketch_recon, _, _ = tokenizer.detokenize(token_data)

            # 3. Compare Entities
            orig_entities = list(sketch_orig.entities.values())
            recon_entities = list(sketch_recon.entities.values())

            if len(orig_entities) != len(recon_entities):
                print(f"Mismatch at index {i}: Entity count differs. Orig: {len(orig_entities)}, Recon: {len(recon_entities)}")
                # Count as failure
                continue

            num_bins = tokenizer.context.num_quantization_bins
            tolerance = (1.0 / num_bins) * 1.5  # NOTE : MAGIC NUMBER

            match_all = True
            for idx, (ent_orig, ent_recon) in enumerate(zip(orig_entities, recon_entities)):
                # type check
                if type(ent_orig) != type(ent_recon):
                    print(f"Mismatch at index {i}, entity {idx}: Type mismatch ({type(ent_orig).__name__} vs {type(ent_recon).__name__})")
                    match_all = False
                    break

                # Position check
                deviation_list = check_entity_match(ent_orig, ent_recon)
                for name, deviation in deviation_list:
                    if deviation > tolerance:
                        print(f"Mismatch at index {i}, entity {idx} ({name}) {deviation:.4f} > {tolerance:.4f}")
                        match_all = False

            if match_all:
                total_passed += 1

        except Exception as e:
            print(f"Error at index {i}: {e}")
            raise e

    print(f"\nVerification Complete.")
    print(f"Total Passed: {total_passed} / {num_to_test} ({total_passed/num_to_test*100:.2f}%)")

    return total_passed == num_to_test


if __name__ == "__main__":
    data_npy = 'data/processed/sg_filtered_unique.npy'
    if verify_primitives_tokenization(data_npy, limit=10000):
        sys.exit(0)
    else:
        sys.exit(1)
