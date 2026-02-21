import itertools
import os
import random
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sketchgraphs.data as datalib
from src.sketch_handlers.sketch_utils import normalize_sketch
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.constants import ConstraintToken
from sketchgraphs.data import flat_array
from sketchgraphs.data.sequence import sketch_from_sequence


UNSUPPORTED_TYPES = set()

def renumber_primitive_ids(sk: datalib.Sketch) -> datalib.Sketch:
    """
    Renumber entity IDs to 0, 1, 2... based on entity order, to match tokenizer output.
    Returns a new Sketch object.
    """
    id_map = {}
    new_entities = {}

    # 1. Map Entities
    # We iterate over values() to preserve order
    for i, ent in enumerate(sk.entities.values()):
        old_id = ent.entityId
        new_id = str(i)
        id_map[old_id] = new_id

        # Create shallow copy and update ID
        # Entities are often classes in datalib, let's assume mutable or copyable.
        # We'll try to use copy() if available, else direct modification on seq-loaded sketch
        try:
            ent.entityId = new_id
        except:
            pass
        new_entities[new_id] = ent

    # 2. Map Constraints
    new_constraints = {}
    for cid, c in sk.constraints.items():
        new_params = []
        for p in c.parameters:
            if isinstance(p, datalib.LocalReferenceParameter):
                # p.value is the reference string
                ref = p.value
                if '.' in ref:
                    eid, sub = ref.split('.', 1)
                    if eid in id_map:
                        new_ref = f"{id_map[eid]}.{sub}"
                    else:
                        new_ref = ref
                else:
                    new_ref = id_map.get(ref, ref)

                # Create new param
                new_p = datalib.LocalReferenceParameter(p.parameterId, new_ref)
                new_params.append(new_p)
            else:
                new_params.append(p)

        # Create new constraint
        # Constraint(identifier, constraint_type, parameters)
        new_c = datalib.Constraint(c.identifier, c.constraint_type, new_params)
        new_constraints[cid] = new_c

    return datalib.Sketch(entities=new_entities, constraints=new_constraints)


def verify_constraints_tokenization(data_npy: str, limit: int = 1000):
    print("Loading SketchGraphs Dataset...", flush=True)
    try:
        seq_data = flat_array.load_dictionary_flat(data_npy)
        sequences = seq_data['sequences']
        print(f"Loaded {len(sequences)} sequences.", flush=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}", flush=True)
        return False

    # Define test configurations
    stop_padding_opts = [True, False]
    align_opts = [None, 4]
    order_opts = ["sort_by_ref_id", "random_shuffle", "original"]
    configs = list(itertools.product(stop_padding_opts, align_opts, order_opts))

    # Limit to specified number
    num_to_test = min(len(sequences), limit)

    total_assertions = 0
    failures = 0

    for (use_stop, align, order) in configs:
        print(f"Testing Config: use_stop_padding={use_stop}, align={align}, order='{order}'")

        context = SketchContext(
            max_token_length=4096,  # Large enough to prevent truncation
            use_stop_padding=use_stop,
            align_constraint_token_length_to=align,
            constraint_order=order
        )
        tokenizer = SketchTokenizer(context)

        for idx in range(num_to_test):
            seq = sequences[idx]

            try:
                # 1. Reconstruct Sketch
                sketch_orig = sketch_from_sequence(seq)
                normalize_sketch(sketch_orig)

                # NORMALIZE SKETCH IDs to match tokenizer's implicit re-ordering/re-numbering
                sketch_orig = renumber_primitive_ids(sketch_orig)

                # 2. Tokenize / Detokenize
                token_data = tokenizer.tokenize(sketch_orig)
                sketch_recon = tokenizer.detokenize(token_data)

                # 3. Compare Constraints
                orig_constrs = list(sketch_orig.constraints.values())
                recon_constrs = list(sketch_recon.constraints.values())

                # Filter original constraints that are NOT supported or skipped
                # This logic mimics what the tokenizer does in `tokenize_constraints`
                valid_orig_constrs = []
                for c in orig_constrs:
                    c_type = c.constraint_type.name
                    handler = context.get_constraint_handler(c_type)
                    if not handler:
                        if c_type not in UNSUPPORTED_TYPES:
                            print(f"[WARNING] Skip `{c_type}` because its not supported")
                            UNSUPPORTED_TYPES.add(c_type)
                        continue # Skip unsupported

                    valid_orig_constrs.append(c)

                def extract_type_and_refs(c: datalib.Constraint):
                    refs = []
                    from sketchgraphs.data._constraint import LocalReferenceParameter
                    for p in c.parameters:
                        if isinstance(p, LocalReferenceParameter):
                            refs.append(p.value) # Entity ID string
                    return (c.constraint_type.name, tuple(sorted(refs)))

                valid_orig_constrs = sorted([extract_type_and_refs(c) for c in valid_orig_constrs])
                recon_constrs = sorted([extract_type_and_refs(c) for c in recon_constrs])

                # Checking Count
                if valid_orig_constrs != recon_constrs:
                    print(f"Mismatch at index {idx}: Constraint count differs. Orig: {len(valid_orig_constrs)}, Recon: {len(recon_constrs)}")

                    # Print diff
                    valid_orig_set = set(valid_orig_constrs)
                    recon_set = set(recon_constrs)

                    only_in_orig = valid_orig_set - recon_set
                    only_in_recon = recon_set - valid_orig_set

                    if only_in_orig:
                        print(f"  Only in Orig ({len(only_in_orig)}): {list(only_in_orig)[:10]}...")
                    if only_in_recon:
                        print(f"  Only in Recon ({len(only_in_recon)}): {list(only_in_recon)[:10]}...")

                    failures += 1
                    continue

                total_assertions += 1

            except Exception as e:
                print(f"CRASH at idx {idx} with config {use_stop, align, order}: {e}")
                failures += 1
                # traceback.print_exc()

    print("\nVerification Complete.")
    print(f"Total Config+Sketches Checked: {total_assertions + failures}")
    print(f"Failures: {failures}")

    return failures == 0

if __name__ == "__main__":
    data_npy = 'data_processed/sg_filtered_unique.npy'
    if verify_constraints_tokenization(data_npy, limit=10000):
        sys.exit(0)
    else:
        sys.exit(1)
