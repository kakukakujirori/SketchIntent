#!/usr/bin/env python3
"""Cluster fully-constrained sketches for representative validation sampling.

This script:
1. Loads fully-constrained sketch indices from sketchgraphs_fc.txt
2. Extracts features (primitive/constraint type histograms) for each sketch
3. Clusters sketches using K-means
4. Outputs cluster assignments to JSON for reproducible sampling

Usage:
    mamba activate freecad && python scripts/cluster_fully_constrained_in_vitruvion.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array


def load_fc_indices(fc_file: Path) -> list[int]:
    """Load fully-constrained sketch indices from file."""
    with open(fc_file, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]


def extract_sketch_features(seq) -> np.ndarray:
    """Extract feature vector from a sketch sequence.

    Features:
    - Primitive type counts (Point, Line, Circle, Arc)
    - Constraint type counts (all constraint types)
    - Total entity count
    - Total constraint count
    """
    sketch = datalib.sketch_from_sequence(seq)

    # Primitive type histogram
    prim_types = ['Point', 'Line', 'Circle', 'Arc']
    prim_counts = Counter()
    for ent in sketch.entities.values():
        prim_counts[ent.type.name] += 1

    prim_features = [prim_counts.get(t, 0) for t in prim_types]

    # Constraint type histogram
    constraint_types = [
        'Coincident', 'Concentric', 'Equal', 'Fix', 'Horizontal',
        'Midpoint', 'Normal', 'Parallel', 'Perpendicular', 'Quadrant',
        'Tangent', 'Vertical'
    ]
    constr_counts = Counter()
    for c in sketch.constraints.values():
        constr_counts[c.constraint_type.name] += 1

    constr_features = [constr_counts.get(t, 0) for t in constraint_types]

    # Aggregate counts
    total_entities = len(sketch.entities)
    total_constraints = len(sketch.constraints)

    return np.array(prim_features + constr_features + [total_entities, total_constraints], dtype=np.float32)


def compute_signature(seq) -> tuple:
    """Compute a signature for deduplication (exact match detection).

    Returns a tuple that can be used as a dict key for finding duplicates.
    """
    sketch = datalib.sketch_from_sequence(seq)

    # Sort entities by type and approximate position
    entity_sigs = []
    for ent in sorted(sketch.entities.values(), key=lambda e: e.type.name):
        if hasattr(ent, 'start_point'):
            pos = tuple(np.round(ent.start_point, 2))
        elif hasattr(ent, 'center_point'):
            pos = tuple(np.round(ent.center_point, 2))
        else:
            pos = (0, 0)
        entity_sigs.append((ent.type.name, pos))

    # Constraint signature
    constr_sigs = []
    for c in sorted(sketch.constraints.values(), key=lambda c: c.constraint_type.name):
        refs = tuple(sorted(str(p.value) for p in c.parameters if hasattr(p, 'value')))
        constr_sigs.append((c.constraint_type.name, refs))

    return (tuple(entity_sigs), tuple(constr_sigs))


def main():
    parser = argparse.ArgumentParser(description='Cluster fully-constrained sketches')
    parser.add_argument('--fc-file', type=Path,
                        default=Path('scripts/sketchgraphs_fc_search/sketchgraphs_fc.txt'),
                        help='Path to FC indices file')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data_processed/sg_filtered_unique.npy'),
                        help='Path to main dataset')
    parser.add_argument('--output', type=Path,
                        default=Path('scripts/sketchgraphs_fc_search/sketchgraphs_fc_clustered.json'),
                        help='Output JSON file')
    parser.add_argument('--n-clusters', type=int, default=100,
                        help='Number of clusters for K-means')
    parser.add_argument('--n-val-samples', type=int, default=1000,
                        help='Number of samples to select for validation')
    parser.add_argument('--n-test-samples', type=int, default=500,
                        help='Number of samples to select for test')
    args = parser.parse_args()

    print(f"Loading FC indices from {args.fc_file}...")
    fc_indices = load_fc_indices(args.fc_file)
    print(f"Found {len(fc_indices)} fully-constrained sketches")

    print(f"Loading dataset from {args.data_file}...")
    # Use flat_array to load properly (same as find_fully_constrained_in_vitruvion.py)
    seq_data = flat_array.load_dictionary_flat(str(args.data_file))
    sequences = seq_data['sequences']
    print(f"Dataset size: {len(sequences)}")

    # Extract features for all FC sketches
    print("Extracting features...")
    features = []
    valid_indices = []
    signatures = {}
    duplicate_count = 0

    for idx in tqdm(fc_indices):
        if idx >= len(sequences):
            continue
        try:
            seq = sequences[idx]

            # Check for duplicates using signature
            sig = compute_signature(seq)
            if sig in signatures:
                duplicate_count += 1
                continue
            signatures[sig] = idx

            feat = extract_sketch_features(seq)
            features.append(feat)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Failed to process index {idx}: {e}")
            continue

    print(f"Extracted features for {len(features)} unique sketches (removed {duplicate_count} duplicates)")

    features = np.array(features)

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Cluster
    print(f"Clustering into {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_normalized)

    # Analyze cluster distribution
    cluster_sizes = Counter(cluster_labels)
    print(f"Cluster size range: {min(cluster_sizes.values())} - {max(cluster_sizes.values())}")
    print(f"Mean cluster size: {np.mean(list(cluster_sizes.values())):.1f}")

    # Sample from clusters (stratified) for validation + test
    total_samples = args.n_val_samples + args.n_test_samples
    print(f"Sampling {total_samples} representative sketches (val:{args.n_val_samples}, test:{args.n_test_samples})...")
    samples_per_cluster = max(1, total_samples // args.n_clusters)
    selected_indices = []

    for cluster_id in range(args.n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.array(valid_indices)[cluster_mask]

        # Sample from this cluster
        n_to_sample = min(len(cluster_indices), samples_per_cluster)
        if n_to_sample > 0:
            sampled = np.random.choice(cluster_indices, size=n_to_sample, replace=False)
            selected_indices.extend(sampled.tolist())

    # If we need more samples, randomly pick from remaining
    remaining = total_samples - len(selected_indices)
    if remaining > 0:
        unselected = [i for i in valid_indices if i not in selected_indices]
        if unselected:
            extra = np.random.choice(unselected, size=min(remaining, len(unselected)), replace=False)
            selected_indices.extend(extra.tolist())

    # Shuffle and split into validation and test
    np.random.shuffle(selected_indices)
    val_indices = selected_indices[:args.n_val_samples]
    test_indices = selected_indices[args.n_val_samples:args.n_val_samples + args.n_test_samples]

    print(f"Selected {len(val_indices)} for validation, {len(test_indices)} for test")

    # Build output data
    selected_set = set(val_indices) | set(test_indices)
    output = {
        'metadata': {
            'total_fc_sketches': len(fc_indices),
            'unique_sketches': len(valid_indices),
            'duplicates_removed': duplicate_count,
            'n_clusters': args.n_clusters,
            'n_validation': len(val_indices),
            'n_test': len(test_indices),
        },
        'clusters': {},
        'selected_for_validation': val_indices,
        'selected_for_test': test_indices,
        'remaining_for_train': [i for i in valid_indices if i not in selected_set],
    }

    # Store cluster assignments
    for i, (idx, label) in enumerate(zip(valid_indices, cluster_labels)):
        cluster_id = int(label)
        if cluster_id not in output['clusters']:
            output['clusters'][cluster_id] = []
        output['clusters'][cluster_id].append(int(idx))

    # Convert cluster keys to strings for JSON
    output['clusters'] = {str(k): v for k, v in output['clusters'].items()}

    # Save
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print("\nSummary:")
    print(f"  Total FC sketches: {len(fc_indices)}")
    print(f"  Unique sketches: {len(valid_indices)}")
    print(f"  Duplicates removed: {duplicate_count}")
    print(f"  Selected for validation: {len(val_indices)}")
    print(f"  Selected for test: {len(test_indices)}")
    print(f"  Remaining for train: {len(output['remaining_for_train'])}")
    print(f"\nOutput saved to {args.output}")


if __name__ == '__main__':
    main()
