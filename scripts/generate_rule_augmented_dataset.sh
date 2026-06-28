#!/usr/bin/env bash
# Generate the rule-augmented relational pseudo-reference dataset used for RLOO.
#
# Runs scripts/generate_pseudo_rel_augmented_dataset.py over the training set in
# NSHARDS parallel shards, then merges them into a single npy. Every geometric
# relation detected by the rule-based predicates is added as a relational
# constraint at tolerance scale SCALE (operating point 0.25).
#
# Usage:
#   bash scripts/generate_rule_augmented_dataset.sh
#
# Override defaults via env vars, e.g.:
#   NSHARDS=8 SCALE=0.25 OUTPUT_NPY=data_processed/sg_filtered_unique_augmented.npy \
#     bash scripts/generate_rule_augmented_dataset.sh
#
# Output:
#   data_processed/sg_filtered_unique_augmented.npy   (~1.67M sketches)
#
# Monitoring (each shard processes ~139k sketches; --log_every 5000 prints
# "processed/total valid=... rate/s ETA"):
#   # snapshot the latest progress line of every shard:
#   for i in $(seq 0 11); do echo -n "shard$i: "; tail -n1 /tmp/rule_augmented.shard$i.log; done
#   # follow one shard:
#   tail -f /tmp/rule_augmented.shard0.log
#   # confirm the workers are alive (expect NSHARDS of them):
#   pgrep -af generate_pseudo_rel_augmented_dataset
#   # merge log (appears after all shards finish):
#   cat /tmp/rule_augmented.merge.log
set -euo pipefail

NSHARDS="${NSHARDS:-12}"
INPUT_NPY="${INPUT_NPY:-data_processed/sg_filtered_unique_valid.npy}"
OUTPUT_NPY="${OUTPUT_NPY:-data_processed/sg_filtered_unique_augmented.npy}"
SCALE="${SCALE:-0.25}"
TMP="${TMPDIR:-/tmp}"

# Single-thread BLAS so the parallel shards do not oversubscribe CPU cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RUN=(conda run --no-capture-output -n freecad python -u -m scripts.generate_pseudo_rel_augmented_dataset)

echo "Launching ${NSHARDS} shards over ${INPUT_NPY} (scale=${SCALE})..."
pids=()
shard_npys=()
for i in $(seq 0 $((NSHARDS - 1))); do
    shard_npy="${TMP}/rule_augmented.shard${i}.npy"
    shard_npys+=("${shard_npy}")
    "${RUN[@]}" --input_npy "${INPUT_NPY}" --scale "${SCALE}" --log_every 5000 \
        --shard "${i}/${NSHARDS}" --output_npy "${shard_npy}" \
        > "${TMP}/rule_augmented.shard${i}.log" 2>&1 &
    pids+=("$!")
done

# Wait for every shard; collect failures instead of aborting on the first.
fail=0
for p in "${pids[@]}"; do
    wait "${p}" || fail=1
done
if [ "${fail}" -ne 0 ]; then
    echo "ERROR: at least one shard failed. Inspect ${TMP}/rule_augmented.shard*.log" >&2
    exit 1
fi

echo "All shards finished. Merging into ${OUTPUT_NPY}..."
merge_inputs=$(IFS=,; echo "${shard_npys[*]}")
"${RUN[@]}" --merge_inputs "${merge_inputs}" --output_npy "${OUTPUT_NPY}" \
    > "${TMP}/rule_augmented.merge.log" 2>&1

echo "DONE: ${OUTPUT_NPY}"
