#!/usr/bin/env bash
# Orchestrate sharded, PARALLEL generation of pseudo-GT v2:
#   data_processed/sg_pseudo_rel_v2_full_all.npy   (all DETECTABLE_TYPES)
#   data_processed/sg_pseudo_rel_v2_no_mirror.npy  (--no_mirror ablation)
# from the first 167k sketches of sg_filtered_unique_valid.npy at scale 0.25.
#
# Designed to run fully DETACHED so it survives the launching session:
#   setsid bash scripts/run_pseudo_rel_v2_gen.sh < /dev/null \
#       > /tmp/gen_v2_orchestrator.log 2>&1 & disown
#
# Progress / completion:
#   tail -f /tmp/gen_v2_orchestrator.status      # human-readable milestones
#   ls /tmp/gen_v2_DONE                          # exists when ALL done
#   per-shard logs: /tmp/gen_v2_<variant>.shard<i>.log
REPO=/home/ryotaro/my_works/My-Vitruvion
cd "$REPO" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CONDA=/home/ryotaro/miniforge3/condabin/conda
RUN="$CONDA run --no-capture-output -n freecad python -u -m scripts.generate_pseudo_rel_augmented_dataset"
COMMON="--input_npy data_processed/sg_filtered_unique_valid.npy --scale 0.25 --limit 167000 --log_every 5000"
NSHARD=12
STATUS=/tmp/gen_v2_orchestrator.status

log() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

run_variant() {
  local name="$1" extra="$2" pids="" i f inputs="" sz
  log "$name: launching $NSHARD shards"
  for i in $(seq 0 $((NSHARD-1))); do
    $RUN $COMMON $extra --shard ${i}/${NSHARD} \
      --output_npy /tmp/v2_${name}.shard${i}.npy \
      > /tmp/gen_v2_${name}.shard${i}.log 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait $p; done
  log "$name: shards done, merging"
  for i in $(seq 0 $((NSHARD-1))); do
    f=/tmp/v2_${name}.shard${i}.npy
    [ -f "$f" ] && inputs="${inputs}${f},"
  done
  $RUN --merge_inputs "${inputs%,}" \
    --output_npy data_processed/sg_pseudo_rel_v2_${name}.npy \
    > /tmp/gen_v2_${name}.merge.log 2>&1
  if [ -f "data_processed/sg_pseudo_rel_v2_${name}.npy" ]; then
    sz=$(du -h "data_processed/sg_pseudo_rel_v2_${name}.npy" | cut -f1)
    log "$name: MERGED -> data_processed/sg_pseudo_rel_v2_${name}.npy ($sz)"
  else
    log "$name: MERGE FAILED (see /tmp/gen_v2_${name}.merge.log)"
  fi
}

echo "START $(date '+%F %T')  NSHARD=$NSHARD  cores=$(nproc)" > "$STATUS"
run_variant full_all "" &
A=$!
run_variant no_mirror "--no_mirror" &
B=$!
wait $A
wait $B
log "ALL DONE"
touch /tmp/gen_v2_DONE
