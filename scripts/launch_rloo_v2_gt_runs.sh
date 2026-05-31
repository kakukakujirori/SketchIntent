#!/usr/bin/env bash
set -euo pipefail

# Launch the two scaffold-v2 + intent pseudo-GT v2 training runs detached.
# These need to run outside Codex's sandbox if the launching process should
# survive after the tool command exits.

REPO=${REPO:-/home/ryotaro/my_works/My-Vitruvion}
CONDA=${CONDA:-/home/ryotaro/miniforge3/condabin/conda}
GROUP=${GROUP:-pseudo_rel_v2_scaffold2_intent}
LOG_DIR=${LOG_DIR:-/tmp}

launch() {
  local gpu="$1"
  local config="$2"
  local run_name="$3"
  local log_file="$4"

  if pgrep -f "src/train.py experiment=${config}" >/dev/null; then
    echo "already running: ${config}"
    return
  fi

  echo "launching ${config} on GPU ${gpu}; log=${log_file}"
  setsid -f bash -lc \
    "cd '${REPO}'; CUDA_VISIBLE_DEVICES=${gpu} '${CONDA}' run --no-capture-output -n freecad python src/train.py experiment=${config} +logger.wandb.name=${run_name} logger.wandb.group=${GROUP}" \
    >"${log_file}" 2>&1
}

launch 0 \
  rloo_response_scaffold2_intent_v2_full_all \
  rloo_response_scaffold2_intent_v2_full_all \
  "${LOG_DIR}/rloo_v2_full_all.log"

launch 1 \
  rloo_response_scaffold2_intent_v2_no_mirror \
  rloo_response_scaffold2_intent_v2_no_mirror \
  "${LOG_DIR}/rloo_v2_no_mirror.log"

echo "launched. monitor with:"
echo "  tail -f ${LOG_DIR}/rloo_v2_full_all.log"
echo "  tail -f ${LOG_DIR}/rloo_v2_no_mirror.log"
