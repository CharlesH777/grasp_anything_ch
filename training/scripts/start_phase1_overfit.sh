#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_file="${CONFIG_FILE:-${project_root}/training/configs/grasp_anything_realvlg_contact.remote.env}"
log_dir="${LOG_DIR:-/data2/zhenghengcao/grasp_anything_2d/logs}"
timestamp="$(date '+%Y%m%d-%H%M%S')"
log_file="${LOG_FILE:-${log_dir}/phase1-overfit64-${timestamp}.log}"
meta_path="/data2/zhenghengcao/grasp_anything_2d/prepared/overfit64_grasp_v2_meta.json"

if [[ ! -f "${config_file}" ]]; then
  echo "Missing training config: ${config_file}" >&2
  exit 1
fi
if [[ ! -f "${meta_path}" ]]; then
  echo "Missing Phase 1 meta: ${meta_path}" >&2
  exit 1
fi
if pgrep -f 'locany_finetune_magi_stream.py' >/dev/null 2>&1; then
  echo "A contact training process is already running; refusing a duplicate launch." >&2
  pgrep -af 'locany_finetune_magi_stream.py' >&2 || true
  exit 1
fi

mkdir -p "${log_dir}"

echo "============================================================"
echo "Grasp Anything Phase 1: overfit64"
echo "Config:     ${config_file}"
echo "Meta:       ${meta_path}"
echo "Steps:      300"
echo "Warmup:     0.03 (9 steps)"
echo "GPUs:       0,1,2,3"
echo "Log:        ${log_file}"
echo "Started:    $(date --iso-8601=seconds)"
echo "============================================================"
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader || true
echo "============================================================"

set +e
env \
  CONFIG_FILE="${config_file}" \
  CONTACT_PHASE=overfit \
  META_PATH="${meta_path}" \
  MAX_STEPS=300 \
  WARMUP_RATIO=0.03 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  PYTHONUNBUFFERED=1 \
  bash "${project_root}/training/scripts/train_realvlg_contact.sh" \
  2>&1 | tee "${log_file}"
pipeline_status=("${PIPESTATUS[@]}")
train_status=${pipeline_status[0]}
if (( pipeline_status[1] != 0 && train_status == 0 )); then
  train_status=${pipeline_status[1]}
fi
set -e

echo "============================================================" | tee -a "${log_file}"
echo "Finished: $(date --iso-8601=seconds)" | tee -a "${log_file}"
echo "Exit code: ${train_status}" | tee -a "${log_file}"
echo "Log: ${log_file}" | tee -a "${log_file}"
exit "${train_status}"
