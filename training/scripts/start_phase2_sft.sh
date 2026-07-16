#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_file="${CONFIG_FILE:-${project_root}/training/configs/grasp_anything_realvlg_contact.remote.env}"
checkpoint="${MODEL_PATH:-/data2/zhenghengcao/grasp_anything_2d/outputs/realvlg-contact-lora/overfit/checkpoint-300}"
meta_path="${META_PATH:-/data2/zhenghengcao/grasp_anything_2d/prepared/full_contact_meta.json}"
max_steps="${MAX_STEPS:-13000}"
log_dir="${LOG_DIR:-/data2/zhenghengcao/grasp_anything_2d/logs}"
timestamp="$(date '+%Y%m%d-%H%M%S')"
log_file="${LOG_FILE:-${log_dir}/phase2-sft-${timestamp}.log}"

if [[ ! -f "${config_file}" ]]; then
  echo "Missing training config: ${config_file}" >&2
  exit 1
fi
if [[ ! -f "${checkpoint}/config.json" ]]; then
  echo "Missing Phase 1 checkpoint config: ${checkpoint}/config.json" >&2
  exit 1
fi
if [[ ! -f "${meta_path}" ]]; then
  echo "Missing Phase 2 meta: ${meta_path}" >&2
  exit 1
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  echo "Do not set RESUME_FROM_CHECKPOINT when changing from overfit64 to full SFT." >&2
  echo "MODEL_PATH loads Phase 1 weights; Phase 2 must start with fresh optimizer/dataloader state." >&2
  exit 1
fi
if pgrep -f 'locany_finetune_magi_stream.py' >/dev/null 2>&1; then
  echo "A contact training process is already running; refusing a duplicate launch." >&2
  pgrep -af 'locany_finetune_magi_stream.py' >&2 || true
  exit 1
fi

mkdir -p "${log_dir}"

echo "============================================================"
echo "Grasp Anything Phase 2: full contact SFT"
echo "Config:     ${config_file}"
echo "Model:      ${checkpoint}"
echo "Meta:       ${meta_path}"
echo "Steps:      ${max_steps}"
echo "Warmup:     ${WARMUP_RATIO:-0.03}"
echo "Losses:     token CE only (contact auxiliary losses disabled)"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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
  MODEL_PATH="${checkpoint}" \
  CONTACT_PHASE=sft \
  META_PATH="${meta_path}" \
  MAX_STEPS="${max_steps}" \
  WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  DRY_RUN="${DRY_RUN:-0}" \
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
