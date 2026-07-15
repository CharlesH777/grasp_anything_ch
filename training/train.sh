#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_dir}"

mkdir -p logs run

unit_name="grasp-anything-training.service"
unit_file="${project_dir}/run/full_train.unit"
if systemctl --user is-active --quiet "${unit_name}"; then
  main_pid="$(systemctl --user show --property=MainPID --value "${unit_name}")"
  echo "Training is already running in ${unit_name} with PID ${main_pid}."
  exit 0
fi
systemctl --user reset-failed "${unit_name}" >/dev/null 2>&1 || true

timestamp="$(date +%Y%m%d-%H%M%S)"
launcher_log="${project_dir}/logs/full-train-${timestamp}.log"
latest_log="${project_dir}/logs/full-train-latest.log"
ln -sfn "$(basename "${launcher_log}")" "${latest_log}"

export RUN_MODE=train
export CONFIRM_TRAIN=YES
export AUTO_STOP_INFERENCE=1
export LAUNCHER="${LAUNCHER:-pytorch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export TRAIN_LORA_RANK="${TRAIN_LORA_RANK:-32}"
export TRAIN_MAX_SEQ_LENGTH="${TRAIN_MAX_SEQ_LENGTH:-3072}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export TRAIN_GRADIENT_ACCUMULATION="${TRAIN_GRADIENT_ACCUMULATION:-8}"
export PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-4}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export MINIMUM_FREE_MEMORY_MIB="${MINIMUM_FREE_MEMORY_MIB:-23000}"

export ENDLESS_TRAIN="${ENDLESS_TRAIN:-1}"
if [[ "${ENDLESS_TRAIN}" == "1" ]]; then
  export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-2147483647}"
  export LEARNING_RATE="${LEARNING_RATE:-3.85e-6}"
  export WARMUP_RATIO="${WARMUP_RATIO:-0}"
  export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
  export LOCATE_FORCE_RESUME_LR="${LOCATE_FORCE_RESUME_LR:-${LEARNING_RATE}}"
fi

printf -v service_command 'exec bash %q >> %q 2>&1' \
  "${project_dir}/scripts/train_locateanything_lora.sh" "${launcher_log}"

systemd-run --user \
  --unit="${unit_name}" \
  --collect \
  --property=KillMode=control-group \
  --setenv=HOME="${HOME}" \
  --setenv=PATH="${PATH}" \
  --setenv=RUN_MODE="${RUN_MODE}" \
  --setenv=CONFIRM_TRAIN="${CONFIRM_TRAIN}" \
  --setenv=AUTO_STOP_INFERENCE="${AUTO_STOP_INFERENCE}" \
  --setenv=LAUNCHER="${LAUNCHER}" \
  --setenv=CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --setenv=TRAIN_LORA_RANK="${TRAIN_LORA_RANK}" \
  --setenv=TRAIN_MAX_SEQ_LENGTH="${TRAIN_MAX_SEQ_LENGTH}" \
  --setenv=PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --setenv=TRAIN_GRADIENT_ACCUMULATION="${TRAIN_GRADIENT_ACCUMULATION}" \
  --setenv=PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE}" \
  --setenv=DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
  --setenv=MINIMUM_FREE_MEMORY_MIB="${MINIMUM_FREE_MEMORY_MIB}" \
  --setenv=ENDLESS_TRAIN="${ENDLESS_TRAIN}" \
  --setenv=TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-2500}" \
  --setenv=LEARNING_RATE="${LEARNING_RATE:-2e-5}" \
  --setenv=WARMUP_RATIO="${WARMUP_RATIO:-0.03}" \
  --setenv=LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}" \
  --setenv=LOCATE_FORCE_RESUME_LR="${LOCATE_FORCE_RESUME_LR:-}" \
  /bin/bash -lc "${service_command}" >/dev/null

printf '%s\n' "${unit_name}" > "${unit_file}"

sleep 2
if ! systemctl --user is-active --quiet "${unit_name}"; then
  echo "Training failed during startup. Log: ${launcher_log}" >&2
  tail -n 40 "${launcher_log}" >&2 || true
  journalctl --user-unit="${unit_name}" --no-pager -n 40 >&2 || true
  exit 1
fi

training_pid="$(systemctl --user show --property=MainPID --value "${unit_name}")"

echo "grasp_anything historical LoRA training started."
echo "PID: ${training_pid}"
echo "Service: ${unit_name}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Micro batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Packed sequence length: ${TRAIN_MAX_SEQ_LENGTH}"
echo "Gradient accumulation: ${TRAIN_GRADIENT_ACCUMULATION}"
echo "Log: ${latest_log}"
echo "Monitor: tail -f '${latest_log}'"
echo "Stop: systemctl --user stop '${unit_name}'"
