#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${CONFIG_FILE:-${project_dir}/configs/locateanything_voc_lora.env}"

if [[ ! -f "${config_file}" ]]; then
  echo "Missing config: ${config_file}" >&2
  exit 1
fi

set -a
source "${config_file}"
set +a

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES
export LAUNCHER="${LAUNCHER:-pytorch}"

run_mode="${RUN_MODE:-probe}"
confirm_train="${CONFIRM_TRAIN:-NO}"
auto_stop_inference="${AUTO_STOP_INFERENCE:-0}"
inference_service="${INFERENCE_SERVICE:-locate-anything.service}"
minimum_free_memory_mib="${MINIMUM_FREE_MEMORY_MIB:-22000}"
gpu_release_timeout_seconds="${GPU_RELEASE_TIMEOUT_SECONDS:-30}"
gpu_release_poll_seconds="${GPU_RELEASE_POLL_SECONDS:-1}"

required_commands=(git nvidia-smi systemctl)
for command_name in "${required_commands[@]}"; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    exit 1
  fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${EAGLE_ROOT}/Embodied" ]]; then
  echo "Official Eagle repository not found: ${EAGLE_ROOT}" >&2
  echo "Clone it with: git clone https://github.com/NVlabs/Eagle.git '${EAGLE_ROOT}'" >&2
  exit 1
fi

if [[ ! -f "${META_PATH}" ]]; then
  echo "Training metadata not found: ${META_PATH}" >&2
  echo "Prepare VOC JSONL and metadata before running training." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" "${project_dir}/scripts/validate_training_meta.py" "${META_PATH}"; then
  exit 1
fi

if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then
  echo "Local MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

case "${run_mode}" in
  probe)
    max_steps="${PROBE_MAX_STEPS:-50}"
    lora_rank="${PROBE_LORA_RANK:-16}"
    max_seq_length="${PROBE_MAX_SEQ_LENGTH:-1536}"
    gradient_accumulation="${PROBE_GRADIENT_ACCUMULATION:-8}"
    save_strategy="no"
    output_dir="${OUTPUT_ROOT}/probe"
    ;;
  train)
    max_steps="${TRAIN_MAX_STEPS:-2500}"
    lora_rank="${TRAIN_LORA_RANK:-32}"
    max_seq_length="${TRAIN_MAX_SEQ_LENGTH:-2048}"
    gradient_accumulation="${TRAIN_GRADIENT_ACCUMULATION:-16}"
    save_strategy="steps"
    output_dir="${OUTPUT_ROOT}/lora-r${lora_rank}-seq${max_seq_length}"
    ;;
  *)
    echo "RUN_MODE must be probe or train, got: ${run_mode}" >&2
    exit 1
    ;;
esac

selected_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
selected_gpu="${selected_gpu//[[:space:]]/}"
if [[ -z "${selected_gpu}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must identify at least one GPU." >&2
  exit 1
fi

query_free_memory_mib() {
  local output

  if ! output="$(nvidia-smi --id="${selected_gpu}" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1)"; then
    echo "Unable to query free memory for GPU ${selected_gpu}: ${output}" >&2
    return 1
  fi

  output="${output//[[:space:]]/}"
  if [[ ! "${output}" =~ ^[0-9]+$ ]]; then
    echo "Unable to determine free memory for GPU ${selected_gpu}: ${output}" >&2
    return 1
  fi

  printf '%s\n' "${output}"
}

if ! free_memory_mib="$(query_free_memory_mib)"; then
  exit 1
fi

service_was_active=0
restore_inference_service() {
  if [[ "${service_was_active}" == "1" ]]; then
    echo "Restarting ${inference_service}..."
    systemctl --user start "${inference_service}" || true
  fi
}
handle_interrupt() {
  exit 130
}
handle_termination() {
  exit 143
}
trap restore_inference_service EXIT
trap handle_interrupt INT
trap handle_termination TERM

wait_for_gpu_memory() {
  local deadline=$((SECONDS + gpu_release_timeout_seconds))
  local current_free_memory_mib

  while true; do
    if ! current_free_memory_mib="$(query_free_memory_mib)"; then
      return 1
    fi
    if (( current_free_memory_mib >= minimum_free_memory_mib )); then
      printf '%s\n' "${current_free_memory_mib}"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "GPU ${selected_gpu} memory did not reach ${minimum_free_memory_mib} MiB within ${gpu_release_timeout_seconds}s." >&2
      echo "Current free memory: ${current_free_memory_mib} MiB" >&2
      return 1
    fi
    echo "Waiting for GPU ${selected_gpu} memory release: ${current_free_memory_mib}/${minimum_free_memory_mib} MiB" >&2
    sleep "${gpu_release_poll_seconds}"
  done
}

if systemctl --user is-active --quiet "${inference_service}"; then
  if [[ "${confirm_train}" != "YES" ]]; then
    echo "Warning: ${inference_service} is active; confirmed training would require stopping it."
  elif [[ "${auto_stop_inference}" == "1" ]]; then
    echo "Stopping ${inference_service} to release GPU memory..."
    service_was_active=1
    if ! systemctl --user stop "${inference_service}"; then
      echo "Unable to stop ${inference_service}." >&2
      exit 1
    fi
    if ! free_memory_mib="$(wait_for_gpu_memory)"; then
      exit 1
    fi
  else
    echo "Inference service is active and occupies GPU memory: ${inference_service}" >&2
    echo "Stop it manually, or execute with AUTO_STOP_INFERENCE=1 CONFIRM_TRAIN=YES." >&2
    exit 1
  fi
fi

if [[ "${confirm_train}" == "YES" ]] && (( free_memory_mib < minimum_free_memory_mib )); then
  echo "Insufficient free GPU memory: ${free_memory_mib} MiB" >&2
  echo "Required: at least ${minimum_free_memory_mib} MiB" >&2
  nvidia-smi >&2
  exit 1
fi

training_script="${EAGLE_ROOT}/Embodied/eaglevl/train/locany_finetune_magi_stream.py"
patch_file="${project_dir}/patches/locateanything-single-gpu-sdpa.patch"

if git -C "${EAGLE_ROOT}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "SDPA compatibility patch is already applied."
elif git -C "${EAGLE_ROOT}" apply --check "${patch_file}" >/dev/null 2>&1; then
  if [[ "${confirm_train}" == "YES" ]]; then
    echo "Applying RTX 3090 Ti SDPA compatibility patch..."
    git -C "${EAGLE_ROOT}" apply "${patch_file}"
  else
    echo "SDPA compatibility patch is ready and will be applied for confirmed training."
  fi
else
  echo "Unable to apply SDPA patch cleanly. Check Eagle repository revision." >&2
  exit 1
fi

mkdir -p "${output_dir}" "${LOG_ROOT}"

export PYTHONPATH="${EAGLE_ROOT}/Embodied${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

training_command=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node=1
  --master_addr=127.0.0.1
  --master_port="${MASTER_PORT:-29520}"
  "${training_script}"
  --model_name_or_path "${MODEL_PATH}"
  --max_steps "${max_steps}"
  --output_dir "${output_dir}"
  --meta_path "${META_PATH}"
  --overwrite_output_dir False
  --block_size 6
  --attn_implementation sdpa
  --causal_attn False
  --freeze_llm True
  --freeze_mlp False
  --freeze_backbone True
  --use_llm_lora "${lora_rank}"
  --use_backbone_lora 0
  --vision_select_layer -1
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}"
  --bf16 True
  --tf32 True
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${gradient_accumulation}"
  --save_strategy "${save_strategy}"
  --save_steps "${SAVE_STEPS:-250}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --video_total_pixels 8192
  --sample_log_interval "${SAMPLE_LOG_INTERVAL:-50}"
  --packing_buffer_size "${PACKING_BUFFER_SIZE:-4}"
  --max_seq_length "${max_seq_length}"
  --max_num_tokens_per_sample "${max_seq_length}"
  --max_num_tokens "${max_seq_length}"
  --do_train True
  --grad_checkpoint True
  --group_by_length False
  --report_to tensorboard
  --run_name "locateanything-voc-${run_mode}-r${lora_rank}"
  --use_onelogger False
  --mlp_connector_layers 2
  --seed "${SEED:-42}"
  --data_seed "${SEED:-42}"
)

printf '\nTraining plan:\n'
printf '  mode: %s\n' "${run_mode}"
printf '  GPU: %s\n' "${selected_gpu}"
printf '  GPU free: %s MiB\n' "${free_memory_mib}"
printf '  model: %s\n' "${MODEL_PATH}"
printf '  metadata: %s\n' "${META_PATH}"
printf '  LoRA rank: %s\n' "${lora_rank}"
printf '  max sequence: %s\n' "${max_seq_length}"
printf '  micro batch: %s\n' "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
printf '  gradient accumulation: %s\n' "${gradient_accumulation}"
printf '  max steps: %s\n' "${max_steps}"
printf '  output: %s\n\n' "${output_dir}"
printf 'Command:\n'
printf ' %q' "${training_command[@]}"
printf '\n\n'

if [[ "${confirm_train}" != "YES" ]]; then
  echo "Dry run only. No training was started."
  echo "Run with CONFIRM_TRAIN=YES after reviewing the command."
  exit 0
fi

log_file="${LOG_ROOT}/$(date +%Y%m%d-%H%M%S)-${run_mode}.log"
echo "Starting training. Log: ${log_file}"
cd "${EAGLE_ROOT}/Embodied"
"${training_command[@]}" 2>&1 | tee "${log_file}"
