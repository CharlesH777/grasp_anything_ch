#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${CONFIG_FILE:-${project_dir}/configs/grasp_anything_realvlg_contact.env}"
if [[ ! -f "${config_file}" ]]; then
  echo "Missing config: ${config_file}" >&2
  exit 1
fi

override_names=(
  MODEL_PATH EAGLE_ROOT PYTHON_BIN REALVLG_OUTPUT_DIR META_PATH
  REALVLG_META_PATH CONTACT_PHASE NPROC_PER_NODE CUDA_VISIBLE_DEVICES
  MAX_SEQ_LENGTH GRADIENT_ACCUMULATION_STEPS MAX_STEPS LEARNING_RATE
  RESUME_FROM_CHECKPOINT DRY_RUN CONTACT_MAX_CANDIDATES
  CONTACT_PAIR_WEIGHT CONTACT_CENTER_WEIGHT CONTACT_ANGLE_WEIGHT
  CONTACT_WIDTH_WEIGHT CONTACT_GEOMETRY_START_BLOCKS
  CONTACT_GEOMETRY_RAMP_BLOCKS CONTACT_COORD_MASS_THRESHOLD
  CONTACT_COORD_ENTROPY_THRESHOLD CONTACT_COLLISION_THRESHOLD
  CONTACT_OUTSIDE_THRESHOLD
)
declare -A environment_overrides=()
for name in "${override_names[@]}"; do
  if [[ -v "${name}" ]]; then
    environment_overrides["${name}"]="${!name}"
  fi
done

set -a
source "${config_file}"
set +a
for name in "${!environment_overrides[@]}"; do
  printf -v "${name}" '%s' "${environment_overrides[${name}]}"
  export "${name}"
done

python_bin="${PYTHON_BIN:-python3}"
eagle_root="${EAGLE_ROOT:-${project_dir}/Eagle}"
phase="${CONTACT_PHASE:-overfit}"
nproc="${NPROC_PER_NODE:-4}"
meta_path="${META_PATH:-${REALVLG_META_PATH:-}}"

for required in MODEL_PATH REALVLG_OUTPUT_DIR; do
  if [[ -z "${!required:-}" ]]; then
    echo "${required} must be set in ${config_file}" >&2
    exit 1
  fi
done
if [[ -z "${meta_path}" || ! -f "${meta_path}" ]]; then
  echo "META_PATH must point to an existing Eagle dataset meta JSON." >&2
  exit 1
fi
if [[ ! -d "${eagle_root}/Embodied" ]]; then
  echo "Eagle checkout not found: ${eagle_root}" >&2
  exit 1
fi

contact_patch="${project_dir}/patches/locateanything-grasp-contact.patch"
if [[ ! -f "${contact_patch}" ]]; then
  echo "Missing contact training patch: ${contact_patch}" >&2
  exit 1
fi
if git -C "${eagle_root}" apply --reverse --check "${contact_patch}" \
    >/dev/null 2>&1; then
  echo "Contact training patch is already applied."
elif git -C "${eagle_root}" apply --check "${contact_patch}" \
    >/dev/null 2>&1; then
  echo "Applying grasp_anything contact training patch..."
  git -C "${eagle_root}" apply "${contact_patch}"
else
  echo "Unable to apply contact training patch cleanly." >&2
  echo "Check the Eagle revision and local modifications: ${eagle_root}" >&2
  exit 1
fi

contact_loss=False
active_candidates=1
center_weight=0.0
angle_weight=0.0
width_weight=0.0
case "${phase}" in
  overfit|sft)
    ;;
  pair)
    contact_loss=True
    ;;
  geometry)
    contact_loss=True
    center_weight="${CONTACT_CENTER_WEIGHT:-0.25}"
    angle_weight="${CONTACT_ANGLE_WEIGHT:-0.10}"
    width_weight="${CONTACT_WIDTH_WEIGHT:-0.10}"
    ;;
  multigt)
    contact_loss=True
    active_candidates="${CONTACT_MAX_CANDIDATES:-8}"
    center_weight="${CONTACT_CENTER_WEIGHT:-0.25}"
    angle_weight="${CONTACT_ANGLE_WEIGHT:-0.10}"
    width_weight="${CONTACT_WIDTH_WEIGHT:-0.10}"
    ;;
  *)
    echo "CONTACT_PHASE must be overfit, sft, pair, geometry, or multigt." >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LAUNCHER=pytorch
export PYTHONPATH="${eagle_root}/Embodied${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

visible_count="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
if (( visible_count < nproc )); then
  echo "CUDA_VISIBLE_DEVICES exposes ${visible_count} GPUs, but NPROC_PER_NODE=${nproc}." >&2
  exit 1
fi

output_dir="${REALVLG_OUTPUT_DIR}/${phase}"
mkdir -p "${output_dir}"
training_script="${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py"
deepspeed_config="${DEEPSPEED_CONFIG:-${eagle_root}/Embodied/deepspeed_configs/zero_stage2_config.json}"

command=(
  "${python_bin}" -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node="${nproc}"
  --master_addr="${MASTER_ADDR:-127.0.0.1}"
  --master_port="${MASTER_PORT:-29520}"
  "${training_script}"
  --model_name_or_path "${MODEL_PATH}"
  --meta_path "${meta_path}"
  --output_dir "${output_dir}"
  --overwrite_output_dir False
  --do_train True
  --block_size 6
  --contact_max_candidates "${active_candidates}"
  --contact_loss_enabled "${contact_loss}"
  --contact_pair_weight "${CONTACT_PAIR_WEIGHT:-1.0}"
  --contact_center_weight "${center_weight}"
  --contact_angle_weight "${angle_weight}"
  --contact_width_weight "${width_weight}"
  --contact_geometry_start_blocks "${CONTACT_GEOMETRY_START_BLOCKS:-0}"
  --contact_geometry_ramp_blocks "${CONTACT_GEOMETRY_RAMP_BLOCKS:-20000}"
  --contact_coord_mass_threshold "${CONTACT_COORD_MASS_THRESHOLD:-0.35}"
  --contact_coord_entropy_threshold "${CONTACT_COORD_ENTROPY_THRESHOLD:-0.85}"
  --contact_collision_threshold "${CONTACT_COLLISION_THRESHOLD:-0.0}"
  --contact_outside_threshold "${CONTACT_OUTSIDE_THRESHOLD:-0.0}"
  --attn_implementation sdpa
  --causal_attn False
  --freeze_backbone True
  --freeze_llm False
  --freeze_mlp False
  --use_llm_lora 0
  --use_backbone_lora 0
  --unfreeze_lm_head False
  --bf16 True
  --tf32 True
  --grad_checkpoint True
  --deepspeed "${deepspeed_config}"
  --per_device_train_batch_size 1
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}"
  --max_num_tokens_per_sample "${MAX_SEQ_LENGTH:-2048}"
  --max_num_tokens "${MAX_SEQ_LENGTH:-2048}"
  --packing_buffer_size "${PACKING_BUFFER_SIZE:-16}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --max_steps "${MAX_STEPS:-1000}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS:-250}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --average_tokens_across_devices True
  --ddp_find_unused_parameters False
  --group_by_length False
  --report_to tensorboard
  --run_name "grasp-anything-realvlg-contact-${phase}"
  --use_onelogger False
  --seed "${SEED:-42}"
  --data_seed "${SEED:-42}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  command+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${python_bin}" "${project_dir}/scripts/validate_training_meta.py" "${meta_path}"
printf 'Launching phase=%s, GPUs=%s, candidates=%s, output=%s\n' \
  "${phase}" "${CUDA_VISIBLE_DEVICES}" "${active_candidates}" "${output_dir}"
if [[ "${phase}" =~ ^(pair|geometry|multigt)$ ]] && \
   [[ -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  printf 'Phase transition: loading MODEL_PATH=%s with fresh optimizer state.\n' \
    "${MODEL_PATH}"
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
