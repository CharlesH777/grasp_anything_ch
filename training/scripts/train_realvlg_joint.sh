#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${CONFIG_FILE:-${project_dir}/configs/grasp_anything_realvlg_joint.env}"
if [[ ! -f "${config_file}" ]]; then
  echo "Missing config: ${config_file}" >&2
  exit 1
fi

override_names=(
  MODEL_PATH EAGLE_ROOT PYTHON_BIN META_PATH REALVLG_OUTPUT_DIR JOINT_PHASE
  NPROC_PER_NODE CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT DEEPSPEED_CONFIG
  MAX_SEQ_LENGTH PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
  MAX_STEPS LEARNING_RATE WARMUP_RATIO WEIGHT_DECAY MAX_GRAD_NORM
  LR_SCHEDULER_TYPE LOGGING_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT SEED
  DATALOADER_NUM_WORKERS PACKING_BUFFER_SIZE RESUME_FROM_CHECKPOINT DRY_RUN
  CONTACT_MAX_CANDIDATES CONTACT_PAIR_WEIGHT CONTACT_CENTER_WEIGHT
  CONTACT_ANGLE_WEIGHT CONTACT_WIDTH_WEIGHT CONTACT_GEOMETRY_START_BLOCKS
  CONTACT_GEOMETRY_RAMP_BLOCKS CONTACT_COORD_MASS_THRESHOLD
  CONTACT_COORD_ENTROPY_THRESHOLD CONTACT_MINIMUM_WIDTH_DIAGONAL
  CONTACT_COLLISION_THRESHOLD CONTACT_OUTSIDE_THRESHOLD
  GRASP_RECT_MAX_CANDIDATES GRASP_RECT_POSE_WEIGHT
  GRASP_RECT_CENTER_WEIGHT GRASP_RECT_ANGLE_WEIGHT GRASP_RECT_WIDTH_WEIGHT
  GRASP_RECT_GEOMETRY_START_BLOCKS GRASP_RECT_GEOMETRY_RAMP_BLOCKS
  GRASP_RECT_COORD_MASS_THRESHOLD GRASP_RECT_COORD_ENTROPY_THRESHOLD
  GRASP_RECT_ANGLE_RESULTANT_THRESHOLD GRASP_RECT_MINIMUM_WIDTH_DIAGONAL
  GRASP_RECT_COLLISION_THRESHOLD GRASP_RECT_OUTSIDE_THRESHOLD
  JOINT_MIN_CONTACT_SAMPLES JOINT_MIN_GRASP_RECT_SAMPLES
  JOINT_MIN_FULL_CONTACT_SAMPLES JOINT_MIN_FULL_GRASP_RECT_SAMPLES
  JOINT_MIN_CONTACT_NEGATIVE_SAMPLES JOINT_MIN_GRASP_RECT_NEGATIVE_SAMPLES
  JOINT_MIN_CONTACT_FRACTION JOINT_MIN_GRASP_RECT_FRACTION
  JOINT_MIN_CONTACT_NEGATIVE_FRACTION JOINT_MIN_GRASP_RECT_NEGATIVE_FRACTION
  GROUNDING_MIN_REPLAY_SAMPLES GROUNDING_MIN_REPLAY_FRACTION
  LLM_LORA_RANK VISION_LORA_RANK VISION_LORA_LAST_LAYERS
  ALLOW_OVERFIT_PHASE_TRANSITION ALLOW_SAME_PHASE_WEIGHT_RESTART
  CUDA_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR MAX_JOBS
  PREBUILD_DEEPSPEED_FUSED_ADAM JOINT_SMOKE_SKIP_SAVE
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
phase="${JOINT_PHASE:-overfit}"
export JOINT_PHASE="${phase}"
nproc="${NPROC_PER_NODE:-4}"
meta_path="${META_PATH:-}"
llm_lora_rank="${LLM_LORA_RANK:-32}"
vision_lora_rank="${VISION_LORA_RANK:-0}"
vision_lora_last_layers="${VISION_LORA_LAST_LAYERS:-4}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"

for integer_setting in \
  "NPROC_PER_NODE:${nproc}" \
  "LLM_LORA_RANK:${llm_lora_rank}"; do
  name="${integer_setting%%:*}"
  value="${integer_setting#*:}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer." >&2
    exit 1
  fi
done
for nonnegative_setting in \
  "VISION_LORA_RANK:${vision_lora_rank}" \
  "VISION_LORA_LAST_LAYERS:${vision_lora_last_layers}" \
  "DATALOADER_NUM_WORKERS:${dataloader_num_workers}"; do
  name="${nonnegative_setting%%:*}"
  value="${nonnegative_setting#*:}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer." >&2
    exit 1
  fi
done
if (( vision_lora_rank > 0 && vision_lora_last_layers == 0 )); then
  echo "VISION_LORA_LAST_LAYERS must be positive with Vision LoRA." >&2
  exit 1
fi
if [[ "${per_device_train_batch_size}" != "1" && \
      "${per_device_train_batch_size}" != "2" ]]; then
  echo "Packed joint training supports batch size 1 or 2." >&2
  exit 1
fi
for required in MODEL_PATH REALVLG_OUTPUT_DIR; do
  if [[ -z "${!required:-}" ]]; then
    echo "${required} must be set in ${config_file}" >&2
    exit 1
  fi
done
if [[ -z "${meta_path}" || ! -f "${meta_path}" ]]; then
  echo "META_PATH must point to an existing joint dataset meta JSON." >&2
  exit 1
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA_HOME does not contain bin/nvcc: ${CUDA_HOME}" >&2
    exit 1
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
[[ -z "${TORCH_EXTENSIONS_DIR:-}" ]] || mkdir -p "${TORCH_EXTENSIONS_DIR}"
[[ -z "${TRITON_CACHE_DIR:-}" ]] || mkdir -p "${TRITON_CACHE_DIR}"
"${python_bin}" "${project_dir}/scripts/validate_training_environment.py"
bash "${project_dir}/scripts/bootstrap_eagle.sh" \
  --eagle-root "${eagle_root}" --no-clone --task joint

contact_loss=False
rect_loss=False
angle_wrap_radius=0
contact_candidates=1
rect_candidates=1
contact_center_weight=0.0
contact_angle_weight=0.0
contact_width_weight=0.0
rect_center_weight=0.0
rect_angle_weight=0.0
rect_width_weight=0.0
rect_collision_threshold=1.0
rect_outside_threshold=1.0
min_contact_samples="${JOINT_MIN_CONTACT_SAMPLES:-1}"
min_rect_samples="${JOINT_MIN_GRASP_RECT_SAMPLES:-1}"
min_grounding_samples="${GROUNDING_MIN_REPLAY_SAMPLES:-1}"
min_contact_negative_samples=0
min_rect_negative_samples=0
min_contact_fraction=0.0
min_rect_fraction=0.0
min_grounding_fraction=0.0
min_contact_negative_fraction=0.0
min_rect_negative_fraction=0.0

set_full_data_gates() {
  min_contact_samples="${JOINT_MIN_FULL_CONTACT_SAMPLES:-1000}"
  min_rect_samples="${JOINT_MIN_FULL_GRASP_RECT_SAMPLES:-1000}"
  min_contact_fraction="${JOINT_MIN_CONTACT_FRACTION:-0.30}"
  min_rect_fraction="${JOINT_MIN_GRASP_RECT_FRACTION:-0.30}"
  min_grounding_fraction="${GROUNDING_MIN_REPLAY_FRACTION:-0.15}"
}

enable_structured_losses() {
  contact_loss=True
  rect_loss=True
}

enable_geometry_losses() {
  contact_center_weight="${CONTACT_CENTER_WEIGHT:-0.25}"
  contact_angle_weight="${CONTACT_ANGLE_WEIGHT:-0.10}"
  contact_width_weight="${CONTACT_WIDTH_WEIGHT:-0.10}"
  rect_center_weight="${GRASP_RECT_CENTER_WEIGHT:-0.25}"
  rect_angle_weight="${GRASP_RECT_ANGLE_WEIGHT:-0.10}"
  rect_width_weight="${GRASP_RECT_WIDTH_WEIGHT:-0.10}"
}

enable_negative_gates() {
  min_contact_negative_samples="${JOINT_MIN_CONTACT_NEGATIVE_SAMPLES:-1}"
  min_rect_negative_samples="${JOINT_MIN_GRASP_RECT_NEGATIVE_SAMPLES:-1}"
  min_contact_negative_fraction="${JOINT_MIN_CONTACT_NEGATIVE_FRACTION:-0.01}"
  min_rect_negative_fraction="${JOINT_MIN_GRASP_RECT_NEGATIVE_FRACTION:-0.01}"
}

case "${phase}" in
  overfit)
    ;;
  sft)
    set_full_data_gates
    ;;
  structured_r0|structured)
    set_full_data_gates
    enable_structured_losses
    [[ "${phase}" == "structured" ]] && angle_wrap_radius=1
    ;;
  geometry|negative|multigt|collision)
    set_full_data_gates
    enable_structured_losses
    enable_geometry_losses
    angle_wrap_radius=1
    if [[ "${phase}" =~ ^(negative|multigt|collision)$ ]]; then
      enable_negative_gates
    fi
    if [[ "${phase}" =~ ^(multigt|collision)$ ]]; then
      contact_candidates="${CONTACT_MAX_CANDIDATES:-8}"
      rect_candidates="${GRASP_RECT_MAX_CANDIDATES:-8}"
    fi
    if [[ "${phase}" == "collision" ]]; then
      rect_collision_threshold="${GRASP_RECT_COLLISION_THRESHOLD:-0.0}"
      rect_outside_threshold="${GRASP_RECT_OUTSIDE_THRESHOLD:-0.0}"
    fi
    ;;
  *)
    echo "Invalid JOINT_PHASE=${phase}." >&2
    exit 1
    ;;
esac

for positive_setting in \
  "CONTACT_MAX_CANDIDATES:${contact_candidates}" \
  "GRASP_RECT_MAX_CANDIDATES:${rect_candidates}"; do
  name="${positive_setting%%:*}"
  value="${positive_setting#*:}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer." >&2
    exit 1
  fi
done
for count_setting in \
  "JOINT_MIN_CONTACT_SAMPLES:${min_contact_samples}" \
  "JOINT_MIN_GRASP_RECT_SAMPLES:${min_rect_samples}" \
  "GROUNDING_MIN_REPLAY_SAMPLES:${min_grounding_samples}" \
  "JOINT_MIN_CONTACT_NEGATIVE_SAMPLES:${min_contact_negative_samples}" \
  "JOINT_MIN_GRASP_RECT_NEGATIVE_SAMPLES:${min_rect_negative_samples}"; do
  name="${count_setting%%:*}"
  value="${count_setting#*:}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer." >&2
    exit 1
  fi
done

worker_shards_per_rank="${dataloader_num_workers}"
(( worker_shards_per_rank > 0 )) || worker_shards_per_rank=1
total_data_shards=$((nproc * worker_shards_per_rank))
(( min_contact_samples >= total_data_shards )) || \
  min_contact_samples="${total_data_shards}"
(( min_rect_samples >= total_data_shards )) || \
  min_rect_samples="${total_data_shards}"
if (( min_grounding_samples > 0 && min_grounding_samples < total_data_shards )); then
  min_grounding_samples="${total_data_shards}"
fi
if (( min_contact_negative_samples > 0 && \
      min_contact_negative_samples < total_data_shards )); then
  min_contact_negative_samples="${total_data_shards}"
fi
if (( min_rect_negative_samples > 0 && \
      min_rect_negative_samples < total_data_shards )); then
  min_rect_negative_samples="${total_data_shards}"
fi

phase_validation=(
  "${python_bin}" "${project_dir}/scripts/validate_phase_transition.py"
  --task joint --phase "${phase}" --model-path "${MODEL_PATH}"
  --meta-path "${meta_path}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  phase_validation+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
[[ "${ALLOW_OVERFIT_PHASE_TRANSITION:-0}" != "1" ]] || \
  phase_validation+=(--allow-overfit)
if [[ "${ALLOW_SAME_PHASE_WEIGHT_RESTART:-0}" == "1" ]]; then
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    echo "Same-phase weight restart cannot be combined with exact resume." >&2
    exit 1
  fi
  phase_validation+=(--allow-same-phase-weight-restart)
fi
"${phase_validation[@]}"

"${python_bin}" "${project_dir}/scripts/validate_training_meta.py" \
  "${meta_path}" \
  --collision-threshold "${CONTACT_COLLISION_THRESHOLD:-0.0}" \
  --outside-threshold "${CONTACT_OUTSIDE_THRESHOLD:-0.0}" \
  --min-contact-samples "${min_contact_samples}" \
  --min-grasp-rect-samples "${min_rect_samples}" \
  --min-grounding-samples "${min_grounding_samples}" \
  --min-negative-samples "${min_contact_negative_samples}" \
  --min-grasp-rect-negative-samples "${min_rect_negative_samples}" \
  --min-contact-fraction "${min_contact_fraction}" \
  --min-grasp-rect-fraction "${min_rect_fraction}" \
  --min-grounding-fraction "${min_grounding_fraction}" \
  --min-negative-fraction "${min_contact_negative_fraction}" \
  --min-grasp-rect-negative-fraction "${min_rect_negative_fraction}" \
  --grasp-rect-minimum-width-diagonal \
    "${GRASP_RECT_MINIMUM_WIDTH_DIAGONAL:-0.0001}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export LAUNCHER=pytorch
export PYTHONPATH="${eagle_root}/Embodied${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

visible_count="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
if (( visible_count < nproc )); then
  echo "CUDA_VISIBLE_DEVICES has ${visible_count} GPUs; need ${nproc}." >&2
  exit 1
fi

output_dir="${REALVLG_OUTPUT_DIR}/${phase}"
mkdir -p "${output_dir}"
launch_model_path="${MODEL_PATH}"
[[ -z "${RESUME_FROM_CHECKPOINT:-}" ]] || \
  launch_model_path="${RESUME_FROM_CHECKPOINT}"
training_script="${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py"
deepspeed_config="${DEEPSPEED_CONFIG:-${eagle_root}/Embodied/deepspeed_configs/zero_stage2_config.json}"
save_strategy=steps
[[ "${JOINT_SMOKE_SKIP_SAVE:-0}" != "1" ]] || save_strategy=no

command=(
  "${python_bin}" -m torch.distributed.run --nnodes=1 --node_rank=0
  --nproc_per_node="${nproc}" --master_addr="${MASTER_ADDR:-127.0.0.1}"
  --master_port="${MASTER_PORT:-29540}" "${training_script}"
  --model_name_or_path "${launch_model_path}" --meta_path "${meta_path}"
  --output_dir "${output_dir}" --overwrite_output_dir False --do_train True
  --block_size 6 --joint_task_enabled True --grasp_rect_task_enabled True
  --contact_max_candidates "${contact_candidates}"
  --grasp_rect_max_candidates "${rect_candidates}"
  --contact_loss_enabled "${contact_loss}"
  --contact_pair_weight "${CONTACT_PAIR_WEIGHT:-1.0}"
  --contact_center_weight "${contact_center_weight}"
  --contact_angle_weight "${contact_angle_weight}"
  --contact_width_weight "${contact_width_weight}"
  --contact_geometry_start_blocks "${CONTACT_GEOMETRY_START_BLOCKS:-0}"
  --contact_geometry_ramp_blocks "${CONTACT_GEOMETRY_RAMP_BLOCKS:-20000}"
  --contact_coord_mass_threshold "${CONTACT_COORD_MASS_THRESHOLD:-0.35}"
  --contact_coord_entropy_threshold "${CONTACT_COORD_ENTROPY_THRESHOLD:-0.85}"
  --contact_minimum_width_diagonal "${CONTACT_MINIMUM_WIDTH_DIAGONAL:-0.0001}"
  --contact_collision_threshold "${CONTACT_COLLISION_THRESHOLD:-0.0}"
  --contact_outside_threshold "${CONTACT_OUTSIDE_THRESHOLD:-0.0}"
  --grasp_rect_loss_enabled "${rect_loss}"
  --grasp_rect_pose_weight "${GRASP_RECT_POSE_WEIGHT:-1.0}"
  --grasp_rect_center_weight "${rect_center_weight}"
  --grasp_rect_angle_weight "${rect_angle_weight}"
  --grasp_rect_width_weight "${rect_width_weight}"
  --grasp_rect_angle_wrap_radius "${angle_wrap_radius}"
  --grasp_rect_geometry_start_blocks \
    "${GRASP_RECT_GEOMETRY_START_BLOCKS:-0}"
  --grasp_rect_geometry_ramp_blocks \
    "${GRASP_RECT_GEOMETRY_RAMP_BLOCKS:-20000}"
  --grasp_rect_coord_mass_threshold \
    "${GRASP_RECT_COORD_MASS_THRESHOLD:-0.35}"
  --grasp_rect_coord_entropy_threshold \
    "${GRASP_RECT_COORD_ENTROPY_THRESHOLD:-0.85}"
  --grasp_rect_angle_resultant_threshold \
    "${GRASP_RECT_ANGLE_RESULTANT_THRESHOLD:-0.05}"
  --grasp_rect_minimum_width_diagonal \
    "${GRASP_RECT_MINIMUM_WIDTH_DIAGONAL:-0.0001}"
  --grasp_rect_collision_threshold "${rect_collision_threshold}"
  --grasp_rect_outside_threshold "${rect_outside_threshold}"
  --attn_implementation sdpa --causal_attn False --freeze_backbone True
  --freeze_llm True --freeze_mlp False --use_llm_lora "${llm_lora_rank}"
  --use_backbone_lora "${vision_lora_rank}"
  --backbone_lora_last_layers "${vision_lora_last_layers}"
  --unfreeze_lm_head False --bf16 True --tf32 True --grad_checkpoint True
  --deepspeed "${deepspeed_config}"
  --per_device_train_batch_size "${per_device_train_batch_size}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --max_seq_length "${MAX_SEQ_LENGTH:-2048}"
  --max_num_tokens_per_sample "${MAX_SEQ_LENGTH:-2048}"
  --max_num_tokens "${MAX_SEQ_LENGTH:-2048}"
  --packing_buffer_size "${PACKING_BUFFER_SIZE:-16}"
  --dataloader_num_workers "${dataloader_num_workers}"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}" --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --max_steps "${MAX_STEPS:-1000}" --logging_steps "${LOGGING_STEPS:-5}"
  --save_strategy "${save_strategy}" --save_steps "${SAVE_STEPS:-250}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --average_tokens_across_devices True --ddp_find_unused_parameters False
  --group_by_length False --report_to tensorboard
  --run_name "grasp-anything-realvlg-joint-${phase}" --use_onelogger False
  --seed "${SEED:-42}" --data_seed "${SEED:-42}"
)
[[ -z "${RESUME_FROM_CHECKPOINT:-}" ]] || \
  command+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

if [[ "${PREBUILD_DEEPSPEED_FUSED_ADAM:-1}" == "1" && \
      "${DRY_RUN:-0}" != "1" ]]; then
  "${python_bin}" -c \
    'from deepspeed.ops.op_builder import FusedAdamBuilder; FusedAdamBuilder().load(verbose=True)'
fi
printf 'Joint phase=%s GPUs=%s contact_K=%s rect_K=%s output=%s\n' \
  "${phase}" "${CUDA_VISIBLE_DEVICES}" "${contact_candidates}" \
  "${rect_candidates}" "${output_dir}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
