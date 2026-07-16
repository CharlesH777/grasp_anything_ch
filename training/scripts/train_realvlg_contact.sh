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
  DATALOADER_NUM_WORKERS PACKING_BUFFER_SIZE
  RESUME_FROM_CHECKPOINT DRY_RUN CONTACT_MAX_CANDIDATES
  CONTACT_PAIR_WEIGHT CONTACT_CENTER_WEIGHT CONTACT_ANGLE_WEIGHT
  CONTACT_WIDTH_WEIGHT CONTACT_GEOMETRY_START_BLOCKS
  CONTACT_GEOMETRY_RAMP_BLOCKS CONTACT_COORD_MASS_THRESHOLD
  CONTACT_COORD_ENTROPY_THRESHOLD CONTACT_COLLISION_THRESHOLD
  CONTACT_OUTSIDE_THRESHOLD CONTACT_MIN_CONTACT_SAMPLES
  CONTACT_MIN_FULL_SAMPLES
  CONTACT_MIN_NEGATIVE_SAMPLES GROUNDING_MIN_REPLAY_SAMPLES
  CONTACT_MIN_POSITIVE_FRACTION CONTACT_MIN_NEGATIVE_FRACTION
  GROUNDING_MIN_REPLAY_FRACTION
  LLM_LORA_RANK ALLOW_OVERFIT_PHASE_TRANSITION
  CUDA_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR MAX_JOBS
  PREBUILD_DEEPSPEED_FUSED_ADAM GRASP_SMOKE_SKIP_SAVE
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
export CONTACT_PHASE="${phase}"
nproc="${NPROC_PER_NODE:-4}"
meta_path="${META_PATH:-${REALVLG_META_PATH:-}}"
llm_lora_rank="${LLM_LORA_RANK:-32}"
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"

if [[ ! "${llm_lora_rank}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LLM_LORA_RANK must be a positive integer." >&2
  exit 1
fi
if [[ ! "${nproc}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer." >&2
  exit 1
fi
if [[ ! "${dataloader_num_workers}" =~ ^[0-9]+$ ]]; then
  echo "DATALOADER_NUM_WORKERS must be a non-negative integer." >&2
  exit 1
fi

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

if [[ -n "${CUDA_HOME:-}" ]]; then
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "CUDA_HOME does not contain an executable bin/nvcc: ${CUDA_HOME}" >&2
    exit 1
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${TORCH_EXTENSIONS_DIR:-}" ]]; then
  mkdir -p "${TORCH_EXTENSIONS_DIR}"
fi
if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
  mkdir -p "${TRITON_CACHE_DIR}"
fi
"${python_bin}" "${project_dir}/scripts/validate_training_environment.py"

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
elif [[ -f "${eagle_root}/Embodied/eaglevl/train/grasp_contact.py" ]] \
    && [[ -f "${eagle_root}/Embodied/eaglevl/utils/locany/grasp_adapter_utils.py" ]] \
    && grep -q "contact_loss_enabled" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "output_loading_info=True" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "initialize_missing_task_adapters" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "enable_task_token_training" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "configure_llm_lora" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "contact_outside_threshold=model_args.contact_outside_threshold" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "assistant contact target is unsafe" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "data_fingerprint" \
      "${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py" \
    && grep -q "combine_base_and_pair_ce" \
      "${eagle_root}/Embodied/eaglevl/train/grasp_contact.py" \
    && grep -q "compute_coordinate_token_metrics" \
      "${eagle_root}/Embodied/eaglevl/model/locany/modeling_locateanything.py" \
    && grep -q "apply_grasp_task_output_delta" \
      "${eagle_root}/Embodied/eaglevl/utils/locany/modeling_locateanything.py" \
    && grep -q "contact_loss_enabled" \
      "${eagle_root}/Embodied/eaglevl/model/locany/modeling_locateanything.py"; then
  echo "Contact training code is already integrated; skipping Git patch check."
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
min_contact_samples="${CONTACT_MIN_CONTACT_SAMPLES:-1}"
min_grounding_samples=0
min_negative_samples=0
min_contact_fraction=0.0
min_grounding_fraction=0.0
min_negative_fraction=0.0
case "${phase}" in
  overfit)
    ;;
  sft)
    min_contact_samples="${CONTACT_MIN_FULL_SAMPLES:-1000}"
    min_grounding_samples="${GROUNDING_MIN_REPLAY_SAMPLES:-1}"
    min_contact_fraction="${CONTACT_MIN_POSITIVE_FRACTION:-0.70}"
    min_grounding_fraction="${GROUNDING_MIN_REPLAY_FRACTION:-0.15}"
    ;;
  pair)
    contact_loss=True
    min_contact_samples="${CONTACT_MIN_FULL_SAMPLES:-1000}"
    min_grounding_samples="${GROUNDING_MIN_REPLAY_SAMPLES:-1}"
    min_contact_fraction="${CONTACT_MIN_POSITIVE_FRACTION:-0.70}"
    min_grounding_fraction="${GROUNDING_MIN_REPLAY_FRACTION:-0.15}"
    ;;
  geometry)
    contact_loss=True
    min_contact_samples="${CONTACT_MIN_FULL_SAMPLES:-1000}"
    min_grounding_samples="${GROUNDING_MIN_REPLAY_SAMPLES:-1}"
    min_contact_fraction="${CONTACT_MIN_POSITIVE_FRACTION:-0.70}"
    min_grounding_fraction="${GROUNDING_MIN_REPLAY_FRACTION:-0.15}"
    # Geometry is introduced against one stable target.  Multi-GT hard-min is
    # a separate curriculum step and is enabled only by the multigt phase.
    active_candidates=1
    center_weight="${CONTACT_CENTER_WEIGHT:-0.25}"
    angle_weight="${CONTACT_ANGLE_WEIGHT:-0.10}"
    width_weight="${CONTACT_WIDTH_WEIGHT:-0.10}"
    ;;
  negative|multigt)
    contact_loss=True
    min_contact_samples="${CONTACT_MIN_FULL_SAMPLES:-1000}"
    min_grounding_samples="${GROUNDING_MIN_REPLAY_SAMPLES:-1}"
    min_negative_samples="${CONTACT_MIN_NEGATIVE_SAMPLES:-1}"
    min_contact_fraction="${CONTACT_MIN_POSITIVE_FRACTION:-0.70}"
    min_grounding_fraction="${GROUNDING_MIN_REPLAY_FRACTION:-0.15}"
    min_negative_fraction="${CONTACT_MIN_NEGATIVE_FRACTION:-0.01}"
    if [[ "${phase}" == "multigt" ]]; then
      active_candidates="${CONTACT_MAX_CANDIDATES:-8}"
    fi
    center_weight="${CONTACT_CENTER_WEIGHT:-0.25}"
    angle_weight="${CONTACT_ANGLE_WEIGHT:-0.10}"
    width_weight="${CONTACT_WIDTH_WEIGHT:-0.10}"
    ;;
  *)
    echo "CONTACT_PHASE must be overfit, sft, pair, geometry, negative, or multigt." >&2
    exit 1
    ;;
esac
for requirement in \
  "CONTACT_MIN_CONTACT_SAMPLES:${min_contact_samples}" \
  "GROUNDING_MIN_REPLAY_SAMPLES:${min_grounding_samples}" \
  "CONTACT_MIN_NEGATIVE_SAMPLES:${min_negative_samples}"; do
  requirement_name="${requirement%%:*}"
  requirement_value="${requirement#*:}"
  if [[ ! "${requirement_value}" =~ ^[0-9]+$ ]]; then
    echo "${requirement_name} must be a non-negative integer." >&2
    exit 1
  fi
done
worker_shards_per_rank="${dataloader_num_workers}"
if (( worker_shards_per_rank == 0 )); then
  worker_shards_per_rank=1
fi
total_data_shards=$((nproc * worker_shards_per_rank))
if (( min_contact_samples < total_data_shards )); then
  min_contact_samples="${total_data_shards}"
fi
if (( min_grounding_samples > 0 && min_grounding_samples < total_data_shards )); then
  min_grounding_samples="${total_data_shards}"
fi
if (( min_negative_samples > 0 && min_negative_samples < total_data_shards )); then
  min_negative_samples="${total_data_shards}"
fi

phase_validation=(
  "${python_bin}" "${project_dir}/scripts/validate_phase_transition.py"
  --phase "${phase}"
  --model-path "${MODEL_PATH}"
  --meta-path "${meta_path}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  phase_validation+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${ALLOW_OVERFIT_PHASE_TRANSITION:-0}" == "1" ]]; then
  phase_validation+=(--allow-overfit)
fi
"${phase_validation[@]}"

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
launch_model_path="${MODEL_PATH}"
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  launch_model_path="${RESUME_FROM_CHECKPOINT}"
fi
training_script="${eagle_root}/Embodied/eaglevl/train/locany_finetune_magi_stream.py"
deepspeed_config="${DEEPSPEED_CONFIG:-${eagle_root}/Embodied/deepspeed_configs/zero_stage2_config.json}"
save_strategy=steps
if [[ "${GRASP_SMOKE_SKIP_SAVE:-0}" == "1" ]]; then
  save_strategy=no
fi

command=(
  "${python_bin}" -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node="${nproc}"
  --master_addr="${MASTER_ADDR:-127.0.0.1}"
  --master_port="${MASTER_PORT:-29520}"
  "${training_script}"
  --model_name_or_path "${launch_model_path}"
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
  --freeze_llm True
  --freeze_mlp False
  --use_llm_lora "${llm_lora_rank}"
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
  --dataloader_num_workers "${dataloader_num_workers}"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --max_steps "${MAX_STEPS:-1000}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_strategy "${save_strategy}"
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

"${python_bin}" "${project_dir}/scripts/validate_training_meta.py" \
  "${meta_path}" \
  --collision-threshold "${CONTACT_COLLISION_THRESHOLD:-0.0}" \
  --outside-threshold "${CONTACT_OUTSIDE_THRESHOLD:-0.0}" \
  --min-contact-samples "${min_contact_samples}" \
  --min-grounding-samples "${min_grounding_samples}" \
  --min-negative-samples "${min_negative_samples}" \
  --min-contact-fraction "${min_contact_fraction}" \
  --min-grounding-fraction "${min_grounding_fraction}" \
  --min-negative-fraction "${min_negative_fraction}"
if [[ "${PREBUILD_DEEPSPEED_FUSED_ADAM:-1}" == "1" ]] && \
   [[ "${DRY_RUN:-0}" != "1" ]]; then
  echo "Prebuilding DeepSpeed fused_adam with ${CUDA_HOME:-auto-detected CUDA}..."
  "${python_bin}" -c \
    'from deepspeed.ops.op_builder import FusedAdamBuilder; FusedAdamBuilder().load(verbose=True); print("DeepSpeed fused_adam is ready.")'
fi
printf 'Launching phase=%s, GPUs=%s, candidates=%s, output=%s\n' \
  "${phase}" "${CUDA_VISIBLE_DEVICES}" "${active_candidates}" "${output_dir}"
if [[ "${phase}" =~ ^(sft|pair|geometry|negative|multigt)$ ]] && \
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
