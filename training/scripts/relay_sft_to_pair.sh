#!/usr/bin/env bash
set -euo pipefail

# Finish a running SFT job, evaluate its last complete checkpoint, record the
# acceptance artifact, and launch pair training with a fresh optimizer state.
# This script is intentionally fail-closed: missing evaluation inputs or a
# failed gate stop the relay instead of manufacturing phase_acceptance.json.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${CONFIG_FILE:-${project_dir}/configs/grasp_anything_realvlg_contact.env}"
python_bin="${PYTHON_BIN:-python3}"
train_pid="${TRAIN_PID:-}"
sft_output_dir="${SFT_OUTPUT_DIR:-${REALVLG_OUTPUT_DIR:-}}/sft"
pair_output_base="${PAIR_OUTPUT_DIR:-${REALVLG_OUTPUT_DIR:-}}"
meta_path="${META_PATH:-${REALVLG_META_PATH:-}}"
data_root="${EVAL_DATA_ROOT:-${REALVLG_ROOT:-}}"
eval_limit="${EVAL_LIMIT:-}"
poll_seconds="${RELAY_POLL_SECONDS:-60}"
pair_steps="${PAIR_MAX_STEPS:-13000}"
pair_log="${PAIR_LOG:-${pair_output_base}/pair-relay-$(date '+%Y%m%d-%H%M%S').log}"

if [[ -z "${train_pid}" || ! "${train_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_PID must be the PID of the running SFT torchrun process." >&2
  exit 2
fi
for required in config_file sft_output_dir pair_output_base meta_path data_root; do
  if [[ -z "${!required}" ]]; then
    echo "${required} is required." >&2
    exit 2
  fi
done
if [[ ! -f "${config_file}" ]]; then
  echo "Missing config: ${config_file}" >&2
  exit 2
fi
if [[ ! -f "${meta_path}" ]]; then
  echo "Missing training meta: ${meta_path}" >&2
  exit 2
fi
if [[ ! -d "${sft_output_dir}" ]]; then
  echo "Missing SFT output directory: ${sft_output_dir}" >&2
  exit 2
fi
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RELAY_POLL_SECONDS must be a positive integer." >&2
  exit 2
fi

declare -a eval_specs=()
for split in seen similar novel; do
  var="EVAL_ANNOTATIONS_${split^^}"
  if [[ -n "${!var:-}" ]]; then
    eval_specs+=("${split}=${!var}")
  fi
done
if (( ${#eval_specs[@]} == 0 )); then
  echo "Set at least one EVAL_ANNOTATIONS_SEEN/SIMILAR/NOVEL JSONL path." >&2
  exit 2
fi
for spec in "${eval_specs[@]}"; do
  split="${spec%%=*}"
  annotation="${spec#*=}"
  if [[ ! -f "${annotation}" ]]; then
    echo "Missing ${split} evaluation annotations: ${annotation}" >&2
    exit 2
  fi
done

echo "Waiting for SFT PID ${train_pid} to exit..."
while kill -0 "${train_pid}" 2>/dev/null; do
  sleep "${poll_seconds}"
done
echo "SFT process exited; locating the newest complete checkpoint."

latest_checkpoint=""
while IFS= read -r candidate; do
  if [[ -f "${candidate}/config.json" ]] \
      && [[ -f "${candidate}/trainer_state.json" ]] \
      && [[ -f "${candidate}/grasp_contact_trainer_state.json" ]]; then
    latest_checkpoint="${candidate}"
  fi
done < <(find "${sft_output_dir}" -mindepth 1 -maxdepth 1 -type d \
  -name 'checkpoint-*' -printf '%f\t%p\n' | sort -V | cut -f2-)
if [[ -z "${latest_checkpoint}" ]]; then
  echo "No complete SFT checkpoint found in ${sft_output_dir}." >&2
  exit 1
fi

eval_dir="${pair_output_base}/relay-eval/$(basename "${latest_checkpoint}")"
mkdir -p "${eval_dir}"
declare -a metric_args=()
for spec in "${eval_specs[@]}"; do
  split="${spec%%=*}"
  annotation="${spec#*=}"
  output="${eval_dir}/${split}.predictions.jsonl"
  metrics="${eval_dir}/${split}.metrics.json"
  eval_command=(
    "${python_bin}" "${project_dir}/scripts/evaluate_realvlg_contact.py"
    --annotations "${annotation}"
    --data-root "${data_root}"
    --model-path "${latest_checkpoint}"
    --output "${output}"
    --metrics "${metrics}"
    --generation-mode fast
  )
  if [[ -n "${eval_limit}" ]]; then
    eval_command+=(--limit "${eval_limit}")
  fi
  echo "Evaluating ${split} on ${latest_checkpoint}..."
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" "${eval_command[@]}" \
    2>&1 | tee "${eval_dir}/${split}.eval.log"
  metric_args+=("${split}=${metrics}")
done

echo "Recording SFT acceptance..."
acceptance_command=(
  "${python_bin}" "${project_dir}/scripts/record_phase_acceptance.py"
  --checkpoint "${latest_checkpoint}"
  --phase sft
)
for metric_arg in "${metric_args[@]}"; do
  acceptance_command+=(--metrics "${metric_arg}")
done
"${acceptance_command[@]}"

mkdir -p "${pair_output_base}"
echo "Launching pair phase from ${latest_checkpoint}."
env \
  CONFIG_FILE="${config_file}" \
  MODEL_PATH="${latest_checkpoint}" \
  META_PATH="${meta_path}" \
  REALVLG_OUTPUT_DIR="${pair_output_base}" \
  CONTACT_PHASE=pair \
  GRASP_ONLY=1 \
  GROUNDING_MIN_REPLAY_SAMPLES=0 \
  GROUNDING_MIN_REPLAY_FRACTION=0 \
  MAX_STEPS="${pair_steps}" \
  RESUME_FROM_CHECKPOINT= \
  ALLOW_SAME_PHASE_WEIGHT_RESTART= \
  PYTHONUNBUFFERED=1 \
  bash "${project_dir}/scripts/train_realvlg_contact.sh" \
  2>&1 | tee "${pair_log}"
