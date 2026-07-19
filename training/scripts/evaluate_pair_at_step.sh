#!/usr/bin/env bash
set -euo pipefail

# Wait for a complete Pair checkpoint, stop training gracefully, and evaluate it.
# This script never deletes checkpoints or restarts training.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
output_root="${PAIR_OUTPUT_DIR:-${REALVLG_OUTPUT_DIR:-}}/pair"
target_step="${PAIR_EVAL_STEP:-6000}"
poll_seconds="${PAIR_EVAL_POLL_SECONDS:-60}"
train_pid="${TRAIN_PID:-}"
data_root="${EVAL_DATA_ROOT:-${REALVLG_ROOT:-}}"
eval_gpu="${EVAL_GPU:-0}"
eval_dir="${PAIR_EVAL_DIR:-${PAIR_OUTPUT_DIR:-${REALVLG_OUTPUT_DIR:-}}/pair-eval-${target_step}-$(date '+%Y%m%d-%H%M%S')}"

[[ -d "${output_root}" ]] || { echo "Missing Pair output: ${output_root}" >&2; exit 2; }
[[ "${target_step}" =~ ^[1-9][0-9]*$ ]] || { echo "PAIR_EVAL_STEP must be positive" >&2; exit 2; }
[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "PAIR_EVAL_POLL_SECONDS must be positive" >&2; exit 2; }
[[ -n "${train_pid}" && "${train_pid}" =~ ^[1-9][0-9]*$ ]] || { echo "TRAIN_PID is required" >&2; exit 2; }
[[ -d "${data_root}" ]] || { echo "Missing evaluation data root: ${data_root}" >&2; exit 2; }

declare -a eval_specs=()
for split in seen similar novel; do
  var="EVAL_ANNOTATIONS_${split^^}"
  if [[ -n "${!var:-}" ]]; then
    [[ -f "${!var}" ]] || { echo "Missing ${split} annotations: ${!var}" >&2; exit 2; }
    eval_specs+=("${split}=${!var}")
  fi
done
(( ${#eval_specs[@]} > 0 )) || { echo "Set an EVAL_ANNOTATIONS_* path" >&2; exit 2; }

checkpoint="${output_root}/checkpoint-${target_step}"
echo "Waiting for complete checkpoint: ${checkpoint}"
while [[ ! -f "${checkpoint}/config.json" \
      || ! -f "${checkpoint}/trainer_state.json" \
      || ! -f "${checkpoint}/grasp_contact_trainer_state.json" ]]; do
  sleep "${poll_seconds}"
done

# Avoid evaluating while the checkpoint state files are still being flushed.
before="$(stat -c '%s:%Y' "${checkpoint}/trainer_state.json" "${checkpoint}/grasp_contact_trainer_state.json")"
sleep 10
after="$(stat -c '%s:%Y' "${checkpoint}/trainer_state.json" "${checkpoint}/grasp_contact_trainer_state.json")"
[[ "${before}" == "${after}" ]] || { echo "Checkpoint still changing; rerun the script" >&2; exit 1; }

if kill -0 "${train_pid}" 2>/dev/null; then
  echo "Checkpoint ready; requesting graceful shutdown of training PID ${train_pid}."
  kill -INT "${train_pid}"
  for _ in {1..120}; do
    kill -0 "${train_pid}" 2>/dev/null || break
    sleep 5
  done
  kill -0 "${train_pid}" 2>/dev/null && { echo "Training did not exit" >&2; exit 1; } || true
fi

mkdir -p "${eval_dir}"
metric_args=()
for spec in "${eval_specs[@]}"; do
  split="${spec%%=*}"
  annotation="${spec#*=}"
  prediction="${eval_dir}/${split}.predictions.jsonl"
  metrics="${eval_dir}/${split}.metrics.json"
  echo "Evaluating ${split}"
  CUDA_VISIBLE_DEVICES="${eval_gpu}" "${python_bin}" \
    "${project_dir}/scripts/evaluate_realvlg_contact.py" \
    --annotations "${annotation}" --data-root "${data_root}" \
    --model-path "${checkpoint}" --output "${prediction}" \
    --metrics "${metrics}" --generation-mode fast \
    2>&1 | tee "${eval_dir}/${split}.eval.log"
  metric_args+=("${split}=${metrics}")
done

acceptance=("${python_bin}" "${project_dir}/scripts/record_phase_acceptance.py" --checkpoint "${checkpoint}" --phase pair)
for metric in "${metric_args[@]}"; do acceptance+=(--metrics "${metric}"); done
"${acceptance[@]}"
echo "Pair evaluation complete: ${eval_dir}"
