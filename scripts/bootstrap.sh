#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=service
python_bin="${PYTHON_BIN:-}"
venv_dir="${VENV_DIR:-${project_dir}/.venv}"

while (( $# > 0 )); do
  case "$1" in
    --service)
      mode=service
      shift
      ;;
    --dev)
      mode=dev
      shift
      ;;
    --training)
      mode=training
      shift
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --venv)
      venv_dir="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bootstrap.sh [--service|--dev|--training] [--python PATH] [--venv PATH]

Environment:
  TORCH_INDEX_URL  PyTorch wheel index (default: CUDA 12.1)
  EAGLE_ROOT       Training checkout path (training mode only)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${python_bin}" ]]; then
  for candidate in python3.11 python3.12 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      python_bin="${candidate}"
      break
    fi
  done
fi
[[ -n "${python_bin}" ]] || {
  echo "Python 3.10-3.12 is required; pass --python PATH." >&2
  exit 1
}
"${python_bin}" -c \
  'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
fi
venv_python="${venv_dir}/bin/python"
"${venv_python}" -m pip install --upgrade pip setuptools wheel

torch_index_url="${TORCH_INDEX_URL-https://download.pytorch.org/whl/cu121}"
if [[ -n "${torch_index_url}" ]]; then
  "${venv_python}" -m pip install \
    --index-url "${torch_index_url}" torch==2.5.1 torchvision==0.20.1
else
  "${venv_python}" -m pip install torch==2.5.1 torchvision==0.20.1
fi

cd "${project_dir}"
case "${mode}" in
  service)
    extras=model
    ;;
  dev)
    extras=model,dev
    ;;
  training)
    extras=model,training,dev
    bash training/scripts/bootstrap_eagle.sh
    ;;
esac
"${venv_python}" -m pip install -e ".[${extras}]"

if [[ ! -e "${project_dir}/.env" ]]; then
  cp "${project_dir}/.env.example" "${project_dir}/.env"
  echo "Created ${project_dir}/.env; set the checkpoint and device before serving."
fi

set -a
source "${project_dir}/.env"
set +a
"${venv_dir}/bin/grasp-anything" doctor --skip-cuda
echo "Bootstrap complete (${mode}): ${venv_dir}"
