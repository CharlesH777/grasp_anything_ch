#!/usr/bin/env bash
set -euo pipefail

# grasp_anything UI 一键启动；Ctrl+C 同时停止 UI 和 API 子进程。

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-${PROJECT_DIR}/.venv}"
SERVE_UI="${PROJECT_DIR}/scripts/serve_ui.py"

G='\033[32m'; R='\033[31m'; C='\033[36m'; B='\033[1m'; N='\033[0m'
info(){ echo -e "${C}[launch]${N} $*"; }
fail(){ echo -e "${C}[launch]${N} ${R}$*${N}"; exit 1; }

[[ -d "${VENV}" ]] || fail "venv 不存在: ${VENV}"
[[ -f "${SERVE_UI}" ]] || fail "serve_ui.py 不存在"

source "${VENV}/bin/activate"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8001}"

info "启动 grasp_anything UI；Ctrl+C 停止全部"
python "${SERVE_UI}" \
  --port "${UI_PORT}" \
  --api-port "${API_PORT}" \
  --project-dir "${PROJECT_DIR}" \
  --env-file "${PROJECT_DIR}/.env"
