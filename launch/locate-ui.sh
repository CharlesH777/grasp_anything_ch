#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────
# 兼容入口：请优先使用 grasp-anything-ui.sh
#
#   serve_ui.py 管理 API 子进程，支持运行时切换模型权重
#   Ctrl+C 一键停止全部
#
#   用法:
#     ./launch/locate-ui.sh
# ──────────────────────────────────────────────────────────

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/grasp-anything-ui.sh" "$@"
