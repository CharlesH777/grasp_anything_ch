#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_path="${unit_dir}/grasp-anything.service"

for required in "${project_dir}/.env" "${project_dir}/.venv/bin/grasp-anything"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing deployment prerequisite: ${required}" >&2
    echo "Run: bash scripts/bootstrap.sh --service" >&2
    exit 1
  fi
done

mkdir -p "${unit_dir}"
sed "s|@PROJECT_DIR@|${project_dir}|g" \
  "${project_dir}/deploy/systemd/grasp-anything.service.in" > "${unit_path}"

systemctl --user daemon-reload
systemctl --user enable --now grasp-anything.service
printf 'Installed %s\n' "${unit_path}"
