#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/systemd/user"
unit_path="${unit_dir}/grasp-anything.service"

mkdir -p "${unit_dir}"
sed "s|@PROJECT_DIR@|${project_dir}|g" \
  "${project_dir}/deploy/systemd/grasp-anything.service.in" > "${unit_path}"

systemctl --user daemon-reload
systemctl --user enable --now grasp-anything.service
printf 'Installed %s\n' "${unit_path}"
