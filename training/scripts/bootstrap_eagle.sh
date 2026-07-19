#!/usr/bin/env bash
set -euo pipefail

training_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eagle_root="${EAGLE_ROOT:-${training_dir}/Eagle}"
revision_file="${training_dir}/EAGLE_REVISION"
patch_file="${training_dir}/patches/locateanything-grasp-contact.patch"
repository="${EAGLE_REPOSITORY:-https://github.com/NVlabs/Eagle.git}"
allow_clone=1
created_checkout=0

while (( $# > 0 )); do
  case "$1" in
    --eagle-root)
      eagle_root="$2"
      shift 2
      ;;
    --no-clone)
      allow_clone=0
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bootstrap_eagle.sh [--eagle-root PATH] [--no-clone]

Clone the pinned NVIDIA Eagle revision when needed and apply the tracked
grasp-contact patch. Existing unrelated Eagle modifications are rejected.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -s "${revision_file}" ]] || { echo "Missing ${revision_file}" >&2; exit 1; }
[[ -s "${patch_file}" ]] || { echo "Missing ${patch_file}" >&2; exit 1; }
expected_revision="$(tr -d '[:space:]' < "${revision_file}")"
[[ "${expected_revision}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Invalid Eagle revision in ${revision_file}" >&2
  exit 1
}

if [[ ! -e "${eagle_root}/.git" ]]; then
  if (( ! allow_clone )); then
    echo "Eagle checkout not found: ${eagle_root}" >&2
    exit 1
  fi
  if [[ -e "${eagle_root}" \
      && -n "$(find "${eagle_root}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to clone into non-empty directory: ${eagle_root}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${eagle_root}")"
  git clone --filter=blob:none "${repository}" "${eagle_root}"
  created_checkout=1
fi

current_revision="$(git -C "${eagle_root}" rev-parse HEAD)"
if [[ "${current_revision}" != "${expected_revision}" ]]; then
  if [[ -n "$(git -C "${eagle_root}" status --porcelain)" ]]; then
    echo "Eagle has local changes at unexpected revision ${current_revision}." >&2
    echo "Expected clean revision ${expected_revision}; refusing to overwrite it." >&2
    exit 1
  fi
  git -C "${eagle_root}" fetch origin "${expected_revision}"
  git -C "${eagle_root}" checkout --detach "${expected_revision}"
fi
if (( created_checkout )); then
  git -C "${eagle_root}" checkout --detach "${expected_revision}"
fi

if git -C "${eagle_root}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
  unexpected_paths="$(
    comm -3 \
      <(git -C "${eagle_root}" status --short | sed -E 's/^.. //' | sort) \
      <(sed -n -E 's#^diff --git a/[^ ]+ b/(.+)$#\1#p' "${patch_file}" | sort)
  )"
  if [[ -n "${unexpected_paths}" ]]; then
    echo "Eagle differs from the tracked patch file set:" >&2
    echo "${unexpected_paths}" >&2
    exit 1
  fi
  echo "Eagle grasp-contact patch is already applied (${expected_revision})."
elif git -C "${eagle_root}" apply --check "${patch_file}" >/dev/null 2>&1; then
  if [[ -n "$(git -C "${eagle_root}" status --porcelain)" ]]; then
    echo "Eagle contains unrelated local changes; refusing to apply the patch." >&2
    exit 1
  fi
  git -C "${eagle_root}" apply "${patch_file}"
  echo "Applied grasp-contact patch to Eagle ${expected_revision}."
else
  echo "Eagle does not match the pinned clean or fully patched state." >&2
  echo "Checkout: ${eagle_root}" >&2
  echo "Expected revision: ${expected_revision}" >&2
  exit 1
fi

git -C "${eagle_root}" diff --check
