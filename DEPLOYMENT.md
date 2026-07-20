# Deployment

This repository contains code and configuration only. Datasets, checkpoints,
optimizer state, generated predictions, and logs are intentionally excluded
from Git. A deployment therefore needs both a clean repository checkout and a
separately transferred full checkpoint directory.

## Prerequisites

- Linux with Python 3.10, 3.11, or 3.12.
- An NVIDIA driver compatible with the selected PyTorch CUDA wheel.
- Git, enough disk space for the 3B checkpoint, and at least 16 GB GPU memory
  for the default BF16 single-model service.

## Native service

```bash
git clone https://github.com/CharlesH777/grasp_anything_ch.git
cd grasp_anything_ch
bash scripts/bootstrap.sh --service
```

Place or mount the complete checkpoint outside Git, then edit `.env`:

```bash
LOCATE_MODEL_ID=/srv/models/grasp-anything-checkpoint
LOCATE_MODEL_REVISION=
LOCATE_DEVICE=cuda
LOCATE_REQUIRE_GRASP_CHECKPOINT=1
```

Validate the machine before starting the API:

```bash
set -a
source .env
set +a
.venv/bin/grasp-anything doctor
.venv/bin/grasp-anything serve
```

`doctor` checks the supported Python range, runtime imports, CUDA, checkpoint
files, and the two distinct `grasp_task_token_ids`. It does not load the 3B
weights. `LOCATE_REQUIRE_GRASP_CHECKPOINT=1` also makes the runtime reject a
base LocateAnything checkpoint that lacks grasp task tokens.

For an independent Grasp Rect checkpoint use
`LOCATE_REQUIRE_GRASP_RECT_CHECKPOINT=1`. The preflight then requires two
distinct `grasp_rect_task_token_ids`; do not enable the Contact requirement
unless the deployed checkpoint is intentionally the separate Contact model.

Install the user service only after `doctor` passes:

```bash
make service-install
systemctl --user status grasp-anything.service
journalctl --user -u grasp-anything.service -f
```

## Docker Compose

Use separate host and container checkpoint paths in `.env`:

```bash
GRASP_CHECKPOINT_DIR=/srv/models/grasp-anything-checkpoint
DOCKER_MODEL_ID=/models/grasp-checkpoint
LOCATE_MODEL_REVISION=
LOCATE_REQUIRE_GRASP_CHECKPOINT=1
```

Then start the GPU service:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f grasp-anything
```

Compose mounts `GRASP_CHECKPOINT_DIR` read-only at `/models/grasp-checkpoint`.
The Hugging Face cache remains a separate writable volume. The application
filesystem is read-only except for `/tmp` and the model cache.

## Checkpoint transfer

The deployment checkpoint must contain at least:

```text
config.json
model.safetensors or model.safetensors.index.json plus all referenced shards
tokenizer_config.json
preprocessor_config.json
remote-code Python files referenced by config.json auto_map
```

For inference-only transfer, omit `global_step*`, optimizer shards, RNG state,
and dataloader state. Do not omit model shards or remote-code files. Run this
after every transfer:

```bash
LOCATE_MODEL_ID=/srv/models/grasp-anything-checkpoint \
LOCATE_REQUIRE_GRASP_CHECKPOINT=1 \
.venv/bin/grasp-anything doctor --skip-cuda
```

## Training checkout

Training modifies a pinned NVIDIA Eagle checkout through a tracked patch:

```bash
bash scripts/bootstrap.sh --training
```

The command checks out the exact commit stored in `training/EAGLE_REVISION`
and applies `training/patches/locateanything-grasp-contact.patch`. It refuses
an unexpected revision or unrelated local modifications. Supply all machine
paths at launch instead of committing them:

```bash
MODEL_PATH=/srv/models/LocateAnything-3B \
META_PATH=/srv/data/grasp/contact_meta.json \
REALVLG_OUTPUT_DIR=/srv/outputs/grasp-contact \
CONTACT_PHASE=overfit \
CONFIG_FILE=training/configs/grasp_anything_realvlg_contact.remote.env \
bash training/scripts/train_realvlg_contact.sh
```

The independent RealVLG Grasp Rect path uses the complete rect patch and its
own phase sequence:

```bash
bash training/scripts/bootstrap_eagle.sh --no-clone --task grasp-rect
MODEL_PATH=/srv/models/LocateAnything-3B \
META_PATH=/srv/data/grasp/grasp_rect_overfit64_meta.json \
PHASE0_AUDIT_PATH=/srv/data/grasp/grasp_rect_phase0 \
REALVLG_OUTPUT_DIR=/srv/outputs/grasp-rect \
GRASP_RECT_PHASE=overfit \
CONFIG_FILE=training/configs/grasp_anything_realvlg_grasp.env \
bash training/scripts/train_realvlg_grasp.sh
```

Run `training/scripts/audit_realvlg_grasp.py` before starting overfit. The first
run generates the audit images without accepting them; rerun with
`--confirm-visual-review` only after inspecting all 200 random and 50 boundary
images. Overfit requires that accepted Phase 0 directory through
`PHASE0_AUDIT_PATH`. Cross-phase loading requires the previous accepted Grasp
Rect checkpoint and always starts a fresh optimizer.

## Release verification

Before publishing a commit:

```bash
make lint
make test
for script in scripts/bootstrap.sh training/scripts/*.sh; do bash -n "$script"; done
bash training/scripts/bootstrap_eagle.sh --no-clone
git diff --check
git status --short
```

The Git status must not contain `.env`, datasets, checkpoints, outputs, or
machine-specific path files.
