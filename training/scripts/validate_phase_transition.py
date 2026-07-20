#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONTACT_LATER_PHASES = {"sft", "pair", "geometry", "negative", "multigt"}
CONTACT_PREVIOUS_PHASE = {
    "sft": "overfit",
    "pair": "sft",
    "geometry": "pair",
    "negative": "geometry",
    "multigt": "negative",
}
GRASP_RECT_LATER_PHASES = {
    "sft",
    "pose_r0",
    "pose",
    "geometry",
    "multigt",
    "negative",
    "collision",
}
GRASP_RECT_PREVIOUS_PHASE = {
    "sft": "overfit",
    "pose_r0": "sft",
    "pose": "pose_r0",
    "geometry": "pose",
    "multigt": "geometry",
    "negative": "multigt",
    "collision": "negative",
}
REALVLG_OFFICIAL_COMMIT = "040562e0cf8f64a8c6e922d8f7e5e098bb3633c3"
FROZEN_OFFICIAL_SPLIT_COUNTS = {
    "seen": 253,
    "similar": 235,
    "novel": 164,
}
FROZEN_OFFICIAL_SPLIT_HASHES = {
    "seen": "09f71f3ebdc16e9f965dba7ee709ef4f72d4a09f5a93b556d7fca7ef19681998",
    "similar": "ab9434cdf4d97c74ae800e0bf6504d4720729fd0bc972a80d0c0025bff9a2bf6",
    "novel": "25b159c9c0379aa5c9f0ebcd6f9d91bcc8fc4eb841ab07faa99c896e173edfda",
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _meta_uses_overfit_data(meta_path: Path) -> bool:
    metadata = _load_json(meta_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"training meta must be a JSON object: {meta_path}")
    if "overfit64" in str(meta_path).lower():
        return True
    for name, dataset in metadata.items():
        if "overfit64" in str(name).lower():
            return True
        if isinstance(dataset, dict) and "overfit64" in str(
            dataset.get("annotation", "")
        ).lower():
            return True
    return False


def _checkpoint_weight_keys(path: Path) -> set[str]:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        payload = _load_json(index_path)
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint weight index is empty: {index_path}")
        for shard_name in set(weight_map.values()):
            shard_path = path / str(shard_name)
            if not shard_path.is_file() or shard_path.stat().st_size == 0:
                raise ValueError(f"checkpoint weight shard is missing: {shard_path}")
        return set(weight_map)

    safetensors_path = path / "model.safetensors"
    if safetensors_path.is_file():
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ValueError(
                "safetensors is required to inspect an unsharded checkpoint"
            ) from error
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            return set(handle.keys())
    raise ValueError(
        f"checkpoint has no inspectable model weights or weight index: {path}"
    )


def _phase_contract(task: str) -> tuple[set[str], dict[str, str], str]:
    if task == "contact":
        return CONTACT_LATER_PHASES, CONTACT_PREVIOUS_PHASE, "CONTACT_PHASE"
    if task == "grasp_rect":
        return (
            GRASP_RECT_LATER_PHASES,
            GRASP_RECT_PREVIOUS_PHASE,
            "GRASP_RECT_PHASE",
        )
    raise ValueError(f"unsupported phase-transition task: {task!r}")


def _validate_phase0_audit(path: Path) -> None:
    path = path.expanduser().resolve()
    manifest_path = path / "phase0_audit.json" if path.is_dir() else path
    manifest = _load_json(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("phase") != "phase0"
        or manifest.get("accepted") is not True
        or manifest.get("snapshot_verified") is not True
        or manifest.get("visual_review_confirmed") is not True
    ):
        raise ValueError(
            f"grasp_rect overfit requires an accepted, visually reviewed Phase 0 "
            f"manifest: {manifest_path}"
        )
    if manifest.get("realvlg_official_commit") != REALVLG_OFFICIAL_COMMIT:
        raise ValueError("Phase 0 manifest uses a different RealVLG commit")
    if manifest.get("expected_hashes") != FROZEN_OFFICIAL_SPLIT_HASHES:
        raise ValueError("Phase 0 manifest does not contain the frozen official hashes")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Phase 0 manifest has no split results")
    for split, expected_count in FROZEN_OFFICIAL_SPLIT_COUNTS.items():
        result = splits.get(split)
        if not isinstance(result, dict):
            raise ValueError(f"Phase 0 manifest is missing split {split}")
        if result.get("positive_samples") != expected_count:
            raise ValueError(f"Phase 0 {split} sample count is not frozen")
        if result.get("sample_id_sha256") != FROZEN_OFFICIAL_SPLIT_HASHES[split]:
            raise ValueError(f"Phase 0 {split} sample hash is not frozen")
    marker = manifest_path.parent / ".phase0_complete"
    try:
        marker_commit = marker.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"Phase 0 completion marker is missing: {marker}") from error
    if marker_commit != REALVLG_OFFICIAL_COMMIT:
        raise ValueError("Phase 0 completion marker uses a different RealVLG commit")


def _validate_acceptance(
    path: Path, target_phase: str, global_step: int, task: str
) -> None:
    acceptance_path = path / "phase_acceptance.json"
    acceptance = _load_json(acceptance_path)
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise ValueError(
            f"checkpoint has no accepted phase validation: {acceptance_path}"
        )
    recorded_task = acceptance.get("task")
    if recorded_task is not None and recorded_task != task:
        raise ValueError(
            f"phase acceptance belongs to task={recorded_task!r}, expected {task!r}"
        )
    _, previous_phase, phase_env = _phase_contract(task)
    expected_phase = previous_phase[target_phase]
    if acceptance.get("phase") != expected_phase:
        raise ValueError(
            f"{phase_env}={target_phase} requires an accepted {expected_phase} "
            f"checkpoint, got phase={acceptance.get('phase')!r}"
        )
    if acceptance.get("checkpoint_step") != global_step:
        raise ValueError(
            "phase acceptance checkpoint_step does not match trainer_state: "
            f"accepted={acceptance.get('checkpoint_step')}, actual={global_step}"
        )
    if expected_phase == "overfit":
        metrics = acceptance.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("overfit phase acceptance needs a metrics object")
        thresholds = {
            "format_valid_rate": 0.99,
            "coordinate_top1_accuracy": 0.95,
        }
        if task == "grasp_rect":
            thresholds.update(
                width_valid_rate=1.0,
                complete_six_slot_rate=0.99,
                miou_oracle_ratio=0.95,
            )
        for name, threshold in thresholds.items():
            value = metrics.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0.0 <= float(value) <= 1.0
                or value < threshold
            ):
                raise ValueError(
                    f"overfit acceptance {name}={value!r} is below {threshold}"
                )


def _validate_grasp_checkpoint(
    path: Path,
    target_phase: str,
    *,
    same_phase_resume: bool,
    same_phase_weight_restart: bool = False,
    task: str = "contact",
) -> None:
    _, previous_phase, phase_env = _phase_contract(task)
    if not path.is_dir():
        raise ValueError(
            f"later contact phases require a local grasp checkpoint directory: {path}"
        )
    config_path = path / "config.json"
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint config must be a JSON object: {config_path}")
    task_token_key = (
        "grasp_rect_task_token_ids"
        if task == "grasp_rect"
        else "grasp_task_token_ids"
    )
    task_token_ids = config.get(task_token_key)
    if (
        not isinstance(task_token_ids, list)
        or len(task_token_ids) != 2
        or not all(isinstance(value, int) for value in task_token_ids)
        or task_token_ids[0] == task_token_ids[1]
    ):
        checkpoint_label = "grasp" if task == "contact" else "grasp_rect"
        raise ValueError(
            f"{path} is not a {checkpoint_label} checkpoint: config.json needs "
            f"two distinct {task_token_key}"
        )
    lora_rank = config.get("use_llm_lora", 0)
    if not isinstance(lora_rank, int | float) or lora_rank <= 0:
        raise ValueError(
            f"{path} is not a trained LoRA grasp checkpoint: use_llm_lora must "
            "be positive"
        )
    weight_keys = _checkpoint_weight_keys(path)
    adapter_prefix = "grasp_rect_task" if task == "grasp_rect" else "grasp_task"
    required_adapter_keys = {
        f"{adapter_prefix}_embedding_delta",
        f"{adapter_prefix}_output_delta",
    }
    missing_adapters = [
        required
        for required in required_adapter_keys
        if not any(key.endswith(required) for key in weight_keys)
    ]
    if missing_adapters:
        raise ValueError(
            f"checkpoint is missing {task} adapter weights: {missing_adapters}"
        )
    if not any(".lora_A." in key for key in weight_keys) or not any(
        ".lora_B." in key for key in weight_keys
    ):
        raise ValueError("checkpoint is missing trained LoRA A/B weights")

    trainer_state_path = path / "trainer_state.json"
    trainer_state = _load_json(trainer_state_path)
    global_step = (
        trainer_state.get("global_step") if isinstance(trainer_state, dict) else None
    )
    if not isinstance(global_step, int) or global_step <= 0:
        raise ValueError(
            "checkpoint trainer_state has no positive global_step: "
            f"{trainer_state_path}"
        )
    state_name = (
        "grasp_rect_trainer_state.json"
        if task == "grasp_rect"
        else "grasp_contact_trainer_state.json"
    )
    state_path = path / state_name
    task_state = _load_json(state_path)
    seen_key = (
        "seen_grasp_rect_blocks" if task == "grasp_rect" else "seen_contact_blocks"
    )
    if not isinstance(task_state, dict) or not isinstance(
        task_state.get(seen_key), int
    ):
        raise ValueError(f"invalid {task} trainer state: {state_path}")
    training_phase = task_state.get("training_phase")
    expected_training_phase = (
        target_phase
        if same_phase_resume or same_phase_weight_restart
        else previous_phase[target_phase]
    )
    if same_phase_resume and training_phase != expected_training_phase:
        raise ValueError(
            f"checkpoint training_phase={training_phase!r}, expected "
            f"{expected_training_phase!r} for {phase_env}={target_phase}"
        )
    if (
        not same_phase_resume
        and not same_phase_weight_restart
        and training_phase is not None
        and training_phase != expected_training_phase
    ):
        raise ValueError(
            f"checkpoint training_phase={training_phase!r}, expected "
            f"{expected_training_phase!r} for {phase_env}={target_phase}"
        )
    if same_phase_resume:
        if not isinstance(task_state.get("data_fingerprint"), str):
            raise ValueError(
                "same-phase resume checkpoint has no dataset fingerprint; "
                "load it through MODEL_PATH with a fresh data stream"
            )
    elif same_phase_weight_restart:
        if training_phase is not None and training_phase != target_phase:
            raise ValueError(
                f"checkpoint training_phase={training_phase!r}, expected "
                f"{target_phase!r} for a same-phase weight restart"
            )
    else:
        _validate_acceptance(path, target_phase, global_step, task)


def validate_phase_transition(
    phase: str,
    model_path: Path,
    meta_path: Path,
    resume_from_checkpoint: Path | None = None,
    allow_overfit: bool = False,
    allow_same_phase_weight_restart: bool = False,
    task: str = "contact",
    phase0_audit: Path | None = None,
) -> None:
    later_phases, _, phase_env = _phase_contract(task)
    if task == "grasp_rect" and phase == "overfit":
        if phase0_audit is None:
            raise ValueError(
                "GRASP_RECT_PHASE=overfit requires PHASE0_AUDIT_PATH"
            )
        _validate_phase0_audit(phase0_audit)
    if phase not in later_phases:
        return
    if allow_same_phase_weight_restart and resume_from_checkpoint is not None:
        raise ValueError(
            "same-phase weight restart cannot be combined with exact resume"
        )
    checkpoint_path = resume_from_checkpoint or model_path
    _validate_grasp_checkpoint(
        checkpoint_path.expanduser().resolve(),
        phase,
        same_phase_resume=resume_from_checkpoint is not None,
        same_phase_weight_restart=allow_same_phase_weight_restart,
        task=task,
    )
    if not allow_overfit and _meta_uses_overfit_data(meta_path.expanduser().resolve()):
        raise ValueError(
            f"{phase_env}={phase} cannot use overfit64 data; switch META_PATH "
            "to the validated full training meta"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject unsafe RealVLG contact phase transitions."
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--task", choices=("contact", "grasp_rect"), default="contact"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--meta-path", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--phase0-audit", type=Path)
    parser.add_argument("--allow-overfit", action="store_true")
    parser.add_argument(
        "--allow-same-phase-weight-restart",
        action="store_true",
        help=(
            "Explicitly load a checkpoint from the target phase as MODEL_PATH "
            "with fresh optimizer and dataloader state."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_phase_transition(
            args.phase,
            args.model_path,
            args.meta_path,
            args.resume_from_checkpoint,
            args.allow_overfit,
            args.allow_same_phase_weight_restart,
            args.task,
            args.phase0_audit,
        )
    except ValueError as error:
        print(f"Phase transition validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
