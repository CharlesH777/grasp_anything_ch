from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EAGLE_ROOT = ROOT / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.train.locany_finetune_magi_stream import (  # noqa: E402
    IGNORE_TOKEN_ID,
    StreamPackingMTPTrainer,
)


def _load_script(name: str):
    path = ROOT / "training" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


meta_validator = _load_script("validate_training_meta")
phase_validator = _load_script("validate_phase_transition")


class _Accelerator:
    @staticmethod
    def unwrap_model(model):
        return model


def test_global_rect_denominator_and_geometry_ramp_share_accumulation_window() -> None:
    trainer = StreamPackingMTPTrainer.__new__(StreamPackingMTPTrainer)
    trainer.args = SimpleNamespace(
        average_tokens_across_devices=True, world_size=1
    )
    trainer.accelerator = _Accelerator()
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(
            contact_geometry_start_blocks=0,
            contact_geometry_ramp_blocks=1,
            grasp_rect_geometry_start_blocks=1,
            grasp_rect_geometry_ramp_blocks=2,
        )
    )
    trainer._seen_contact_blocks = 0
    trainer._seen_grasp_rect_blocks = 2
    batch = {
        "labels": torch.tensor([[IGNORE_TOKEN_ID, 1, 2, 3, 4, 5]]),
        "sub_sample_lengths": [torch.tensor([6])],
        "contact_positive_mask": torch.tensor([False]),
        "collision_valid": torch.tensor([False]),
        "contact_task_code": torch.tensor([0]),
        "grasp_rect_positive_mask": torch.tensor([True]),
        "grasp_rect_task_code": torch.tensor([1]),
        "grasp_rect_collision_valid": torch.tensor([False]),
    }

    batches, count = trainer.get_batch_samples(
        iter((batch,)), num_batches=1, device=torch.device("cpu")
    )

    assert count.item() == 5
    assert batches[0]["global_grasp_rect_count_in_window"].item() == 1
    assert batches[0]["grasp_rect_geometry_loss_scale"].item() == 0.5
    assert trainer._seen_grasp_rect_blocks == 3
    assert trainer._last_window_counts[4].item() == 0


def _write_rect_meta(tmp_path: Path, width_token: int = 250) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True)
    annotation = tmp_path / "rect.jsonl"
    row = {
        "task_type": "grasp_rect",
        "image": "image.png",
        "image_width": 640,
        "image_height": 480,
        "gripper_depth_pixels": 40.0,
        "grasp_rect_candidates": [[500, 400, 1000, width_token]],
        "candidate_collision_2d": [None],
        "candidate_outside_2d": [0.0],
        "collision_valid": False,
        "conversations": [
            {
                "from": "gpt",
                "value": (
                    "<grasp_rect><500><400><1000>"
                    f"<{width_token}></grasp_rect>"
                ),
            }
        ],
    }
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "rect": {
                    "root": str(root),
                    "annotation": str(annotation),
                    "task_type": "grasp_rect",
                    "sampling_weight": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    return meta


def test_training_meta_accepts_rect_schema_and_strict_width(tmp_path: Path) -> None:
    meta = _write_rect_meta(tmp_path)

    summary = meta_validator.validate_meta(
        meta,
        min_grasp_rect_samples=1,
        min_grasp_rect_fraction=1.0,
        grasp_rect_minimum_width_diagonal=0.001,
    )

    assert summary["grasp_rect"] == 1


def _write_rect_checkpoint(path: Path, phase: str) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "use_llm_lora": 32,
                "grasp_rect_task_token_ids": [100, 101],
            }
        ),
        encoding="utf-8",
    )
    shard = path / "model.safetensors"
    shard.write_bytes(b"weights")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.grasp_rect_task_embedding_delta": shard.name,
                    "model.grasp_rect_task_output_delta": shard.name,
                    "model.language_model.lora_A.default.weight": shard.name,
                    "model.language_model.lora_B.default.weight": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": 1}), encoding="utf-8"
    )
    (path / "grasp_rect_trainer_state.json").write_text(
        json.dumps(
            {
                "seen_grasp_rect_blocks": 64,
                "training_phase": phase,
                "data_fingerprint": "sha256:test",
            }
        ),
        encoding="utf-8",
    )
    (path / "phase_acceptance.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "task": "grasp_rect",
                "phase": phase,
                "checkpoint_step": 1,
                "metrics": {
                    "format_valid_rate": 1.0,
                    "coordinate_top1_accuracy": 1.0,
                    "width_valid_rate": 1.0,
                    "complete_six_slot_rate": 1.0,
                    "miou_oracle_ratio": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_rect_sft_requires_accepted_rect_overfit_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_rect_checkpoint(checkpoint, "overfit")
    meta = _write_rect_meta(tmp_path / "full")

    phase_validator.validate_phase_transition(
        "sft", checkpoint, meta, task="grasp_rect"
    )


def _write_phase0_audit(path: Path) -> Path:
    path.mkdir()
    hashes = phase_validator.FROZEN_OFFICIAL_SPLIT_HASHES
    counts = phase_validator.FROZEN_OFFICIAL_SPLIT_COUNTS
    manifest = {
        "phase": "phase0",
        "accepted": True,
        "snapshot_verified": True,
        "visual_review_confirmed": True,
        "realvlg_official_commit": phase_validator.REALVLG_OFFICIAL_COMMIT,
        "expected_hashes": hashes,
        "splits": {
            split: {
                "positive_samples": counts[split],
                "sample_id_sha256": hashes[split],
            }
            for split in hashes
        },
    }
    (path / "phase0_audit.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (path / ".phase0_complete").write_text(
        phase_validator.REALVLG_OFFICIAL_COMMIT + "\n", encoding="utf-8"
    )
    return path


def test_rect_overfit_requires_accepted_phase0_audit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PHASE0_AUDIT_PATH"):
        phase_validator.validate_phase_transition(
            "overfit",
            tmp_path / "model",
            tmp_path / "meta.json",
            task="grasp_rect",
        )

    phase_validator.validate_phase_transition(
        "overfit",
        tmp_path / "model",
        tmp_path / "meta.json",
        task="grasp_rect",
        phase0_audit=_write_phase0_audit(tmp_path / "phase0"),
    )


def test_training_entrypoint_keeps_rect_curriculum_independent() -> None:
    source = (
        ROOT / "training" / "scripts" / "train_realvlg_grasp.sh"
    ).read_text(encoding="utf-8")

    assert "--grasp_rect_task_enabled True" in source
    assert "--contact_loss_enabled False" in source
    assert "pose_r0|pose" in source
    assert "--task grasp_rect" in source
    assert "PHASE0_AUDIT_PATH" in source
    assert "RESUME_FROM_CHECKPOINT" in source


def test_model_forward_calls_rect_loss_with_global_denominator() -> None:
    source = (
        EAGLE_ROOT
        / "eaglevl"
        / "model"
        / "locany"
        / "modeling_locateanything.py"
    ).read_text(encoding="utf-8")

    assert "compute_grasp_rect_auxiliary_losses(" in source
    assert "global_grasp_rect_count_in_window" in source
    assert "grasp_rect_geometry_loss_scale" in source


def test_token_resize_rehooks_both_tasks_and_activates_only_current_adapter() -> None:
    source = (
        EAGLE_ROOT / "eaglevl" / "train" / "locany_finetune_magi_stream.py"
    ).read_text(encoding="utf-8")
    resize_block = source.split(
        "model.language_model.resize_token_embeddings(len(tokenizer))", 1
    )[1].split("dist.barrier()", 1)[0]

    assert "register_all_task_token_hooks(model)" in resize_block
    assert (
        'activate_task_token_adapter(\n            model,\n            "grasp_rect"'
        in source
    )
    assert (
        'activate_task_token_adapter(\n            model,\n            "grasp_contact"'
        in source
    )
