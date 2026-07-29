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

from eaglevl.train.grasp_contact import (  # noqa: E402
    activate_task_token_adapters,
)
from eaglevl.train.locany_finetune_magi_stream import (  # noqa: E402
    StreamPackingMTPTrainer,
)
from eaglevl.utils.locany.grasp_adapter_utils import (  # noqa: E402
    apply_grasp_task_output_delta,
)


def _load_script(name: str):
    path = ROOT / "training" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase_validator = _load_script("validate_phase_transition")
acceptance = _load_script("record_phase_acceptance")


class _Embedding(torch.nn.Embedding):
    def get_input_embeddings(self):
        return self

    def get_output_embeddings(self):
        return self


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _Embedding(32, 3)
        self.lm_head = torch.nn.Linear(3, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


class _DualTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.config = SimpleNamespace(
            grasp_task_token_ids=[6, 7],
            grasp_rect_task_token_ids=[8, 9],
        )
        self.register_buffer("_grasp_task_token_ids", torch.tensor([6, 7]))
        self.register_buffer("_grasp_rect_task_token_ids", torch.tensor([8, 9]))
        self.grasp_task_embedding_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.grasp_task_output_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.grasp_rect_task_embedding_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.grasp_rect_task_output_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self._grasp_task_embedding_hook = None
        self._grasp_task_output_hook = None
        self._grasp_rect_task_embedding_hook = None
        self._grasp_rect_task_output_hook = None

    def register_grasp_task_embedding_hook(self):
        if self._grasp_task_embedding_hook is not None:
            self._grasp_task_embedding_hook.remove()
        if self._grasp_task_output_hook is not None:
            self._grasp_task_output_hook.remove()

        def add_input(_module, inputs, output):
            result = output
            for row, token_id in enumerate(self._grasp_task_token_ids):
                mask = (inputs[0] == token_id).unsqueeze(-1)
                result = torch.where(
                    mask, result + self.grasp_task_embedding_delta[row], result
                )
            return result

        def add_output(_module, inputs, output):
            return apply_grasp_task_output_delta(
                inputs[0],
                output,
                self._grasp_task_token_ids,
                self.grasp_task_output_delta,
            )

        self._grasp_task_embedding_hook = (
            self.language_model.embedding.register_forward_hook(add_input)
        )
        self._grasp_task_output_hook = (
            self.language_model.lm_head.register_forward_hook(add_output)
        )
        return 12

    def register_grasp_rect_task_embedding_hook(self):
        if self._grasp_rect_task_embedding_hook is not None:
            self._grasp_rect_task_embedding_hook.remove()
        if self._grasp_rect_task_output_hook is not None:
            self._grasp_rect_task_output_hook.remove()

        def add_input(_module, inputs, output):
            result = output
            for row, token_id in enumerate(self._grasp_rect_task_token_ids):
                mask = (inputs[0] == token_id).unsqueeze(-1)
                result = torch.where(
                    mask,
                    result + self.grasp_rect_task_embedding_delta[row],
                    result,
                )
            return result

        def add_output(_module, inputs, output):
            return apply_grasp_task_output_delta(
                inputs[0],
                output,
                self._grasp_rect_task_token_ids,
                self.grasp_rect_task_output_delta,
            )

        self._grasp_rect_task_embedding_hook = (
            self.language_model.embedding.register_forward_hook(add_input)
        )
        self._grasp_rect_task_output_hook = (
            self.language_model.lm_head.register_forward_hook(add_output)
        )
        return 12


def test_joint_adapter_activation_keeps_both_task_adapters_trainable() -> None:
    model = _DualTaskModel()
    model.language_model.embedding.weight.requires_grad = False
    model.language_model.lm_head.weight.requires_grad = False
    enabled = activate_task_token_adapters(
        model,
        [
            ("grasp_contact", [6, 7]),
            ("grasp_rect", [8, 9]),
        ],
    )

    assert enabled == 24
    for name in (
        "grasp_task_embedding_delta",
        "grasp_task_output_delta",
        "grasp_rect_task_embedding_delta",
        "grasp_rect_task_output_delta",
    ):
        assert getattr(model, name).requires_grad is True

    logits = model.language_model.lm_head(
        model.language_model.embedding(torch.tensor([[6, 8, 7, 9]]))
    )
    logits.sum().backward()
    for name in (
        "grasp_task_embedding_delta",
        "grasp_task_output_delta",
        "grasp_rect_task_embedding_delta",
        "grasp_rect_task_output_delta",
    ):
        assert getattr(model, name).grad is not None


def test_joint_trainer_mode_and_geometry_weights_include_both_tasks() -> None:
    config = SimpleNamespace(joint_task_enabled=True)
    assert StreamPackingMTPTrainer._task_mode(config) == "joint"
    assert StreamPackingMTPTrainer._geometry_weight_names("joint") == (
        "contact_center_weight",
        "contact_angle_weight",
        "contact_width_weight",
        "grasp_rect_center_weight",
        "grasp_rect_angle_weight",
        "grasp_rect_width_weight",
    )


def _write_joint_checkpoint(path: Path, *, include_rect: bool = True) -> None:
    path.mkdir()
    config = {
        "use_llm_lora": 32,
        "joint_task_enabled": True,
        "grasp_task_token_ids": [100, 101],
        "grasp_rect_task_token_ids": [102, 103],
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shard_name = "model-00001-of-00001.safetensors"
    (path / shard_name).write_bytes(b"weights")
    weight_map = {
        "model.grasp_task_embedding_delta": shard_name,
        "model.grasp_task_output_delta": shard_name,
        "model.language_model.layers.0.lora_A.default.weight": shard_name,
        "model.language_model.layers.0.lora_B.default.weight": shard_name,
    }
    if include_rect:
        weight_map.update(
            {
                "model.grasp_rect_task_embedding_delta": shard_name,
                "model.grasp_rect_task_output_delta": shard_name,
            }
        )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": 1}), encoding="utf-8"
    )
    (path / "joint_trainer_state.json").write_text(
        json.dumps(
            {
                "seen_contact_blocks": 32,
                "seen_grasp_rect_blocks": 32,
                "training_phase": "overfit",
                "task_mode": "joint",
                "data_fingerprint": "sha256:test",
            }
        ),
        encoding="utf-8",
    )
    (path / "phase_acceptance.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "task": "joint",
                "phase": "overfit",
                "checkpoint_step": 1,
                "metrics": {
                    "contact": {
                        "format_valid_rate": 1.0,
                        "coordinate_top1_accuracy": 1.0,
                    },
                    "grasp_rect": {
                        "format_valid_rate": 1.0,
                        "coordinate_top1_accuracy": 1.0,
                        "width_valid_rate": 1.0,
                        "complete_six_slot_rate": 1.0,
                        "miou_oracle_ratio": 1.0,
                    },
                    "grounding_retention_ratio": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_joint_phase_transition_requires_both_adapters(tmp_path: Path) -> None:
    meta = tmp_path / "full_meta.json"
    meta.write_text(json.dumps({"joint": {}}), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    _write_joint_checkpoint(checkpoint)

    phase_validator.validate_phase_transition(
        "sft", checkpoint, meta, task="joint"
    )

    missing = tmp_path / "missing_rect"
    _write_joint_checkpoint(missing, include_rect=False)
    with pytest.raises(ValueError, match="grasp_rect_task_embedding_delta"):
        phase_validator.validate_phase_transition(
            "sft", missing, meta, task="joint"
        )


def test_joint_acceptance_requires_grounding_retention(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_joint_checkpoint(checkpoint)
    contact_metrics = tmp_path / "contact.json"
    rect_metrics = tmp_path / "rect.json"
    grounding_metrics = tmp_path / "grounding.json"
    payload = {
        "positive_samples": 4,
        "format_valid_rate": 1.0,
        "positive_grasp_output_rate": 1.0,
        "gacc_corrected_strict": 1.0,
        "coordinate_top1_accuracy": 1.0,
        "width_valid_rate": 1.0,
        "complete_six_slot_rate": 1.0,
        "miou_oracle_ratio": 1.0,
    }
    for path in (contact_metrics, rect_metrics):
        path.write_text(json.dumps(payload), encoding="utf-8")
    grounding_metrics.write_text(
        json.dumps({"retention_ratio": 0.97}), encoding="utf-8"
    )

    report = acceptance.build_joint_acceptance(
        checkpoint,
        "overfit",
        {"overfit": contact_metrics},
        {"overfit": rect_metrics},
        grounding_metrics,
        min_format_valid_rate=0.99,
        min_positive_output_rate=0.99,
        min_gacc_strict=0.0,
    )
    assert report["accepted"] is False
    assert "grounding:retention_ratio" in report["failures"]
