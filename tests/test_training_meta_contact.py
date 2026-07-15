from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "scripts"
    / "validate_training_meta.py"
)
SPEC = importlib.util.spec_from_file_location("validate_training_meta", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _write_dataset(tmp_path: Path, candidate: list[int]) -> Path:
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    annotation = tmp_path / "contact.jsonl"
    row = {
        "task_type": "grasp_contact",
        "image": "image.png",
        "image_width": 640,
        "image_height": 480,
        "contact_candidates": [candidate],
        "candidate_collision_2d": [0.0],
        "candidate_outside_2d": [0.0],
        "collision_valid": True,
        "conversations": [
            {"from": "human", "value": "grasp it"},
            {"from": "gpt", "value": "<box><100><200><300><400></box>"},
        ],
    }
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "contact": {
                    "root": str(root.resolve()),
                    "annotation": str(annotation.resolve()),
                    "task_type": "grasp_contact",
                    "sampling_weight": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    return meta


def test_contact_training_meta_validation_accepts_converter_schema(
    tmp_path: Path,
) -> None:
    validator.validate_meta(_write_dataset(tmp_path, [100, 200, 300, 400]))


def test_contact_training_meta_validation_rejects_degenerate_pair(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        validator.validate_meta(_write_dataset(tmp_path, [100, 200, 100, 200]))
