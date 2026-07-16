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
            {
                "from": "gpt",
                "value": "<grasp><100><200><300><400></grasp>",
            },
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


def test_contact_training_meta_rejects_unsafe_primary_at_active_threshold(
    tmp_path: Path,
) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    annotation_path = tmp_path / "contact.jsonl"
    row = json.loads(annotation_path.read_text(encoding="utf-8"))
    row["candidate_outside_2d"] = [0.01]
    annotation_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="primary contact candidate is unsafe"):
        validator.validate_meta(meta_path, outside_threshold=0.0)


def test_contact_training_meta_requires_negative_samples_when_requested(
    tmp_path: Path,
) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])

    with pytest.raises(ValueError, match="at least 1 are required"):
        validator.validate_meta(meta_path, min_negative_samples=1)


def test_contact_training_meta_requires_grounding_replay_when_requested(
    tmp_path: Path,
) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])

    with pytest.raises(ValueError, match="0 grounding samples"):
        validator.validate_meta(meta_path, min_grounding_samples=1)


def test_contact_training_meta_counts_and_validates_configured_task_type(
    tmp_path: Path,
) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    annotation_path = tmp_path / "contact.jsonl"
    row = json.loads(annotation_path.read_text(encoding="utf-8"))
    row.pop("task_type")
    annotation_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    counts = validator.validate_meta(meta_path)

    assert counts["grasp_contact"] == 1


def _add_grounding_dataset(meta_path: Path, *, sampling_weight: float) -> None:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    root = Path(metadata["contact"]["root"])
    annotation = meta_path.parent / "grounding.jsonl"
    annotation.write_text(
        json.dumps(
            {
                "task_type": "grounding",
                "image": "image.png",
                "conversations": [
                    {"from": "human", "value": "locate it"},
                    {"from": "gpt", "value": "<box><1><2><3><4></box>"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata["grounding"] = {
        "root": str(root),
        "annotation": str(annotation.resolve()),
        "task_type": "grounding",
        "sampling_weight": sampling_weight,
    }
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_training_meta_enforces_effective_sampling_fractions(
    tmp_path: Path,
) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["contact"]["sampling_weight"] = 0.8
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    _add_grounding_dataset(meta_path, sampling_weight=0.2)

    summary = validator.validate_meta(
        meta_path,
        min_contact_fraction=0.70,
        min_grounding_fraction=0.15,
    )

    assert summary.sampling_fractions["grasp_contact"] == pytest.approx(0.8)
    assert summary.sampling_fractions["grounding"] == pytest.approx(0.2)


def test_training_meta_rejects_negligible_replay_weight(tmp_path: Path) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    _add_grounding_dataset(meta_path, sampling_weight=0.001)

    with pytest.raises(ValueError, match="grounding with fraction"):
        validator.validate_meta(meta_path, min_grounding_fraction=0.15)


def test_training_meta_requires_explicit_weights_for_phase_gates(
    tmp_path: Path,
) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    del metadata["contact"]["sampling_weight"]
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="explicit sampling_weight"):
        validator.validate_meta(meta_path, min_contact_fraction=0.70)


def test_missing_row_task_type_defaults_to_grounding(tmp_path: Path) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    _add_grounding_dataset(meta_path, sampling_weight=0.2)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    grounding = metadata["grounding"]
    grounding.pop("task_type")
    annotation = Path(grounding["annotation"])
    row = json.loads(annotation.read_text(encoding="utf-8"))
    row.pop("task_type")
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    summary = validator.validate_meta(meta_path)

    assert summary["grounding"] == 1


def test_unknown_row_task_type_is_rejected(tmp_path: Path) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    annotation = tmp_path / "contact.jsonl"
    row = json.loads(annotation.read_text(encoding="utf-8"))
    row["task_type"] = "unknown_task"
    annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["contact"].pop("task_type")
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported task_type"):
        validator.validate_meta(meta_path)


def test_minimum_counts_use_runtime_downsampled_length(tmp_path: Path) -> None:
    meta_path = _write_dataset(tmp_path, [100, 200, 300, 400])
    annotation = tmp_path / "contact.jsonl"
    row = annotation.read_text(encoding="utf-8")
    annotation.write_text(row * 4, encoding="utf-8")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["contact"]["repeat_time"] = 0.25
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="contains 1 grasp_contact samples"):
        validator.validate_meta(meta_path, min_contact_samples=2)
