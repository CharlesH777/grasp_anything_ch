from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from locate_anything_service.grasp_rect_geometry import rect_to_points8

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "training" / "scripts"


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = _load_script("convert_realvlg_grasp")
evaluator = _load_script("evaluate_realvlg_grasp")
audit_script = _load_script("audit_realvlg_grasp")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "realvlg"
    image_path = root / "scenes" / "scene_0100" / "kinect" / "rgb" / "0000.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image_path)
    metadata_path = root / "metadata" / "kinect" / "scene_0100" / "0000.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "object_id": 7,
                    "description": "red cup",
                    "image_path": image_path.relative_to(root).as_posix(),
                    "grasps": [
                        list(rect_to_points8(50, 50, 0, 40, 40)),
                        list(rect_to_points8(50, 50, 90, 30, 40)),
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return root, metadata_path.parent.parent


def _convert_args(root: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=root,
        metadata_dir=None,
        output=output,
        stats=output.with_suffix(".stats.json"),
        dataset_name="GraspNet_VLG",
        camera="kinect",
        split="seen",
        max_candidates=1,
        minimum_width_diagonal=1e-4,
        gripper_depth_pixels=40.0,
        official_graspnet_eval=True,
        scene_start=None,
        scene_end_exclusive=None,
    )


def test_converter_preserves_full_official_gt_when_training_k_is_one(
    tmp_path: Path,
) -> None:
    root, _ = _dataset(tmp_path)
    output = tmp_path / "grasp.jsonl"

    stats = converter.convert(_convert_args(root, output))
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["task_type"] == "grasp_rect"
    assert len(row["grasp_rect_candidates"]) == 1
    assert len(row["evaluation_grasp_rectangles_pixels"]) == 2
    primary = "".join(f"<{value}>" for value in row["grasp_rect_candidates"][0])
    assert f"<grasp_rect>{primary}</grasp_rect>" in row["conversations"][1][
        "value"
    ]
    assert stats["statistics"]["evaluation_gt_candidates"] == 2
    assert stats["protocol"]["realvlg_official_commit"] == (
        "040562e0cf8f64a8c6e922d8f7e5e098bb3633c3"
    )
    assert len(stats["protocol"]["sample_id_sha256"]) == 64
    assert stats["geometry"]["depth_pixels"]["mean"] == pytest.approx(40.0)
    assert stats["geometry"]["roundtrip_iou"]["mean"] == pytest.approx(1.0)


def test_converter_rejects_unrepresentable_width_without_clipping(
    tmp_path: Path,
) -> None:
    root, _ = _dataset(tmp_path)
    metadata = root / "metadata" / "kinect" / "scene_0100" / "0000.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload[0]["grasps"] = [list(rect_to_points8(50, 50, 0, 150, 40))]
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "invalid.jsonl"

    args = _convert_args(root, output)
    args.official_graspnet_eval = False
    stats = converter.convert(args)

    assert output.read_text(encoding="utf-8") == ""
    assert stats["filter_reasons"]["unrepresentable_width"] == 1


def test_official_converter_keeps_object_with_only_unrepresentable_grasps(
    tmp_path: Path,
) -> None:
    root, _ = _dataset(tmp_path)
    metadata = root / "metadata" / "kinect" / "scene_0100" / "0000.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload[0]["grasps"] = [list(rect_to_points8(50, 150, 0, 40, 40))]
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "official.jsonl"
    args = _convert_args(root, output)
    args.official_graspnet_eval = True

    stats = converter.convert(args)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert stats["statistics"]["positive_samples"] == 1
    assert stats["statistics"]["trainable_positive_samples"] == 0
    assert row["grasp_rect_candidates"] == []
    assert len(row["evaluation_grasp_rectangles_pixels"]) == 1


def test_sample_hash_is_independent_of_dataset_mount_path(tmp_path: Path) -> None:
    first_root, _ = _dataset(tmp_path / "first")
    second_root, _ = _dataset(tmp_path / "second")

    first = converter.convert(_convert_args(first_root, tmp_path / "first.jsonl"))
    second = converter.convert(
        _convert_args(second_root, tmp_path / "second.jsonl")
    )

    assert first["protocol"]["sample_id_sha256"] == second["protocol"][
        "sample_id_sha256"
    ]


def test_realvlg_decoder_accepts_pixel_parameters() -> None:
    decoded = evaluator.decode_prediction(
        "<think>...</think><answer>(50, 50, 179, 40)</answer>",
        100,
        100,
        "realvlg",
    )

    assert decoded.status == "ok"
    assert decoded.parameters_pixels == (50.0, 50.0, 179.0, 40.0)
    assert decoded.points8 is not None


def test_corrected_angle_metric_does_not_guess_units() -> None:
    corrected = evaluator.corrected_angle_error_degrees(1.0, 2.0)
    official_buggy = evaluator.official_buggy_angle_error_degrees(1.0, 2.0)

    assert corrected == pytest.approx(1.0)
    assert official_buggy > 50.0


def test_strict_metrics_count_invalid_output_as_zero(tmp_path: Path) -> None:
    gt = list(rect_to_points8(50, 50, 0, 40, 40))
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "results.jsonl"
    common = {
        "task_type": "grasp_rect",
        "image_width": 100,
        "image_height": 100,
        "grasp_rect_candidates": [[500, 500, 0, 283]],
        "evaluation_grasp_rectangles_pixels": [gt],
    }
    _write_jsonl(
        annotations,
        [
            {**common, "sample_id": "valid"},
            {**common, "sample_id": "invalid"},
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": "valid",
                "raw_output": "<answer>(50, 50, 0, 40)</answer>",
            },
            {"sample_id": "invalid", "raw_output": "not a grasp"},
        ],
    )
    args = argparse.Namespace(
        annotations=[annotations],
        predictions=predictions,
        output=output,
        metrics=None,
        prediction_format="realvlg",
        gripper_depth_pixels=40.0,
        minimum_width_diagonal=1e-4,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["format_valid_rate"] == pytest.approx(0.5)
    assert metrics["mIoU_valid"] == pytest.approx(1.0)
    assert metrics["mIoU_strict"] == pytest.approx(0.5)
    assert metrics["gAcc_corrected_valid"] == pytest.approx(1.0)
    assert metrics["gAcc_corrected_strict"] == pytest.approx(0.5)
    assert metrics["representation_oracle_mIoU_strict"] > 0.99
    assert metrics["miou_oracle_ratio"] == pytest.approx(0.5, abs=1e-3)


def test_phase0_audit_requires_review_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "realvlg"
    for scene in (0, 100, 130, 160):
        image_path = (
            root
            / "scenes"
            / f"scene_{scene:04d}"
            / "kinect"
            / "rgb"
            / "0000.png"
        )
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 100), "white").save(image_path)
        metadata = (
            root
            / "metadata"
            / "kinect"
            / f"scene_{scene:04d}"
            / "0000.json"
        )
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            json.dumps(
                [
                    {
                        "object_id": scene,
                        "description": "object",
                        "image_path": image_path.relative_to(root).as_posix(),
                        "grasps": [list(rect_to_points8(50, 50, 0, 40, 40))],
                    }
                ]
            ),
            encoding="utf-8",
        )
    output_dir = tmp_path / "audit"
    expected_hashes = {}
    for split in ("seen", "similar", "novel"):
        split_output = tmp_path / f"{split}.jsonl"
        split_args = _convert_args(root, split_output)
        split_args.split = split
        stats = converter.convert(split_args)
        expected_hashes[split] = stats["protocol"]["sample_id_sha256"]
    monkeypatch.setattr(audit_script, "MIN_RANDOM_VISUALIZATIONS", 1)
    monkeypatch.setattr(audit_script, "MIN_BOUNDARY_VISUALIZATIONS", 1)
    monkeypatch.setattr(
        converter,
        "FROZEN_OFFICIAL_SPLIT_COUNTS",
        {split: 1 for split in expected_hashes},
    )
    monkeypatch.setattr(
        converter, "FROZEN_OFFICIAL_SPLIT_HASHES", expected_hashes
    )
    args = argparse.Namespace(
        data_root=root,
        output_dir=output_dir,
        metadata_dir=None,
        camera="kinect",
        seed=42,
        random_visualizations=1,
        boundary_visualizations=1,
        expected_hash=[],
        confirm_visual_review=False,
    )

    pending = audit_script.audit(args)

    assert pending["accepted"] is False
    assert not (output_dir / ".phase0_complete").exists()

    args.confirm_visual_review = True
    completed = audit_script.audit(args)

    assert completed["accepted"] is True
    manifest = json.loads(
        (output_dir / "phase0_audit.json").read_text(encoding="utf-8")
    )
    assert manifest["accepted"] is True
    assert set(manifest["splits"]) == {"train", "seen", "similar", "novel"}
    assert (output_dir / ".phase0_complete").is_file()
    assert len(list((output_dir / "visualizations" / "random").glob("*.jpg"))) == 1

    args.random_visualizations = 0
    with pytest.raises(ValueError, match="at least 1 random"):
        audit_script.audit(args)
    assert not (output_dir / ".phase0_complete").exists()
