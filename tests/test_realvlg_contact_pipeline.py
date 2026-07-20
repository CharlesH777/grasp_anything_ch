from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "training" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_script("convert_realvlg_contact")
evaluator = _load_script("evaluate_realvlg_contact")
merger = _load_script("merge_realvlg_contact_shards")
validator = _load_script("validate_training_meta")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "realvlg"
    metadata_dir = root / "metadata" / "scene_0000" / "realsense"
    image_dir = root / "images" / "scene_0000" / "realsense"
    mask_dir = root / "masks" / "scene_0000" / "realsense"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image_path = image_dir / "0000.png"
    Image.new("RGB", (200, 100), "white").save(image_path)

    left_mask = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(left_mask).rectangle((20, 20, 80, 80), fill=255)
    left_mask.save(mask_dir / "left.png")
    right_mask = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(right_mask).rectangle((120, 20, 180, 80), fill=255)
    right_mask.save(mask_dir / "right.png")

    relative_image = image_path.relative_to(root).as_posix()
    objects = [
        {
            "object_id": 1,
            "description": "the object on the left",
            "image_path": relative_image,
            "image_size_hw": [100, 200],
            "mask_path": (mask_dir / "left.png").relative_to(root).as_posix(),
            "contact_points": [
                [25.25, 50.0, 75.75, 50.0],
                [25.25, 50.0, 75.75, 50.0],
                [30.0, 40.0, 70.0, 60.0],
            ],
        },
        {
            "object_id": 2,
            "description": "the object on the right",
            "image_path": relative_image,
            "image_size_hw": [100, 200],
            "mask_path": (mask_dir / "right.png").relative_to(root).as_posix(),
            "contact_points": [[125.0, 50.0, 175.0, 50.0]],
        },
    ]
    (metadata_dir / "0000.json").write_text(json.dumps(objects), encoding="utf-8")
    return root, metadata_dir


def test_shard_merger_preserves_order_and_sums_statistics(tmp_path: Path) -> None:
    shards = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    stats_shards = [tmp_path / "a_stats.json", tmp_path / "b_stats.json"]
    _write_jsonl(shards[0], [{"sample_id": "a"}])
    _write_jsonl(shards[1], [{"sample_id": "b"}])
    for index, path in enumerate(stats_shards):
        path.write_text(
            json.dumps(
                {
                    "statistics": {"positive_samples": index + 1},
                    "filter_reasons": {"duplicate": index},
                    "configuration": {
                        "split": "train",
                        "scene_start": index * 25,
                        "scene_end_exclusive": (index + 1) * 25,
                    },
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "merged.jsonl"
    stats_output = tmp_path / "merged_stats.json"

    result = merger.merge(shards, stats_shards, output, stats_output)

    output_ids = [
        json.loads(line)["sample_id"] for line in output.read_text().splitlines()
    ]
    assert output_ids == [
        "a",
        "b",
    ]
    assert result["statistics"]["positive_samples"] == 3
    assert result["filter_reasons"]["duplicate"] == 1
    assert result["configuration"]["scene_shards"] == [[0, 25], [25, 50]]


def test_converter_preserves_raw_gt_and_builds_instance_excluded_collision(
    tmp_path: Path,
) -> None:
    root, metadata_dir = _make_dataset(tmp_path)
    output = tmp_path / "converted" / "train.jsonl"
    stats = tmp_path / "converted" / "stats.json"
    args = argparse.Namespace(
        data_root=root,
        metadata_dir=metadata_dir,
        output=output,
        stats=stats,
        dataset_name="synthetic",
        split="train",
        camera="realsense",
        max_candidates=2,
        rectangle_thickness=20.0,
        min_width_diagonal=1e-4,
        max_width_diagonal=1.0,
        collision_threshold=0.0,
        collision_masks_exhaustive=True,
        derived_mask_dir=None,
    )

    summary = converter.convert(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["statistics"]["positive_samples"] == 2
    assert summary["statistics"]["images_sized_from_metadata"] == 1
    assert len(rows) == 2
    left = next(row for row in rows if row["object_id"] == 1)
    assert left["task_type"] == "grasp_contact"
    assert left["collision_valid"] is True
    assert len(left["contact_candidates"]) == 2
    assert left["contact_candidates_pixels"][0] in (
        [25.25, 50.0, 75.75, 50.0],
        [30.0, 40.0, 70.0, 60.0],
    )
    primary = "".join(f"<{value}>" for value in left["contact_candidates"][0])
    assert f"<grasp>{primary}</grasp>" in left["conversations"][1]["value"]
    assert len(left["candidate_collision_2d"]) == 2
    assert all(math.isfinite(value) for value in left["candidate_collision_2d"])
    assert len(left["candidate_outside_2d"]) == 2
    assert all(0.0 <= value <= 1.0 for value in left["candidate_outside_2d"])
    assert Path(left["obstacle_mask"]).is_file()


def test_converter_deduplicates_identical_metadata_aliases(tmp_path: Path) -> None:
    root, metadata_dir = _make_dataset(tmp_path)
    source = metadata_dir / "0000.json"
    (metadata_dir / "0000.png.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    output = tmp_path / "train.jsonl"

    summary = converter.convert(
        _converter_args(
            root,
            metadata_dir,
            output,
            grasp_candidates_exhaustive=False,
        )
    )

    assert summary["statistics"]["source_records"] == 2
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def _all_left_candidates_collide(root: Path) -> None:
    obstacle_path = root / "masks" / "scene_0000" / "realsense" / "right.png"
    obstacle = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(obstacle).rectangle((10, 10, 90, 90), fill=255)
    obstacle.save(obstacle_path)


def _converter_args(
    root: Path,
    metadata_dir: Path,
    output: Path,
    *,
    grasp_candidates_exhaustive: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=root,
        metadata_dir=metadata_dir,
        output=output,
        stats=None,
        dataset_name="synthetic",
        split="train",
        camera="realsense",
        max_candidates=2,
        rectangle_thickness=20.0,
        min_width_diagonal=1e-4,
        max_width_diagonal=1.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        collision_masks_exhaustive=True,
        grasp_candidates_exhaustive=grasp_candidates_exhaustive,
        derived_mask_dir=None,
    )


def test_converter_skips_all_unsafe_candidates_when_annotations_are_not_exhaustive(
    tmp_path: Path,
) -> None:
    root, metadata_dir = _make_dataset(tmp_path)
    _all_left_candidates_collide(root)
    output = tmp_path / "train.jsonl"

    summary = converter.convert(
        _converter_args(
            root,
            metadata_dir,
            output,
            grasp_candidates_exhaustive=False,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["statistics"]["objects_without_safe_annotated_candidate"] == 1
    assert all(row["object_id"] != 1 for row in rows)


def test_converter_filters_unsafe_candidate_before_selecting_primary(
    tmp_path: Path,
) -> None:
    root, metadata_dir = _make_dataset(tmp_path)
    metadata_path = metadata_dir / "0000.json"
    objects = json.loads(metadata_path.read_text(encoding="utf-8"))
    objects[0]["contact_points"] = [
        [25.0, 30.0, 75.0, 30.0],
        [25.0, 70.0, 75.0, 70.0],
    ]
    metadata_path.write_text(json.dumps(objects), encoding="utf-8")
    obstacle_path = root / "masks" / "scene_0000" / "realsense" / "right.png"
    obstacle = Image.new("L", (200, 100), 0)
    ImageDraw.Draw(obstacle).rectangle((10, 20, 90, 40), fill=255)
    obstacle.save(obstacle_path)
    output = tmp_path / "train.jsonl"

    converter.convert(
        _converter_args(
            root,
            metadata_dir,
            output,
            grasp_candidates_exhaustive=False,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    left = next(row for row in rows if row["object_id"] == 1)

    assert left["contact_candidates"] == [[125, 700, 375, 700]]
    assert left["candidate_collision_2d"] == [0.0]
    assert left["conversations"][1]["value"].endswith(
        "<grasp><125><700><375><700></grasp>"
    )


def test_converter_emits_no_grasp_only_for_exhaustive_unsafe_candidates(
    tmp_path: Path,
) -> None:
    root, metadata_dir = _make_dataset(tmp_path)
    _all_left_candidates_collide(root)
    output = tmp_path / "train.jsonl"

    converter.convert(
        _converter_args(
            root,
            metadata_dir,
            output,
            grasp_candidates_exhaustive=True,
        )
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    left = next(row for row in rows if row["object_id"] == 1)

    assert left["task_type"] == "grasp_contact_negative"
    assert left["negative_reason"] == "ungraspable"
    assert left["contact_candidates"] == []
    assert left["conversations"][1]["value"].endswith("<grasp>none</grasp>")
    validator._validate_contact_row(left, "synthetic", output, 1)


def test_evaluator_counts_invalid_positive_as_zero_and_none_separately(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "image.png")
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "nested" / "predictions.jsonl"
    metrics_path = tmp_path / "nested" / "metrics.json"
    base = {
        "image": "image.png",
        "image_width": 100,
        "image_height": 100,
        "contact_candidates_pixels": [[20.0, 50.0, 80.0, 50.0]],
        "collision_valid": False,
    }
    _write_jsonl(
        annotations,
        [
            {**base, "sample_id": "valid", "task_type": "grasp_contact"},
            {**base, "sample_id": "invalid", "task_type": "grasp_contact"},
            {
                **base,
                "sample_id": "negative",
                "task_type": "grasp_contact_negative",
            },
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": "valid",
                "raw_output": "<grasp><200><500><800><500></grasp>",
            },
            {
                "sample_id": "invalid",
                "raw_output": "<grasp><200><500><800></grasp>",
            },
            {"sample_id": "negative", "raw_output": "<grasp>none</grasp>"},
        ],
    )
    args = argparse.Namespace(
        annotations=annotations,
        data_root=root,
        predictions=predictions,
        model_path=None,
        output=output,
        metrics=metrics_path,
        generation_mode="fast",
        max_new_tokens=64,
        rectangle_thickness=20.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        seed=42,
        limit=None,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["positive_samples"] == 2
    assert metrics["format_valid_rate"] == 0.5
    assert metrics["miou_valid"] == 1.0
    assert metrics["miou_strict"] == 0.5
    assert metrics["gacc_corrected_valid"] == 1.0
    assert metrics["gacc_corrected_strict"] == 0.5
    assert metrics["gacc_official_buggy_valid"] == 1.0
    assert "gacc_legacy_valid" not in metrics
    assert metrics["swap_invariant_endpoint_error_pixels"]["mean"] == 0.0
    assert metrics["swap_invariant_angle_error_degrees"]["mean"] == 0.0
    assert metrics["center_error_pixels"]["mean"] == 0.0
    assert metrics["width_error_pixels"]["mean"] == 0.0
    assert metrics["outside_rate_2d_geometry"] == 0.0
    assert metrics["collision_rate_2d"] is None
    assert metrics["collision_aware_gacc_valid"] is None
    assert metrics["none_precision"] == 1.0
    assert metrics["none_recall"] == 1.0
    assert output.is_file()
    assert metrics_path.is_file()


def test_official_buggy_angle_metric_is_explicitly_not_the_main_metric() -> None:
    corrected = math.degrees(
        abs(((math.radians(1) - math.radians(2) + math.pi / 2) % math.pi) - math.pi / 2)
    )
    official_buggy = evaluator.official_buggy_angular_error_degrees(1.0, 2.0)

    assert corrected == pytest.approx(1.0)
    assert official_buggy > 50.0


def test_realvlg_contact_output_decodes_absolute_pixel_coordinates() -> None:
    parsed = evaluator.decode_prediction(
        "<think>target</think><answer>(20,50),(80,50)</answer>",
        width=100,
        height=100,
        prediction_format="realvlg",
    )

    assert parsed.status == "ok"
    assert parsed.contacts_1000 == (200.0, 500.0, 800.0, 500.0)


def test_evaluator_scores_realvlg_predictions_with_shared_geometry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "image.png")
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        annotations,
        [
            {
                "sample_id": "realvlg",
                "task_type": "grasp_contact",
                "image": "image.png",
                "image_width": 100,
                "image_height": 100,
                "contact_candidates_pixels": [[20, 50, 80, 50]],
                "collision_valid": False,
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": "realvlg",
                "raw_output": "<answer>(20,50),(80,50)</answer>",
            }
        ],
    )
    args = argparse.Namespace(
        annotations=annotations,
        data_root=root,
        predictions=predictions,
        model_path=None,
        model_family="realvlg",
        prediction_format="realvlg",
        output=tmp_path / "output.jsonl",
        metrics=tmp_path / "metrics.json",
        generation_mode="fast",
        max_new_tokens=128,
        coord_mass_threshold=1e-4,
        rectangle_thickness=20.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        seed=42,
        limit=None,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["format_valid_rate"] == 1.0
    assert metrics["miou_strict"] == pytest.approx(1.0)
    assert metrics["gacc_corrected_strict"] == 1.0
    assert metrics["prediction_format"] == "realvlg"


def test_evaluator_keeps_out_of_bounds_realvlg_output_valid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "image.png")
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        annotations,
        [
            {
                "sample_id": "outside",
                "task_type": "grasp_contact",
                "image": "image.png",
                "image_width": 100,
                "image_height": 100,
                "contact_candidates_pixels": [[20, 50, 80, 50]],
                "collision_valid": False,
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": "outside",
                "raw_output": "<answer>(120,50),(180,50)</answer>",
            }
        ],
    )
    args = argparse.Namespace(
        annotations=annotations,
        data_root=root,
        predictions=predictions,
        model_path=None,
        model_family="realvlg",
        prediction_format="realvlg",
        output=tmp_path / "output.jsonl",
        metrics=tmp_path / "metrics.json",
        generation_mode="fast",
        max_new_tokens=128,
        coord_mass_threshold=1e-4,
        rectangle_thickness=20.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        seed=42,
        limit=None,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["format_valid_rate"] == 1.0
    assert metrics["positive_grasp_output_rate"] == 1.0
    assert metrics["miou_strict"] == 0.0
    assert metrics["gacc_corrected_strict"] == 0.0
    result = json.loads(args.output.read_text(encoding="utf-8"))
    assert result["prediction_contacts_pixels"] == [120.0, 50.0, 180.0, 50.0]


def test_evaluator_matches_out_of_bounds_raw_gt_without_clipping(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "image.png")
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        annotations,
        [
            {
                "sample_id": "outside-match",
                "task_type": "grasp_contact",
                "image": "image.png",
                "image_width": 100,
                "image_height": 100,
                "evaluation_contact_candidates_pixels": [[120, 50, 180, 50]],
                "collision_valid": False,
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": "outside-match",
                "raw_output": "<answer>(120,50),(180,50)</answer>",
            }
        ],
    )
    args = argparse.Namespace(
        annotations=annotations,
        data_root=root,
        predictions=predictions,
        model_path=None,
        model_family="realvlg",
        prediction_format="realvlg",
        output=tmp_path / "output.jsonl",
        metrics=tmp_path / "metrics.json",
        generation_mode="fast",
        max_new_tokens=128,
        coord_mass_threshold=1e-4,
        rectangle_thickness=20.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        seed=42,
        limit=None,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["miou_strict"] == pytest.approx(1.0)
    assert metrics["gacc_corrected_strict"] == 1.0


def test_none_is_valid_syntax_but_not_a_positive_grasp_output(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "image.png")
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    common = {
        "image": "image.png",
        "image_width": 100,
        "image_height": 100,
        "contact_candidates_pixels": [[20.0, 50.0, 80.0, 50.0]],
        "collision_valid": False,
    }
    _write_jsonl(
        annotations,
        [
            {**common, "sample_id": "positive", "task_type": "grasp_contact"},
            {
                **common,
                "sample_id": "negative",
                "task_type": "grasp_contact_negative",
            },
        ],
    )
    _write_jsonl(
        predictions,
        [
            {"sample_id": "positive", "raw_output": "<grasp>none</grasp>"},
            {"sample_id": "negative", "raw_output": "<grasp>none</grasp>"},
        ],
    )
    args = argparse.Namespace(
        annotations=annotations,
        data_root=root,
        predictions=predictions,
        model_path=None,
        output=tmp_path / "out.jsonl",
        metrics=tmp_path / "metrics.json",
        generation_mode="fast",
        max_new_tokens=64,
        rectangle_thickness=20.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        coord_mass_threshold=1e-4,
        seed=42,
        limit=None,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["format_valid_rate"] == 1.0
    assert metrics["positive_grasp_output_rate"] == 0.0
    assert metrics["negative_format_valid_rate"] == 1.0
    assert metrics["overall_format_valid_rate"] == 1.0
    assert metrics["gacc_corrected_strict"] == 0.0


def test_converter_split_boundaries_match_realvlg_executable_ranges() -> None:
    assert converter._matches_split(Path("scene_0129/0000.json"), "seen")
    assert not converter._matches_split(Path("scene_0130/0000.json"), "seen")
    assert converter._matches_split(Path("scene_0130/0000.json"), "similar")
    assert converter._matches_split(Path("scene_0160/0000.json"), "novel")


def test_official_eval_conversion_uses_kinect_0000_only(tmp_path: Path) -> None:
    root = tmp_path / "realvlg"
    metadata = root / "metadata" / "kinect" / "scene_0100"
    images = root / "images" / "kinect" / "scene_0100"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(images / "0000.png")
    Image.new("RGB", (100, 100), "white").save(images / "0001.png")
    for frame in ("0000", "0001"):
        row = {
            "object_id": 1,
            "description": "target",
            "image_path": f"images/kinect/scene_0100/{frame}.png",
            "grasps": [[10, 10, 20, 20, 30, 30, 40, 40]],
            "contact_points": [
                [20, 0, 80, 0],
                [10, 0, 90, 0],
                [30, 0, 70, 0],
            ],
        }
        (metadata / f"{frame}.json").write_text(json.dumps([row]), encoding="utf-8")

    output = tmp_path / "seen.jsonl"
    args = argparse.Namespace(
        data_root=root,
        metadata_dir=root / "metadata",
        output=output,
        stats=None,
        dataset_name="synthetic",
        split="seen",
        camera=None,
        official_graspnet_eval=True,
        max_candidates=1,
        rectangle_thickness=20.0,
        min_width_diagonal=1e-4,
        max_width_diagonal=1.0,
        collision_threshold=0.0,
        collision_masks_exhaustive=False,
        derived_mask_dir=None,
    )

    summary = converter.convert(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["configuration"]["camera"] == "kinect"
    assert summary["configuration"]["official_graspnet_eval"] is True
    assert len(rows) == 1
    assert rows[0]["image"].endswith("0000.png")
    assert rows[0]["evaluation_protocol"] == "realvlg_graspnet_official"
    assert rows[0]["evaluation_only"] is True
    assert rows[0]["task_type"] == "grasp_contact"
    assert len(rows[0]["contact_candidates_pixels"]) == 1
    assert len(rows[0]["evaluation_contact_candidates_pixels"]) == 3
    assert [10.0, 0.0, 90.0, 0.0] in rows[0]["evaluation_contact_candidates_pixels"]
    with pytest.raises(ValueError, match="evaluation-only"):
        validator._validate_contact_row(rows[0], "synthetic", output, 1)


def test_evaluator_prefers_complete_official_gt_candidates(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (100, 100), "white").save(root / "image.png")
    annotations = tmp_path / "annotations.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "output.jsonl"
    metrics_path = tmp_path / "metrics.json"
    _write_jsonl(
        annotations,
        [
            {
                "sample_id": "official",
                "task_type": "grasp_contact",
                "image": "image.png",
                "image_width": 100,
                "image_height": 100,
                "contact_candidates_pixels": [[10, 10, 20, 10]],
                "evaluation_contact_candidates_pixels": [[20, 50, 80, 50]],
                "collision_valid": False,
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "sample_id": "official",
                "raw_output": "<grasp><200><500><800><500></grasp>",
            }
        ],
    )
    args = argparse.Namespace(
        annotations=annotations,
        data_root=root,
        predictions=predictions,
        model_path=None,
        output=output,
        metrics=metrics_path,
        generation_mode="fast",
        max_new_tokens=128,
        coord_mass_threshold=1e-4,
        rectangle_thickness=20.0,
        collision_threshold=0.0,
        outside_threshold=0.0,
        seed=42,
        limit=None,
    )

    metrics = evaluator.evaluate(args)

    assert metrics["gacc_corrected_strict"] == 1.0
