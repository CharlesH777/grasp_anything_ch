#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def _validate_contact_row(
    row: dict[str, Any], dataset_name: str, path: Path, line_number: int
) -> None:
    prefix = f"dataset '{dataset_name}' {path}:{line_number}"
    task_type = row.get("task_type")
    if task_type == "grasp_contact":
        width = row.get("image_width")
        height = row.get("image_height")
        if not isinstance(width, int | float) or width <= 0:
            fail(f"{prefix} needs a positive image_width")
        if not isinstance(height, int | float) or height <= 0:
            fail(f"{prefix} needs a positive image_height")
        candidates = row.get("contact_candidates")
        if not isinstance(candidates, list) or not candidates:
            fail(f"{prefix} needs at least one contact candidate")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, list | tuple) or len(candidate) != 4:
                fail(f"{prefix} candidate {candidate_index} must have four values")
            if not all(
                isinstance(value, int | float)
                and math.isfinite(value)
                and float(value).is_integer()
                and 0 <= value <= 1000
                for value in candidate
            ):
                fail(
                    f"{prefix} candidate {candidate_index} must contain integer "
                    "token values in [0, 1000]"
                )
            if candidate[:2] == candidate[2:]:
                fail(f"{prefix} candidate {candidate_index} is degenerate")
        collision_scores = row.get("candidate_collision_2d", [])
        if row.get("collision_valid") and (
            len(collision_scores) < len(candidates)
            or not all(
                isinstance(value, int | float)
                and math.isfinite(value)
                and 0.0 <= value <= 1.0
                for value in collision_scores[: len(candidates)]
            )
        ):
            fail(
                f"{prefix} collision_valid requires one [0, 1] score per candidate"
            )
        outside_scores = row.get("candidate_outside_2d", [])
        if len(outside_scores) < len(candidates) or not all(
            isinstance(value, int | float)
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
            for value in outside_scores[: len(candidates)]
        ):
            fail(f"{prefix} requires one candidate_outside_2d score per candidate")
        assistant_text = "".join(
            str(message.get("value", ""))
            for message in row.get("conversations", [])
            if message.get("from") == "gpt"
        )
        expected_box = "<box>" + "".join(
            f"<{int(value)}>" for value in candidates[0]
        ) + "</box>"
        if assistant_text.count(expected_box) != 1:
            fail(f"{prefix} assistant target does not match contact_candidates[0]")
    elif task_type == "grasp_contact_negative":
        if row.get("contact_candidates") not in (None, []):
            fail(f"{prefix} negative row must not contain contact candidates")
        if row.get("negative_reason") not in {"no_target", "ungraspable"}:
            fail(f"{prefix} has an unsupported negative_reason")
        assistant_text = "".join(
            str(message.get("value", ""))
            for message in row.get("conversations", [])
            if message.get("from") == "gpt"
        )
        if assistant_text.count("<box>none</box>") != 1:
            fail(f"{prefix} negative target must contain exactly one <box>none</box>")


def validate_jsonl(
    path: Path, dataset_name: str, configured_task_type: str | None
) -> None:
    if not path.is_absolute():
        fail(f"dataset '{dataset_name}' annotation must be an absolute path: {path}")
    if not path.is_file():
        fail(f"dataset '{dataset_name}' annotation not found: {path}")
    if path.stat().st_size == 0:
        fail(f"dataset '{dataset_name}' annotation is empty: {path}")

    sample_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    fail(f"invalid JSONL in {path} at line {line_number}: {error}")
                if not isinstance(sample, dict):
                    fail(
                        f"dataset '{dataset_name}' {path}:{line_number} "
                        "is not a JSON object"
                    )
                if configured_task_type is not None and sample.get(
                    "task_type", configured_task_type
                ) != configured_task_type:
                    fail(
                        f"dataset '{dataset_name}' task_type={configured_task_type!r} "
                        f"but {path}:{line_number} differs"
                    )
                conversations = sample.get("conversations")
                if not isinstance(conversations, list) or not conversations:
                    fail(
                        f"dataset '{dataset_name}' {path}:{line_number} needs "
                        "non-empty conversations"
                    )
                if not any(
                    key in sample for key in ("image", "image_list", "data", "video")
                ):
                    fail(
                        f"dataset '{dataset_name}' {path}:{line_number} has no media"
                    )
                _validate_contact_row(sample, dataset_name, path, line_number)
                sample_count += 1
    except OSError as error:
        fail(f"cannot read {path}: {error}")

    if sample_count == 0:
        fail(f"dataset '{dataset_name}' has no JSON object samples: {path}")


def validate_meta(meta_path: Path) -> None:
    metadata = load_json(meta_path)
    if not isinstance(metadata, dict) or not metadata:
        fail("metadata must be a non-empty JSON object")

    for dataset_name, dataset in metadata.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            fail("dataset names must be non-empty strings")
        if not isinstance(dataset, dict):
            fail(f"dataset '{dataset_name}' configuration must be an object")
        task_type = dataset.get("task_type")
        supported_task_types = {
            None,
            "grounding",
            "grasp_contact",
            "grasp_contact_negative",
        }
        if task_type not in supported_task_types:
            fail(f"dataset '{dataset_name}' has unsupported task_type={task_type!r}")
        sampling_weight = dataset.get("sampling_weight")
        if sampling_weight is not None and (
            not isinstance(sampling_weight, int | float) or sampling_weight <= 0
        ):
            fail(f"dataset '{dataset_name}' sampling_weight must be positive")

        root_value = dataset.get("root")
        if not isinstance(root_value, str) or not root_value:
            fail(f"dataset '{dataset_name}' needs a root directory")
        root_path = Path(root_value).expanduser()
        if not root_path.is_absolute():
            fail(f"dataset '{dataset_name}' root must be an absolute path: {root_path}")
        if not root_path.is_dir():
            fail(f"dataset '{dataset_name}' root directory not found: {root_path}")

        annotation_value = dataset.get("annotation")
        if isinstance(annotation_value, str):
            annotation_paths = [annotation_value]
        elif (
            isinstance(annotation_value, list)
            and annotation_value
            and all(isinstance(item, str) for item in annotation_value)
        ):
            annotation_paths = annotation_value
        else:
            fail(
                f"dataset '{dataset_name}' annotation must be a path or "
                "non-empty path list"
            )

        for annotation_path in annotation_paths:
            validate_jsonl(
                Path(annotation_path).expanduser(), dataset_name, task_type
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Eagle training metadata and JSONL inputs."
    )
    parser.add_argument("meta_path", type=Path)
    args = parser.parse_args()

    try:
        validate_meta(args.meta_path.expanduser().resolve())
    except ValueError as error:
        print(f"Training data validation failed: {error}", file=sys.stderr)
        return 1

    print(f"Training data validation passed: {args.meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
