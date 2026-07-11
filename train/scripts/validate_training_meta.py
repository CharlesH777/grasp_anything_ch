#!/usr/bin/env python3
import argparse
import json
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


def validate_jsonl(path: Path, dataset_name: str) -> None:
    if not path.is_absolute():
        fail(f"dataset '{dataset_name}' annotation must be an absolute path: {path}")
    if not path.is_file():
        fail(f"dataset '{dataset_name}' annotation not found: {path}")
    if path.stat().st_size == 0:
        fail(f"dataset '{dataset_name}' annotation is empty: {path}")

    first_sample = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    first_sample = json.loads(line)
                except json.JSONDecodeError as error:
                    fail(f"invalid JSONL in {path} at line {line_number}: {error}")
                break
    except OSError as error:
        fail(f"cannot read {path}: {error}")

    if not isinstance(first_sample, dict):
        fail(f"dataset '{dataset_name}' has no JSON object samples: {path}")
    if not isinstance(first_sample.get("conversations"), list) or not first_sample["conversations"]:
        fail(f"dataset '{dataset_name}' first sample needs a non-empty conversations list: {path}")
    if not any(key in first_sample for key in ("image", "image_list", "data", "video")):
        fail(f"dataset '{dataset_name}' first sample has no image, image_list, data, or video field: {path}")


def validate_meta(meta_path: Path) -> None:
    metadata = load_json(meta_path)
    if not isinstance(metadata, dict) or not metadata:
        fail("metadata must be a non-empty JSON object")

    for dataset_name, dataset in metadata.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            fail("dataset names must be non-empty strings")
        if not isinstance(dataset, dict):
            fail(f"dataset '{dataset_name}' configuration must be an object")

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
        elif isinstance(annotation_value, list) and annotation_value and all(isinstance(item, str) for item in annotation_value):
            annotation_paths = annotation_value
        else:
            fail(f"dataset '{dataset_name}' annotation must be a path or non-empty path list")

        for annotation_path in annotation_paths:
            validate_jsonl(Path(annotation_path).expanduser(), dataset_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Eagle training metadata and JSONL inputs.")
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
