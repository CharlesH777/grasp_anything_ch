#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationSummary:
    task_counts: Counter[str]
    sampling_fractions: dict[str, float]

    def __getitem__(self, task_type: str) -> int:
        return self.task_counts[task_type]


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
    row: dict[str, Any],
    dataset_name: str,
    path: Path,
    line_number: int,
    effective_task_type: str | None = None,
    collision_threshold: float | None = None,
    outside_threshold: float | None = None,
) -> None:
    prefix = f"dataset '{dataset_name}' {path}:{line_number}"
    if row.get("evaluation_only"):
        fail(f"{prefix} is evaluation-only and cannot be used for training")
    task_type = effective_task_type or row.get("task_type")
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
        primary_unsafe = (
            outside_threshold is not None
            and outside_scores[0] > outside_threshold
        ) or (
            collision_threshold is not None
            and row.get("collision_valid")
            and collision_scores[0] > collision_threshold
        )
        if primary_unsafe:
            fail(
                f"{prefix} primary contact candidate is unsafe under the active "
                "collision/outside thresholds"
            )
        assistant_text = "".join(
            str(message.get("value", ""))
            for message in row.get("conversations", [])
            if message.get("from") == "gpt"
        )
        expected_box = "<grasp>" + "".join(
            f"<{int(value)}>" for value in candidates[0]
        ) + "</grasp>"
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
        if assistant_text.count("<grasp>none</grasp>") != 1:
            fail(
                f"{prefix} negative target must contain exactly one "
                "<grasp>none</grasp>"
            )


def validate_jsonl(
    path: Path,
    dataset_name: str,
    configured_task_type: str | None,
    collision_threshold: float | None = None,
    outside_threshold: float | None = None,
) -> Counter[str]:
    if not path.is_absolute():
        fail(f"dataset '{dataset_name}' annotation must be an absolute path: {path}")
    if not path.is_file():
        fail(f"dataset '{dataset_name}' annotation not found: {path}")
    if path.stat().st_size == 0:
        fail(f"dataset '{dataset_name}' annotation is empty: {path}")

    sample_count = 0
    task_counts: Counter[str] = Counter()
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
                effective_task_type = sample.get("task_type")
                if effective_task_type is None:
                    effective_task_type = configured_task_type or "grounding"
                if effective_task_type not in {
                    "grounding",
                    "grasp_contact",
                    "grasp_contact_negative",
                }:
                    fail(
                        f"dataset '{dataset_name}' {path}:{line_number} has "
                        f"unsupported task_type={effective_task_type!r}"
                    )
                if (
                    configured_task_type is not None
                    and effective_task_type != configured_task_type
                ):
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
                _validate_contact_row(
                    sample,
                    dataset_name,
                    path,
                    line_number,
                    effective_task_type,
                    collision_threshold,
                    outside_threshold,
                )
                sample_count += 1
                task_counts[effective_task_type] += 1
    except OSError as error:
        fail(f"cannot read {path}: {error}")

    if sample_count == 0:
        fail(f"dataset '{dataset_name}' has no JSON object samples: {path}")
    return task_counts


def validate_meta(
    meta_path: Path,
    collision_threshold: float | None = None,
    outside_threshold: float | None = None,
    min_contact_samples: int = 0,
    min_grounding_samples: int = 0,
    min_negative_samples: int = 0,
    min_contact_fraction: float = 0.0,
    min_grounding_fraction: float = 0.0,
    min_negative_fraction: float = 0.0,
) -> ValidationSummary:
    minimums = {
        "grasp_contact": min_contact_samples,
        "grounding": min_grounding_samples,
        "grasp_contact_negative": min_negative_samples,
    }
    for task_type, minimum in minimums.items():
        if minimum < 0:
            fail(f"minimum sample count for {task_type} must be non-negative")
    minimum_fractions = {
        "grasp_contact": min_contact_fraction,
        "grounding": min_grounding_fraction,
        "grasp_contact_negative": min_negative_fraction,
    }
    for task_type, minimum in minimum_fractions.items():
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            fail(f"minimum sampling fraction for {task_type} must be in [0, 1]")
    require_explicit_weights = any(
        minimum > 0.0 for minimum in minimum_fractions.values()
    )
    metadata = load_json(meta_path)
    if not isinstance(metadata, dict) or not metadata:
        fail("metadata must be a non-empty JSON object")

    task_counts: Counter[str] = Counter()
    task_sampling_weights: Counter[str] = Counter()
    total_sampling_weight = 0.0
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
            not isinstance(sampling_weight, int | float)
            or not math.isfinite(float(sampling_weight))
            or sampling_weight <= 0
        ):
            fail(f"dataset '{dataset_name}' sampling_weight must be positive")
        if require_explicit_weights and sampling_weight is None:
            fail(
                f"dataset '{dataset_name}' needs explicit sampling_weight when "
                "task sampling fractions are enforced"
            )
        repeat_time = float(dataset.get("repeat_time", 1.0))
        if not math.isfinite(repeat_time) or repeat_time <= 0.0:
            fail(f"dataset '{dataset_name}' repeat_time must be positive")
        if require_explicit_weights and repeat_time < 1.0 and task_type is None:
            fail(
                f"dataset '{dataset_name}' mixes task types while repeat_time "
                "is below 1; split it by task before enforcing phase ratios"
            )

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

        dataset_counts: Counter[str] = Counter()
        for annotation_path in annotation_paths:
            dataset_counts.update(
                validate_jsonl(
                    Path(annotation_path).expanduser(),
                    dataset_name,
                    task_type,
                    collision_threshold,
                    outside_threshold,
                )
            )
        source_total = sum(dataset_counts.values())
        if repeat_time < 1.0:
            active_total = int(source_total * repeat_time)
            if active_total <= 0:
                fail(
                    f"dataset '{dataset_name}' repeat_time={repeat_time} "
                    "leaves no active training samples"
                )
            if task_type is not None:
                active_counts = Counter({task_type: active_total})
            else:
                # Mixed-task downsampling is deterministic at runtime but the
                # exact selected rows are not materialized here. Flooring each
                # task independently is conservative for minimum-count gates.
                active_counts = Counter({
                    counted_task: int(count * repeat_time)
                    for counted_task, count in dataset_counts.items()
                    if int(count * repeat_time) > 0
                })
        else:
            active_counts = dataset_counts
        if not active_counts:
            fail(
                f"dataset '{dataset_name}' has no active task samples after "
                f"repeat_time={repeat_time}"
            )
        task_counts.update(active_counts)
        active_total = sum(active_counts.values())
        if sampling_weight is None:
            active_length = (
                source_total * repeat_time
                if repeat_time >= 1.0
                else int(source_total * repeat_time)
            )
            weight = float(active_length)
        else:
            weight = float(sampling_weight)
        total_sampling_weight += weight
        for counted_task, count in active_counts.items():
            task_sampling_weights[counted_task] += weight * count / active_total

    for task_type, minimum in minimums.items():
        actual = task_counts[task_type]
        if actual < minimum:
            fail(
                f"training metadata contains {actual} {task_type} samples, "
                f"but at least {minimum} are required"
            )
    sampling_fractions = {
        task_type: (
            task_sampling_weights[task_type] / total_sampling_weight
            if total_sampling_weight > 0.0
            else 0.0
        )
        for task_type in minimum_fractions
    }
    for task_type, minimum in minimum_fractions.items():
        actual = sampling_fractions[task_type]
        if actual + 1e-12 < minimum:
            fail(
                f"training metadata samples {task_type} with fraction "
                f"{actual:.6f}, but at least {minimum:.6f} is required"
            )
    return ValidationSummary(task_counts, sampling_fractions)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Eagle training metadata and JSONL inputs."
    )
    parser.add_argument("meta_path", type=Path)
    parser.add_argument("--collision-threshold", type=float)
    parser.add_argument("--outside-threshold", type=float)
    parser.add_argument("--min-contact-samples", type=int, default=0)
    parser.add_argument("--min-grounding-samples", type=int, default=0)
    parser.add_argument("--min-negative-samples", type=int, default=0)
    parser.add_argument("--min-contact-fraction", type=float, default=0.0)
    parser.add_argument("--min-grounding-fraction", type=float, default=0.0)
    parser.add_argument("--min-negative-fraction", type=float, default=0.0)
    args = parser.parse_args()

    try:
        for name in ("collision_threshold", "outside_threshold"):
            value = getattr(args, name)
            if value is not None and not 0.0 <= value <= 1.0:
                fail(f"{name} must be in [0, 1]")
        summary = validate_meta(
            args.meta_path.expanduser().resolve(),
            args.collision_threshold,
            args.outside_threshold,
            args.min_contact_samples,
            args.min_grounding_samples,
            args.min_negative_samples,
            args.min_contact_fraction,
            args.min_grounding_fraction,
            args.min_negative_fraction,
        )
    except ValueError as error:
        print(f"Training data validation failed: {error}", file=sys.stderr)
        return 1

    counts_text = ", ".join(
        f"{task_type}={count}"
        for task_type, count in sorted(summary.task_counts.items())
    )
    fractions_text = ", ".join(
        f"{task_type}={fraction:.2%}"
        for task_type, fraction in sorted(summary.sampling_fractions.items())
    )
    print(
        f"Training data validation passed: {args.meta_path} "
        f"({counts_text}; sampling: {fractions_text})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
