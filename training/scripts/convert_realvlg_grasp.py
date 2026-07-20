#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

from locate_anything_service.grasp_geometry import (
    polygon_area,
    polygon_inside_image_area,
    polygon_iou,
)
from locate_anything_service.grasp_rect_geometry import (
    DEFAULT_GRIPPER_DEPTH_PIXELS,
    DEFAULT_MINIMUM_WIDTH_DIAGONAL,
    encode_grasp_rectangle_pixels,
    points8_to_rect,
    rect_to_points8,
)

SPLIT_RANGES = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RealVLG rectangular grasps to structured PBD JSONL."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--dataset-name", default="GraspNet_VLG")
    parser.add_argument("--camera", default="kinect")
    parser.add_argument("--split", choices=("all", *SPLIT_RANGES), default="all")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument(
        "--minimum-width-diagonal",
        type=float,
        default=DEFAULT_MINIMUM_WIDTH_DIAGONAL,
    )
    parser.add_argument(
        "--gripper-depth-pixels",
        type=float,
        default=DEFAULT_GRIPPER_DEPTH_PIXELS,
    )
    parser.add_argument("--official-graspnet-eval", action="store_true")
    parser.add_argument("--scene-start", type=int)
    parser.add_argument("--scene-end-exclusive", type=int)
    parser.add_argument("--expected-sample-id-sha256")
    return parser.parse_args()


def _scene_id(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("scene_"):
            try:
                return int(part.removeprefix("scene_"))
            except ValueError:
                return None
    return None


def _included(path: Path, args: argparse.Namespace) -> bool:
    scene_id = _scene_id(path)
    if args.scene_start is not None and (
        scene_id is None or scene_id < args.scene_start
    ):
        return False
    if args.scene_end_exclusive is not None and (
        scene_id is None or scene_id >= args.scene_end_exclusive
    ):
        return False
    if args.split != "all" and (
        scene_id is None or scene_id not in SPLIT_RANGES[args.split]
    ):
        return False
    return not args.official_graspnet_eval or path.stem == "0000"


def _resolve_data_path(data_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _infer_image_path(
    data_root: Path, metadata_path: Path, camera: str
) -> Path | None:
    scene = next(
        (part for part in metadata_path.parts if part.startswith("scene_")),
        None,
    )
    if scene is None:
        return None
    frame_name = metadata_path.name.removesuffix(".json")
    if not Path(frame_name).suffix:
        frame_name += ".png"
    return (data_root / "scenes" / scene / camera / "rgb" / frame_name).resolve()


def _relative_or_absolute(path: Path, data_root: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return str(path)


def _sample_id(
    dataset_name: str,
    metadata_path: Path,
    image_path: Path,
    object_id: Any,
    data_root: Path,
) -> str:
    try:
        relative_image_path = image_path.resolve().relative_to(
            data_root.resolve()
        ).as_posix()
    except ValueError as error:
        raise ValueError(
            f"image path is outside data_root and cannot form a stable ID: "
            f"{image_path}"
        ) from error
    digest = hashlib.sha1(relative_image_path.encode("utf-8")).hexdigest()[:12]
    scene = next(
        (part for part in metadata_path.parts if part.startswith("scene_")),
        metadata_path.parent.name,
    )
    return f"{dataset_name}:{scene}:{digest}:{object_id}"


def _polygon(points8: tuple[float, ...]):
    return tuple(
        (points8[index], points8[index + 1]) for index in range(0, 8, 2)
    )


def _depth(points8: tuple[float, ...]) -> float:
    return math.hypot(points8[4] - points8[2], points8[5] - points8[3])


def _valid_candidates(
    raw_candidates: Any,
    width: int,
    height: int,
    minimum_width_diagonal: float,
    gripper_depth_pixels: float,
) -> tuple[list[dict[str, Any]], Counter]:
    valid: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    seen_tokens: set[tuple[int, int, int, int]] = set()
    for raw in raw_candidates or []:
        if not isinstance(raw, list | tuple) or len(raw) != 8:
            reasons["bad_shape"] += 1
            continue
        try:
            points8 = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            reasons["non_numeric"] += 1
            continue
        if not all(math.isfinite(value) for value in points8):
            reasons["non_finite"] += 1
            continue
        try:
            center_x, center_y, theta_degrees, width_pixels = points8_to_rect(
                points8
            )
            tokens = encode_grasp_rectangle_pixels(
                center_x,
                center_y,
                theta_degrees,
                width_pixels,
                width,
                height,
                minimum_width_diagonal=minimum_width_diagonal,
            )
        except ValueError as error:
            message = str(error)
            if "center" in message:
                reasons["center_out_of_bounds"] += 1
            elif "exceed" in message or "positive" in message:
                reasons["too_narrow"] += 1
            elif "token representation" in message:
                reasons["unrepresentable_width"] += 1
            else:
                reasons["invalid_geometry"] += 1
            continue
        if tokens in seen_tokens:
            reasons["quantized_duplicate"] += 1
            continue
        seen_tokens.add(tokens)

        reconstructed = rect_to_points8(
            center_x,
            center_y,
            theta_degrees,
            width_pixels,
            gripper_depth_pixels,
        )
        original_polygon = _polygon(points8)
        reconstructed_polygon = _polygon(reconstructed)
        rectangle_area = width_pixels * gripper_depth_pixels
        if polygon_area(original_polygon) <= 1e-9:
            reasons["degenerate_polygon"] += 1
            continue
        is_near_duplicate = False
        for previous in valid:
            previous_x, previous_y, previous_theta, previous_width = previous[
                "parameters_pixels"
            ]
            angle_delta = abs(theta_degrees - previous_theta) % 180.0
            angle_delta = min(angle_delta, 180.0 - angle_delta)
            if (
                math.hypot(center_x - previous_x, center_y - previous_y) <= 1.0
                and angle_delta <= 1.0
                and abs(width_pixels - previous_width) <= 1.0
                and polygon_iou(original_polygon, previous["polygon"]) >= 0.95
            ):
                is_near_duplicate = True
                break
        if is_near_duplicate:
            reasons["near_duplicate"] += 1
            continue
        outside_area = rectangle_area - polygon_inside_image_area(
            reconstructed_polygon, width, height
        )
        outside_ratio = max(0.0, outside_area) / max(rectangle_area, 1e-9)
        valid.append(
            {
                "tokens": tokens,
                "parameters_pixels": (
                    center_x,
                    center_y,
                    theta_degrees,
                    width_pixels,
                ),
                "points8": points8,
                "polygon": original_polygon,
                "depth": _depth(points8),
                "roundtrip_iou": polygon_iou(
                    original_polygon, reconstructed_polygon
                ),
                "outside_ratio": min(1.0, outside_ratio),
            }
        )
    return valid, reasons


def _evaluation_candidates(raw_candidates: Any) -> list[tuple[float, ...]]:
    """Preserve every numerically usable official GT rectangle for evaluation."""
    result: list[tuple[float, ...]] = []
    for raw in raw_candidates or []:
        if not isinstance(raw, list | tuple) or len(raw) != 8:
            continue
        try:
            points8 = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in points8):
            result.append(points8)
    return result


def _medoid_and_fps(candidates: list[dict[str, Any]], limit: int) -> list[int]:
    if not candidates:
        return []
    count = len(candidates)
    pairwise = [[0.0] * count for _ in range(count)]
    for left in range(count):
        pairwise[left][left] = 1.0
        for right in range(left + 1, count):
            overlap = polygon_iou(
                candidates[left]["polygon"], candidates[right]["polygon"]
            )
            pairwise[left][right] = overlap
            pairwise[right][left] = overlap
    medoid = max(
        range(count),
        key=lambda index: (
            sum(pairwise[index]) / count,
            tuple(-value for value in candidates[index]["tokens"]),
        ),
    )
    selected = [medoid]
    while len(selected) < min(limit, count):
        remaining = [index for index in range(count) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: (
                min(1.0 - pairwise[index][chosen] for chosen in selected),
                tuple(-value for value in candidates[index]["tokens"]),
            ),
        )
        selected.append(next_index)
    return selected


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": mean(ordered),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if args.minimum_width_diagonal < 0:
        raise ValueError("minimum_width_diagonal must be non-negative")
    if args.gripper_depth_pixels <= 0:
        raise ValueError("gripper_depth_pixels must be positive")
    if args.official_graspnet_eval and args.split == "all":
        raise ValueError(
            "--official-graspnet-eval requires seen, similar, or novel split"
        )

    data_root = args.data_root.expanduser().resolve()
    metadata_dir = (
        args.metadata_dir.expanduser().resolve()
        if args.metadata_dir is not None
        else data_root / "metadata" / args.camera
    )
    metadata_paths = sorted(
        path for path in metadata_dir.rglob("*.json") if _included(path, args)
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    stats: Counter = Counter()
    filter_reasons: Counter = Counter()
    widths: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    angles: list[float] = []
    depths: list[float] = []
    polygon_areas: list[float] = []
    outside_ratios: list[float] = []
    roundtrip_ious: list[float] = []
    candidate_counts: list[float] = []
    sample_ids: list[str] = []
    scene_counts: Counter = Counter()
    with temporary_path.open("w", encoding="utf-8") as output_handle:
        for metadata_path in metadata_paths:
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stats["invalid_metadata_files"] += 1
                continue
            if not isinstance(payload, list):
                stats["invalid_metadata_payloads"] += 1
                continue
            stats["metadata_files"] += 1
            for object_index, obj in enumerate(payload):
                stats["objects"] += 1
                if not isinstance(obj, dict):
                    stats["invalid_objects"] += 1
                    continue
                description = str(
                    obj.get("description") or obj.get("label") or ""
                ).strip()
                if not description:
                    stats["missing_description"] += 1
                    continue
                image_path = _resolve_data_path(data_root, obj.get("image_path"))
                if image_path is None:
                    image_path = _infer_image_path(
                        data_root, metadata_path, args.camera
                    )
                    if image_path is not None and image_path.is_file():
                        stats["inferred_image_paths"] += 1
                if image_path is None or not image_path.is_file():
                    stats["missing_images"] += 1
                    continue
                raw_size = obj.get("image_size_hw")
                if (
                    isinstance(raw_size, list | tuple)
                    and len(raw_size) == 2
                    and all(
                        isinstance(value, int | float) and value > 0
                        for value in raw_size
                    )
                ):
                    height, width = int(raw_size[0]), int(raw_size[1])
                else:
                    with Image.open(image_path) as image:
                        width, height = image.size

                raw_grasps = obj.get("grasps", [])
                if args.official_graspnet_eval and not raw_grasps:
                    stats["objects_without_raw_grasps"] += 1
                    continue
                evaluation_candidates = _evaluation_candidates(raw_grasps)
                candidates, reasons = _valid_candidates(
                    raw_grasps,
                    width,
                    height,
                    args.minimum_width_diagonal,
                    args.gripper_depth_pixels,
                )
                filter_reasons.update(reasons)
                if not candidates and not args.official_graspnet_eval:
                    stats["objects_without_valid_grasps"] += 1
                    continue
                if not candidates:
                    stats["evaluation_only_unrepresentable_samples"] += 1
                selected_indices = _medoid_and_fps(
                    candidates, args.max_candidates
                )
                selected = [candidates[index] for index in selected_indices]
                primary = selected[0]["tokens"] if selected else None
                sample_id = _sample_id(
                    args.dataset_name,
                    metadata_path,
                    image_path,
                    obj.get("object_id", object_index),
                    data_root,
                )
                conversations = [
                    {
                        "from": "human",
                        "value": (
                            "Predict one stable 2D rectangular grasp pose for "
                            f"the target described as: {description}."
                        ),
                    }
                ]
                if primary is not None:
                    answer = "".join(f"<{value}>" for value in primary)
                    conversations.append(
                        {
                            "from": "gpt",
                            "value": (
                                "<ref>grasp pose</ref>"
                                f"<grasp_rect>{answer}</grasp_rect>"
                            ),
                        }
                    )
                sample = {
                    "sample_id": sample_id,
                    "dataset": args.dataset_name,
                    "task_type": "grasp_rect",
                    "image_width": width,
                    "image_height": height,
                    "gripper_depth_pixels": args.gripper_depth_pixels,
                    "grasp_rect_candidates": [
                        list(candidate["tokens"]) for candidate in selected
                    ],
                    "grasp_rect_candidates_pixels": [
                        list(candidate["parameters_pixels"])
                        for candidate in selected
                    ],
                    "grasp_rectangles_pixels": [
                        list(candidate["points8"]) for candidate in selected
                    ],
                    "candidate_collision_2d": [None] * len(selected),
                    "candidate_outside_2d": [
                        candidate["outside_ratio"] for candidate in selected
                    ],
                    "collision_valid": False,
                    "collision_detail": "instance masks not declared exhaustive",
                    "conversations": conversations,
                    "image": _relative_or_absolute(image_path, data_root),
                    "scene": next(
                        (
                            part
                            for part in metadata_path.parts
                            if part.startswith("scene_")
                        ),
                        None,
                    ),
                    "object_id": obj.get("object_id"),
                    "description": description,
                    "evaluation_protocol": (
                        "realvlg_graspnet_official"
                        if args.official_graspnet_eval
                        else None
                    ),
                    "evaluation_only": args.official_graspnet_eval,
                }
                if args.official_graspnet_eval:
                    sample["evaluation_grasp_rectangles_pixels"] = [
                        list(points8) for points8 in evaluation_candidates
                    ]
                output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                sample_ids.append(sample_id)
                scene_counts[str(sample.get("scene"))] += 1
                candidate_counts.append(
                    float(
                        len(evaluation_candidates)
                        if args.official_graspnet_eval
                        else len(candidates)
                    )
                )
                stats["positive_samples"] += 1
                stats["trainable_positive_samples"] += int(bool(selected))
                stats["selected_candidates"] += len(selected)
                stats["evaluation_gt_candidates"] += (
                    len(evaluation_candidates) if args.official_graspnet_eval else 0
                )
                widths.extend(
                    candidate["parameters_pixels"][3] for candidate in candidates
                )
                centers_x.extend(
                    candidate["parameters_pixels"][0] for candidate in candidates
                )
                centers_y.extend(
                    candidate["parameters_pixels"][1] for candidate in candidates
                )
                angles.extend(
                    candidate["parameters_pixels"][2] for candidate in candidates
                )
                depths.extend(candidate["depth"] for candidate in candidates)
                polygon_areas.extend(
                    polygon_area(candidate["polygon"]) for candidate in candidates
                )
                outside_ratios.extend(
                    candidate["outside_ratio"] for candidate in candidates
                )
                roundtrip_ious.extend(
                    candidate["roundtrip_iou"] for candidate in candidates
                )

    sample_id_sha256 = hashlib.sha256(
        ("\n".join(sorted(sample_ids)) + ("\n" if sample_ids else "")).encode(
            "utf-8"
        )
    ).hexdigest()
    expected_hash = getattr(args, "expected_sample_id_sha256", None)
    if expected_hash is not None and sample_id_sha256 != expected_hash:
        raise ValueError(
            "official sample ID hash mismatch: "
            f"expected={expected_hash}, actual={sample_id_sha256}"
        )
    temporary_path.replace(output_path)
    result = {
        "protocol": {
            "realvlg_official_commit": REALVLG_OFFICIAL_COMMIT,
            "sample_id_sha256": sample_id_sha256,
            "scene_counts": dict(sorted(scene_counts.items())),
        },
        "statistics": dict(sorted(stats.items())),
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "geometry": {
            "width_pixels": _summary(widths),
            "center_x_pixels": _summary(centers_x),
            "center_y_pixels": _summary(centers_y),
            "theta_degrees_mod_180": _summary(angles),
            "depth_pixels": _summary(depths),
            "polygon_area_pixels2": _summary(polygon_areas),
            "outside_ratio": _summary(outside_ratios),
            "candidates_per_object": _summary(candidate_counts),
            "roundtrip_iou": _summary(roundtrip_ious),
        },
        "configuration": {
            "dataset_name": args.dataset_name,
            "camera": args.camera,
            "split": args.split,
            "max_candidates": args.max_candidates,
            "minimum_width_diagonal": args.minimum_width_diagonal,
            "gripper_depth_pixels": args.gripper_depth_pixels,
            "official_graspnet_eval": args.official_graspnet_eval,
        },
    }
    if args.stats is not None:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    result = convert(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
