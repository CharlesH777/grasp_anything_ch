#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from locate_anything_service.collision_2d import evaluate_collision_2d  # noqa: E402
from locate_anything_service.grasp_geometry import (  # noqa: E402
    derive_grasp_geometry,
    grasp_rectangle,
    polygon_area,
    polygon_inside_image_area,
    polygon_iou,
)

SPLIT_RANGES = {
    "train": range(0, 100),
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RealVLG contact annotations to LocateAnything JSONL."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument(
        "--split", choices=("all", *SPLIT_RANGES), default="all"
    )
    parser.add_argument("--scene-start", type=int)
    parser.add_argument("--scene-end-exclusive", type=int)
    parser.add_argument("--camera")
    parser.add_argument(
        "--official-graspnet-eval",
        action="store_true",
        help=(
            "Match RealVLG GraspNet evaluation: kinect, test split, and only "
            "scene_xxxx/0000.json."
        ),
    )
    parser.add_argument(
        "--official-graspnet-all-frames",
        action="store_true",
        help=(
            "Keep the official GraspNet evaluation semantics but include every "
            "kinect frame instead of only scene_xxxx/0000.json."
        ),
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--rectangle-thickness", type=float, default=80.0)
    parser.add_argument("--min-width-diagonal", type=float, default=1e-4)
    parser.add_argument("--max-width-diagonal", type=float, default=1.0)
    parser.add_argument("--collision-threshold", type=float, default=0.0)
    parser.add_argument("--outside-threshold", type=float, default=0.0)
    parser.add_argument(
        "--collision-masks-exhaustive",
        action="store_true",
        help="Only enable when metadata contains every obstacle instance.",
    )
    parser.add_argument(
        "--grasp-candidates-exhaustive",
        action="store_true",
        help=(
            "Allow all reliably unsafe candidates to produce a no-grasp label; "
            "requires exhaustive obstacle masks and grasp annotations."
        ),
    )
    parser.add_argument("--derived-mask-dir", type=Path)
    return parser.parse_args()


def _scene_id(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("scene_"):
            try:
                return int(part.removeprefix("scene_"))
            except ValueError:
                return None
    return None


def _matches_split(path: Path, split: str) -> bool:
    if split == "all":
        return True
    scene_id = _scene_id(path)
    return scene_id is not None and scene_id in SPLIT_RANGES[split]


def _resolve_data_path(data_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


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
) -> str:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]
    scene = next(
        (part for part in metadata_path.parts if part.startswith("scene_")),
        metadata_path.parent.name,
    )
    return f"{dataset_name}:{scene}:{digest}:{object_id}"


def _canonicalize(candidate: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = candidate
    return candidate if (x1, y1) <= (x2, y2) else (x2, y2, x1, y1)


def _quantize(
    candidate: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = candidate
    values = (
        round(x1 * 1000 / width),
        round(y1 * 1000 / height),
        round(x2 * 1000 / width),
        round(y2 * 1000 / height),
    )
    clipped = tuple(max(0, min(1000, value)) for value in values)
    return _canonicalize(clipped)


def _valid_pixel_candidates(
    raw_candidates: Any,
    width: int,
    height: int,
    min_width_diagonal: float,
    max_width_diagonal: float,
) -> tuple[list[tuple[float, float, float, float]], Counter]:
    valid: list[tuple[float, float, float, float]] = []
    reasons: Counter = Counter()
    diagonal = math.hypot(width, height)
    seen: set[tuple[float, float, float, float]] = set()
    for raw in raw_candidates or []:
        if not isinstance(raw, list | tuple) or len(raw) != 4:
            reasons["bad_shape"] += 1
            continue
        try:
            candidate = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            reasons["non_numeric"] += 1
            continue
        if not all(math.isfinite(value) for value in candidate):
            reasons["non_finite"] += 1
            continue
        x1, y1, x2, y2 = candidate
        if not (0 <= x1 <= width and 0 <= x2 <= width):
            reasons["x_out_of_bounds"] += 1
            continue
        if not (0 <= y1 <= height and 0 <= y2 <= height):
            reasons["y_out_of_bounds"] += 1
            continue
        width_diagonal = math.hypot(x2 - x1, y2 - y1) / diagonal
        if width_diagonal <= min_width_diagonal:
            reasons["too_narrow"] += 1
            continue
        if width_diagonal > max_width_diagonal:
            reasons["too_wide"] += 1
            continue
        key = tuple(round(value, 4) for value in candidate)
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        valid.append(candidate)
    return valid, reasons


def _rectangles(
    candidates: list[tuple[float, float, float, float]], thickness: float
):
    return [grasp_rectangle(candidate, thickness) for candidate in candidates]


def _iou_medoid_and_fps(
    candidates: list[tuple[float, float, float, float]],
    max_candidates: int,
    thickness: float,
) -> list[tuple[float, float, float, float]]:
    if not candidates:
        return []
    rectangles = _rectangles(candidates, thickness)
    count = len(candidates)
    pairwise = [[0.0] * count for _ in range(count)]
    for left in range(count):
        pairwise[left][left] = 1.0
        for right in range(left + 1, count):
            overlap = polygon_iou(rectangles[left], rectangles[right])
            pairwise[left][right] = overlap
            pairwise[right][left] = overlap

    medoid = max(
        range(count),
        key=lambda index: (
            sum(pairwise[index]) / count,
            tuple(-value for value in candidates[index]),
        ),
    )
    selected = [medoid]
    remaining = set(range(count)) - {medoid}
    while remaining and len(selected) < max_candidates:
        next_index = max(
            remaining,
            key=lambda index: (
                min(1.0 - pairwise[index][chosen] for chosen in selected),
                tuple(-value for value in candidates[index]),
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return [candidates[index] for index in selected]


def _load_mask(path: Path | None, size: tuple[int, int]) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as source:
        mask = source.convert("L")
    if mask.size != size:
        raise ValueError(f"mask {path} has size {mask.size}, expected {size}")
    return mask.point(lambda value: 255 if value > 0 else 0)


def _build_obstacle_mask(
    target_index: int,
    group: list[dict[str, Any]],
    data_root: Path,
    size: tuple[int, int],
) -> tuple[Image.Image | None, bool, str | None]:
    masks: list[Image.Image] = []
    for index, entry in enumerate(group):
        if index == target_index:
            continue
        path = _resolve_data_path(data_root, entry["object"].get("mask_path"))
        try:
            mask = _load_mask(path, size)
        except ValueError as error:
            return None, False, str(error)
        if mask is None:
            return None, False, f"missing obstacle mask: {path}"
        masks.append(mask)

    obstacle = Image.new("L", size, 0)
    for mask in masks:
        obstacle = ImageChops.lighter(obstacle, mask)
    return obstacle, True, None


def _load_records(
    metadata_dir: Path,
    data_root: Path,
    split: str,
    camera: str | None,
    official_graspnet_eval: bool = False,
    official_graspnet_all_frames: bool = False,
    scene_start: int | None = None,
    scene_end_exclusive: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    record_keys: dict[tuple[Path, str], dict[str, Any]] = {}
    for metadata_path in sorted(metadata_dir.rglob("*.json")):
        if not _matches_split(metadata_path, split):
            continue
        scene_id = _scene_id(metadata_path)
        if scene_start is not None and (
            scene_id is None or scene_id < scene_start
        ):
            continue
        if scene_end_exclusive is not None and (
            scene_id is None or scene_id >= scene_end_exclusive
        ):
            continue
        if camera and camera not in metadata_path.parts:
            continue
        if (
            official_graspnet_eval
            and not official_graspnet_all_frames
            and metadata_path.name != "0000.json"
        ):
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for object_index, object_data in enumerate(payload):
            if not isinstance(object_data, dict):
                continue
            image_path = _resolve_data_path(data_root, object_data.get("image_path"))
            if image_path is None:
                continue
            object_id = object_data.get("object_id")
            identity = (
                str(object_id)
                if object_id is not None
                else f"{metadata_path}:{object_index}"
            )
            key = (image_path, identity)
            existing = record_keys.get(key)
            if existing is not None:
                if existing["object"] != object_data:
                    raise ValueError(
                        "conflicting duplicate metadata for image/object: "
                        f"{image_path} / {identity}"
                    )
                continue
            record = {
                "metadata_path": metadata_path,
                "image_path": image_path,
                "object": object_data,
            }
            record_keys[key] = record
            records.append(record)
    return records


def convert(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.expanduser().resolve()
    metadata_dir = (
        args.metadata_dir.expanduser().resolve()
        if args.metadata_dir
        else data_root / "metadata"
    )
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"metadata directory not found: {metadata_dir}")
    if args.max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    outside_threshold = float(getattr(args, "outside_threshold", 0.0))
    if not 0.0 <= args.collision_threshold <= 1.0:
        raise ValueError("collision_threshold must be in [0, 1]")
    if not 0.0 <= outside_threshold <= 1.0:
        raise ValueError("outside_threshold must be in [0, 1]")
    grasp_candidates_exhaustive = bool(
        getattr(args, "grasp_candidates_exhaustive", False)
    )
    if grasp_candidates_exhaustive and not args.collision_masks_exhaustive:
        raise ValueError(
            "--grasp-candidates-exhaustive requires "
            "--collision-masks-exhaustive"
        )
    dataset_name = args.dataset_name or data_root.name

    official_graspnet_eval = bool(
        getattr(args, "official_graspnet_eval", False)
    )
    official_graspnet_all_frames = bool(
        getattr(args, "official_graspnet_all_frames", False)
    )
    if official_graspnet_all_frames and not official_graspnet_eval:
        raise ValueError(
            "--official-graspnet-all-frames requires --official-graspnet-eval"
        )
    scene_start = getattr(args, "scene_start", None)
    scene_end_exclusive = getattr(args, "scene_end_exclusive", None)
    if (scene_start is None) != (scene_end_exclusive is None):
        raise ValueError(
            "--scene-start and --scene-end-exclusive must be used together"
        )
    if scene_start is not None and scene_start >= scene_end_exclusive:
        raise ValueError("scene range must be non-empty")
    camera = args.camera
    if official_graspnet_eval:
        if args.split not in {"seen", "similar", "novel"}:
            raise ValueError(
                "--official-graspnet-eval requires seen, similar, or novel split"
            )
        if camera not in {None, "kinect"}:
            raise ValueError(
                "--official-graspnet-eval requires --camera kinect"
            )
        camera = "kinect"

    records = _load_records(
        metadata_dir,
        data_root,
        args.split,
        camera,
        official_graspnet_eval=official_graspnet_eval,
        official_graspnet_all_frames=official_graspnet_all_frames,
        scene_start=scene_start,
        scene_end_exclusive=scene_end_exclusive,
    )
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["image_path"]].append(record)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    derived_mask_dir = (
        args.derived_mask_dir.expanduser().resolve()
        if args.derived_mask_dir
        else output_path.parent / f"{output_path.stem}_obstacle_masks"
    )
    if args.collision_masks_exhaustive:
        derived_mask_dir.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    filter_reasons: Counter = Counter()
    with temporary_path.open("w", encoding="utf-8") as output_handle:
        for image_path, group in sorted(grouped.items(), key=lambda item: str(item[0])):
            if not image_path.is_file():
                stats["missing_images"] += len(group)
                continue
            raw_size = group[0]["object"].get("image_size_hw")
            if (
                isinstance(raw_size, list | tuple)
                and len(raw_size) == 2
                and all(
                    isinstance(value, int | float) and value > 0
                    for value in raw_size
                )
            ):
                height, width = (int(raw_size[0]), int(raw_size[1]))
                stats["images_sized_from_metadata"] += 1
            else:
                with Image.open(image_path) as source:
                    width, height = source.size
                stats["images_sized_from_file"] += 1
            stats["images"] += 1

            for target_index, entry in enumerate(group):
                obj = entry["object"]
                if official_graspnet_eval and not obj.get("grasps"):
                    stats["official_targets_without_grasps"] += 1
                    continue
                description = str(
                    obj.get("description") or obj.get("label") or ""
                ).strip()
                if not description:
                    stats["missing_description"] += 1
                    continue
                candidates, reasons = _valid_pixel_candidates(
                    obj.get("contact_points", []),
                    width,
                    height,
                    args.min_width_diagonal,
                    args.max_width_diagonal,
                )
                filter_reasons.update(reasons)
                if not candidates:
                    stats["objects_without_valid_contacts"] += 1
                    continue
                all_quantized: list[tuple[int, int, int, int]] = []
                all_pixels_after_quantization: list[
                    tuple[float, float, float, float]
                ] = []
                seen_quantized: set[tuple[int, int, int, int]] = set()
                for pixel_candidate in candidates:
                    token_candidate = _quantize(pixel_candidate, width, height)
                    if token_candidate in seen_quantized:
                        filter_reasons["quantized_duplicate"] += 1
                        continue
                    if token_candidate[:2] == token_candidate[2:]:
                        filter_reasons["quantized_degenerate"] += 1
                        continue
                    seen_quantized.add(token_candidate)
                    all_quantized.append(token_candidate)
                    all_pixels_after_quantization.append(pixel_candidate)
                if not all_quantized:
                    stats["objects_degenerate_after_quantization"] += 1
                    continue

                collision_valid = False
                collision_detail = "instance masks not declared exhaustive"
                obstacle_path: Path | None = None
                collision_scores = [0.0] * len(all_quantized)
                outside_scores: list[float] = []
                continuous_geometries = []
                for candidate in all_pixels_after_quantization:
                    x1, y1, x2, y2 = candidate
                    continuous_token_candidate = (
                        x1 * 1000.0 / width,
                        y1 * 1000.0 / height,
                        x2 * 1000.0 / width,
                        y2 * 1000.0 / height,
                    )
                    geometry = derive_grasp_geometry(
                        continuous_token_candidate, width, height
                    )
                    continuous_geometries.append(geometry)
                    polygon = grasp_rectangle(
                        geometry.contacts_pixels_float,
                        args.rectangle_thickness,
                    )
                    full_area = polygon_area(polygon)
                    inside_area = polygon_inside_image_area(
                        polygon, width, height
                    )
                    outside_scores.append(
                        min(
                            1.0,
                            max(0.0, (full_area - inside_area) / full_area),
                        )
                    )
                if args.collision_masks_exhaustive:
                    obstacle, collision_valid, collision_detail = _build_obstacle_mask(
                        target_index, group, data_root, (width, height)
                    )
                    if obstacle is not None and collision_valid:
                        sample_stub = _sample_id(
                            dataset_name,
                            entry["metadata_path"],
                            image_path,
                            obj.get("object_id", target_index),
                        ).replace(":", "_")
                        obstacle_path = derived_mask_dir / f"{sample_stub}.png"
                        obstacle.save(obstacle_path)
                        collision_scores = []
                        for geometry in continuous_geometries:
                            collision = evaluate_collision_2d(
                                geometry,
                                obstacle,
                                width,
                                height,
                                thickness_pixels=args.rectangle_thickness,
                                collision_threshold=args.collision_threshold,
                                outside_threshold=outside_threshold,
                            )
                            collision_scores.append(collision.collision_ratio or 0.0)

                safe_indices = [
                    index
                    for index in range(len(all_quantized))
                    if outside_scores[index] <= outside_threshold
                    and (
                        not collision_valid
                        or collision_scores[index] <= args.collision_threshold
                    )
                ]
                has_safe_candidate = bool(safe_indices)
                if not has_safe_candidate and official_graspnet_eval:
                    stats["official_samples_without_safe_training_candidate"] += 1
                elif not has_safe_candidate and not grasp_candidates_exhaustive:
                    stats["objects_without_safe_annotated_candidate"] += 1
                    continue

                no_safe_candidate = (
                    not has_safe_candidate and not official_graspnet_eval
                )
                if no_safe_candidate:
                    quantized: list[tuple[int, int, int, int]] = []
                    selected_pixels_after_quantization = (
                        all_pixels_after_quantization
                    )
                else:
                    selectable_indices = safe_indices or list(
                        range(len(all_quantized))
                    )
                    selectable_pixels = [
                        all_pixels_after_quantization[index]
                        for index in selectable_indices
                    ]
                    selected_pixels = _iou_medoid_and_fps(
                        selectable_pixels,
                        args.max_candidates,
                        args.rectangle_thickness,
                    )
                    selected_indices = [
                        selectable_indices[selectable_pixels.index(candidate)]
                        for candidate in selected_pixels
                    ]
                    quantized = [all_quantized[index] for index in selected_indices]
                    selected_pixels_after_quantization = [
                        all_pixels_after_quantization[index]
                        for index in selected_indices
                    ]
                    collision_scores = [
                        collision_scores[index] for index in selected_indices
                    ]
                    outside_scores = [
                        outside_scores[index] for index in selected_indices
                    ]

                sample_id = _sample_id(
                    dataset_name,
                    entry["metadata_path"],
                    image_path,
                    obj.get("object_id", target_index),
                )
                if no_safe_candidate:
                    sample_task_type = "grasp_contact_negative"
                    answer_markup = "<grasp>none</grasp>"
                else:
                    primary = quantized[0]
                    answer = "".join(f"<{value}>" for value in primary)
                    sample_task_type = "grasp_contact"
                    answer_markup = f"<grasp>{answer}</grasp>"
                target_mask = _resolve_data_path(data_root, obj.get("mask_path"))
                sample = {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "task_type": sample_task_type,
                    "negative_reason": (
                        "ungraspable" if no_safe_candidate else None
                    ),
                    "image_width": width,
                    "image_height": height,
                    "contact_candidates": [list(candidate) for candidate in quantized],
                    "contact_candidates_pixels": [
                        list(candidate)
                        for candidate in selected_pixels_after_quantization
                    ],
                    "candidate_collision_2d": collision_scores,
                    "candidate_outside_2d": outside_scores,
                    "collision_valid": collision_valid,
                    "collision_detail": collision_detail,
                    "target_mask": (
                        _relative_or_absolute(target_mask, data_root)
                        if target_mask is not None
                        else None
                    ),
                    "obstacle_mask": str(obstacle_path) if obstacle_path else None,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "Predict one plausible two-finger 2D contact pair "
                                f"for the target described as: {description}."
                            ),
                        },
                        {
                            "from": "gpt",
                            "value": (
                                f"<ref>grasp</ref>{answer_markup}"
                            ),
                        },
                    ],
                    "image": _relative_or_absolute(image_path, data_root),
                    "scene": next(
                        (
                            part
                            for part in entry["metadata_path"].parts
                            if part.startswith("scene_")
                        ),
                        None,
                    ),
                    "object_id": obj.get("object_id"),
                    "description": description,
                    "evaluation_protocol": (
                        "realvlg_graspnet_official"
                        if official_graspnet_eval
                        else None
                    ),
                    "evaluation_only": official_graspnet_eval,
                }
                if official_graspnet_eval:
                    sample["evaluation_contact_candidates_pixels"] = [
                        list(candidate) for candidate in candidates
                    ]
                    stats["evaluation_gt_candidates"] += len(candidates)
                output_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["positive_samples"] += int(not no_safe_candidate)
                stats["negative_samples"] += int(no_safe_candidate)
                stats["selected_candidates"] += len(quantized)
                if collision_valid:
                    stats["collision_valid_samples"] += 1

    temporary_path.replace(output_path)
    stats["source_records"] = len(records)
    stats["source_images"] = len(grouped)
    result = {
        "statistics": dict(sorted(stats.items())),
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "configuration": {
            "dataset_name": dataset_name,
            "split": args.split,
            "camera": camera,
            "official_graspnet_eval": official_graspnet_eval,
            "official_graspnet_all_frames": official_graspnet_all_frames,
            "max_candidates": args.max_candidates,
            "rectangle_thickness": args.rectangle_thickness,
            "outside_threshold": outside_threshold,
            "collision_masks_exhaustive": args.collision_masks_exhaustive,
            "grasp_candidates_exhaustive": grasp_candidates_exhaustive,
            "scene_start": scene_start,
            "scene_end_exclusive": scene_end_exclusive,
        },
    }
    if args.stats:
        stats_path = args.stats.expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    args = parse_args()
    result = convert(args)
    print(json.dumps(result["statistics"], ensure_ascii=False, sort_keys=True))
    print(f"Wrote: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
