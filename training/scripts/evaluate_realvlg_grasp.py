#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from locate_anything_service.grasp_geometry import (  # noqa: E402
    polygon_area,
    polygon_inside_image_area,
    polygon_iou,
)
from locate_anything_service.grasp_rect_geometry import (  # noqa: E402
    DEFAULT_GRIPPER_DEPTH_PIXELS,
    DEFAULT_MINIMUM_WIDTH_DIAGONAL,
    canonical_angle_degrees,
    derive_grasp_rectangle_geometry,
    encode_grasp_rectangle_pixels,
    points8_to_rect,
    rect_to_points8,
)
from locate_anything_service.parser import parse_grasp_rect_output  # noqa: E402

_REALVLG_RECT_PATTERN = re.compile(
    r"\(\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
    r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
)
REALVLG_OFFICIAL_COMMIT = "040562e0cf8f64a8c6e922d8f7e5e098bb3633c3"


@dataclass(frozen=True, slots=True)
class DecodedPrediction:
    status: str
    error: str | None
    parameters_pixels: tuple[float, float, float, float] | None
    points8: tuple[float, ...] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict offline evaluation for RealVLG rectangular grasps."
    )
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--generation-mode",
        choices=("fast", "slow", "hybrid"),
        default="fast",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--grasp-rect-coord-mass-threshold", type=float, default=1e-4
    )
    parser.add_argument(
        "--grasp-rect-coord-entropy-threshold", type=float, default=1.0
    )
    parser.add_argument("--grasp-rect-keep-k", type=int, default=4)
    parser.add_argument(
        "--prediction-format",
        choices=("locateanything", "realvlg"),
        default="locateanything",
    )
    parser.add_argument(
        "--gripper-depth-pixels",
        type=float,
        default=DEFAULT_GRIPPER_DEPTH_PIXELS,
    )
    parser.add_argument(
        "--minimum-width-diagonal",
        type=float,
        default=DEFAULT_MINIMUM_WIDTH_DIAGONAL,
    )
    return parser.parse_args()


def _decode_generation_output(output: Any, input_ids: Any, processor: Any) -> str:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and isinstance(output[0], str):
        return output[0]
    try:
        import torch
    except ImportError:
        return str(output)
    if torch.is_tensor(output):
        generated = output[:, input_ids.shape[1] :].detach().cpu()
        decoded = processor.tokenizer.batch_decode(
            generated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]
    return str(output)


class ModelPredictor:
    def __init__(
        self,
        model_path: Path,
        generation_mode: str,
        max_new_tokens: int,
        coord_mass_threshold: float,
        coord_entropy_threshold: float,
        keep_k: int,
        minimum_width_diagonal: float,
        gripper_depth_pixels: float,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.coord_mass_threshold = coord_mass_threshold
        self.coord_entropy_threshold = coord_entropy_threshold
        self.keep_k = keep_k
        self.minimum_width_diagonal = minimum_width_diagonal
        self.gripper_depth_pixels = gripper_depth_pixels
        self.model = (
            AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to("cuda")
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=True
        )

    def __call__(
        self, image_path: Path, description: str
    ) -> tuple[str, float, dict[str, Any]]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prompt = (
            "Predict one stable 2D rectangular grasp pose for the target "
            f"described as: {description}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to("cuda")
        kwargs: dict[str, Any] = {
            "pixel_values": inputs["pixel_values"].to(
                "cuda", dtype=self.torch.bfloat16
            ),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "image_grid_hws": inputs.get("image_grid_hws"),
            "tokenizer": self.processor.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "do_sample": False,
            "generation_mode": self.generation_mode,
            "geometry_type": "grasp_rect",
            "image_size": image.size,
            "grasp_rect_minimum_width_diagonal": self.minimum_width_diagonal,
            "grasp_rect_gripper_depth_pixels": self.gripper_depth_pixels,
            "grasp_rect_coord_mass_threshold": self.coord_mass_threshold,
            "grasp_rect_coord_entropy_threshold": self.coord_entropy_threshold,
            "grasp_rect_keep_k": self.keep_k,
        }
        if self.generation_mode in {"fast", "hybrid"}:
            kwargs["n_future_tokens"] = 6
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**kwargs)
        latency = time.perf_counter() - started
        stats = output[1] if isinstance(output, tuple) and len(output) > 1 else {}
        return (
            _decode_generation_output(output, inputs["input_ids"], self.processor),
            latency,
            stats if isinstance(stats, dict) else {},
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("prediction row is missing sample_id")
        if sample_id in result:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        result[sample_id] = row
    return result


def _validate_pixel_parameters(
    parameters: tuple[float, float, float, float],
    width: int,
    height: int,
    minimum_width_diagonal: float,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("grasp rectangle parameters must be finite")
    center_x, center_y, theta_degrees, width_pixels = parameters
    if not 0.0 <= center_x <= width or not 0.0 <= center_y <= height:
        raise ValueError("grasp rectangle center must be inside the image")
    if width_pixels / math.hypot(width, height) <= minimum_width_diagonal:
        raise ValueError("grasp rectangle width must exceed minimum_width_diagonal")
    return (
        center_x,
        center_y,
        canonical_angle_degrees(theta_degrees),
        width_pixels,
    )


def decode_prediction(
    raw_output: str,
    width: int,
    height: int,
    prediction_format: str,
    *,
    gripper_depth_pixels: float = DEFAULT_GRIPPER_DEPTH_PIXELS,
    minimum_width_diagonal: float = DEFAULT_MINIMUM_WIDTH_DIAGONAL,
) -> DecodedPrediction:
    if prediction_format == "locateanything":
        parsed = parse_grasp_rect_output(
            raw_output,
            width,
            height,
            gripper_depth_pixels=gripper_depth_pixels,
            minimum_width_diagonal=minimum_width_diagonal,
        )
        if parsed.status != "ok":
            return DecodedPrediction(parsed.status, parsed.error, None, None)
        rectangle = parsed.rectangles[0]
        parameters = (
            rectangle.center_pixels_float[0],
            rectangle.center_pixels_float[1],
            rectangle.angle_degrees_image,
            rectangle.opening_width_pixels,
        )
        return DecodedPrediction(
            "ok",
            None,
            parameters,
            rectangle.rectangle_points_pixels_float,
        )

    answer_match = re.search(
        r"<answer>(.*?)</answer>", raw_output, flags=re.DOTALL | re.IGNORECASE
    )
    content = answer_match.group(1).strip() if answer_match else raw_output.strip()
    match = _REALVLG_RECT_PATTERN.fullmatch(content)
    if match is None:
        return DecodedPrediction(
            "invalid",
            "expected RealVLG (x, y, theta, width) output",
            None,
            None,
        )
    try:
        parameters = _validate_pixel_parameters(
            tuple(float(value) for value in match.groups()),
            width,
            height,
            minimum_width_diagonal,
        )
        points8 = rect_to_points8(
            *parameters,
            gripper_depth_pixels=gripper_depth_pixels,
        )
    except ValueError as error:
        return DecodedPrediction("invalid", str(error), None, None)
    return DecodedPrediction("ok", None, parameters, points8)


def corrected_angle_error_degrees(first: float, second: float) -> float:
    delta = (first - second) % 180.0
    return min(delta, 180.0 - delta)


def official_buggy_angle_error_degrees(first: float, second: float) -> float:
    if abs(first) <= math.pi and abs(second) <= math.pi:
        first_radians, second_radians = first, second
    else:
        first_radians, second_radians = math.radians(first), math.radians(second)
    cosine = max(
        -1.0,
        min(
            1.0,
            math.cos(first_radians) * math.cos(second_radians)
            + math.sin(first_radians) * math.sin(second_radians),
        ),
    )
    difference = math.degrees(math.acos(cosine))
    return 180.0 - difference if difference > 90.0 else difference


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def _corner_error_with_polygon_symmetry(
    predicted: tuple[float, ...], target: tuple[float, ...]
) -> float:
    pred = [(predicted[index], predicted[index + 1]) for index in range(0, 8, 2)]
    gt = [(target[index], target[index + 1]) for index in range(0, 8, 2)]
    candidates = []
    for sequence in (gt, list(reversed(gt))):
        for offset in range(4):
            aligned = sequence[offset:] + sequence[:offset]
            candidates.append(
                mean(
                    math.hypot(left[0] - right[0], left[1] - right[1])
                    for left, right in zip(pred, aligned, strict=True)
                )
            )
    return min(candidates)


def _outside_ratio(points8: tuple[float, ...], width: int, height: int) -> float:
    polygon = tuple(
        (points8[index], points8[index + 1]) for index in range(0, 8, 2)
    )
    area = polygon_area(polygon)
    if area <= 1e-9:
        return 1.0
    inside = polygon_inside_image_area(polygon, width, height)
    return max(0.0, min(1.0, (area - inside) / area))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    annotations = [
        row
        for path in args.annotations
        for row in _load_jsonl(path.expanduser().resolve())
    ]
    prediction_path = getattr(args, "predictions", None)
    model_path = getattr(args, "model_path", None)
    predictions = (
        _prediction_map(prediction_path.expanduser().resolve())
        if prediction_path is not None
        else {}
    )
    predictor = None
    data_root = None
    if model_path is not None:
        raw_data_root = getattr(args, "data_root", None)
        if raw_data_root is None:
            raise ValueError("--data-root is required with --model-path")
        data_root = raw_data_root.expanduser().resolve()
        predictor = ModelPredictor(
            model_path.expanduser().resolve(),
            getattr(args, "generation_mode", "fast"),
            getattr(args, "max_new_tokens", 128),
            getattr(args, "grasp_rect_coord_mass_threshold", 1e-4),
            getattr(args, "grasp_rect_coord_entropy_threshold", 1.0),
            getattr(args, "grasp_rect_keep_k", 4),
            args.minimum_width_diagonal,
            args.gripper_depth_pixels,
        )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positive_count = 0
    format_valid_count = 0
    grasp_output_count = 0
    width_valid_count = 0
    complete_six_slot_count = 0
    valid_ious: list[float] = []
    strict_iou_sum = 0.0
    representation_oracle_iou_sum = 0.0
    corrected_valid: list[int] = []
    corrected_strict_sum = 0
    buggy_valid: list[int] = []
    seam_corrected: list[int] = []
    center_errors: list[float] = []
    angle_errors: list[float] = []
    width_errors: list[float] = []
    coordinate_top1_correct = 0
    center_errors_normalized: list[float] = []
    width_errors_normalized: list[float] = []
    corner_errors: list[float] = []
    outside_ratios: list[float] = []
    geometry_invalid_count = 0
    structure_invalid_count = 0
    false_none_count = 0
    negative_count = 0
    true_none_count = 0
    predicted_none_count = 0
    fallback_count = 0
    generation_latencies: list[float] = []
    results: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as output_handle:
        for annotation in annotations:
            task_type = annotation.get("task_type")
            if task_type not in {"grasp_rect", "grasp_rect_negative"}:
                continue
            is_positive = task_type == "grasp_rect"
            if not is_positive:
                negative_count += 1
            else:
                positive_count += 1
            sample_id = str(annotation.get("sample_id", ""))
            width = int(annotation["image_width"])
            height = int(annotation["image_height"])
            if predictor is not None:
                assert data_root is not None
                image_path = _resolve(data_root, annotation.get("image"))
                if image_path is None:
                    raise ValueError(f"sample {sample_id} has no image")
                raw_output, latency, generation_stats = predictor(
                    image_path, str(annotation.get("description", ""))
                )
                generation_stats = {
                    "generation_seconds": latency,
                    "hybrid_fallback": bool(
                        generation_stats.get("hybrid_fallback", False)
                    ),
                }
                prediction_row: dict[str, Any] = {}
            else:
                prediction_row = predictions.get(sample_id, {})
                generation_stats = prediction_row.get("generation_stats") or {}
                latency = generation_stats.get(
                    "generation_seconds", prediction_row.get("generation_seconds")
                )
                raw_output = str(prediction_row.get("raw_output", ""))
            if isinstance(latency, int | float) and math.isfinite(float(latency)):
                generation_latencies.append(float(latency))
            fallback_count += int(
                bool(
                    generation_stats.get("hybrid_fallback")
                    or prediction_row.get("hybrid_fallback")
                )
            )
            gt_candidates = (
                annotation.get("evaluation_grasp_rectangles_pixels")
                or annotation.get("grasp_rectangles_pixels")
                or []
            )
            primary_tokens = (annotation.get("grasp_rect_candidates") or [None])[0]
            if is_positive and primary_tokens is not None:
                try:
                    oracle_geometry = derive_grasp_rectangle_geometry(
                        primary_tokens,
                        width,
                        height,
                        gripper_depth_pixels=args.gripper_depth_pixels,
                        minimum_width_diagonal=args.minimum_width_diagonal,
                    )
                    oracle_polygon = tuple(
                        (
                            oracle_geometry.rectangle_points_pixels_float[index],
                            oracle_geometry.rectangle_points_pixels_float[index + 1],
                        )
                        for index in range(0, 8, 2)
                    )
                    oracle_overlaps = []
                    for raw_gt in gt_candidates:
                        gt_points = tuple(float(value) for value in raw_gt)
                        if len(gt_points) != 8:
                            continue
                        oracle_overlaps.append(
                            polygon_iou(
                                oracle_polygon,
                                tuple(
                                    (gt_points[index], gt_points[index + 1])
                                    for index in range(0, 8, 2)
                                ),
                            )
                        )
                    representation_oracle_iou_sum += max(
                        oracle_overlaps, default=0.0
                    )
                except (TypeError, ValueError):
                    pass
            decoded = decode_prediction(
                raw_output,
                width,
                height,
                args.prediction_format,
                gripper_depth_pixels=args.gripper_depth_pixels,
                minimum_width_diagonal=args.minimum_width_diagonal,
            )
            if is_positive:
                format_valid_count += int(
                    not decoded.status.startswith("invalid")
                )
                geometry_invalid_count += int(
                    decoded.status == "invalid_geometry"
                )
                structure_invalid_count += int(
                    decoded.status in {"invalid", "invalid_structure"}
                )
            result: dict[str, Any] = {
                "sample_id": sample_id,
                "raw_output": raw_output,
                "status": decoded.status,
                "parse_error": decoded.error,
                "generation_stats": generation_stats,
                "iou": 0.0,
                "gacc_corrected": 0,
                "gacc_official_buggy": 0,
            }
            if decoded.status == "none":
                predicted_none_count += 1
                if is_positive:
                    false_none_count += 1
                else:
                    true_none_count += 1
            if not is_positive:
                output_handle.write(json.dumps(result) + "\n")
                results.append(result)
                continue
            if decoded.status != "ok":
                output_handle.write(json.dumps(result) + "\n")
                results.append(result)
                continue

            grasp_output_count += 1
            width_valid_count += 1
            complete_six_slot_count += 1
            assert decoded.parameters_pixels is not None
            assert decoded.points8 is not None
            best_iou = -1.0
            best_gt_points: tuple[float, ...] | None = None
            best_gt_parameters: tuple[float, float, float, float] | None = None
            for raw_gt in gt_candidates:
                try:
                    gt_points = tuple(float(value) for value in raw_gt)
                    gt_parameters = points8_to_rect(gt_points)
                    overlap = polygon_iou(
                        tuple(
                            (decoded.points8[index], decoded.points8[index + 1])
                            for index in range(0, 8, 2)
                        ),
                        tuple(
                            (gt_points[index], gt_points[index + 1])
                            for index in range(0, 8, 2)
                        ),
                    )
                except (TypeError, ValueError):
                    continue
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_points = gt_points
                    best_gt_parameters = gt_parameters
            if best_gt_points is None or best_gt_parameters is None:
                result["evaluation_error"] = "sample has no valid GT rectangles"
                output_handle.write(json.dumps(result) + "\n")
                results.append(result)
                continue

            pred_x, pred_y, pred_theta, pred_width = decoded.parameters_pixels
            if primary_tokens is not None:
                try:
                    predicted_tokens = encode_grasp_rectangle_pixels(
                        pred_x,
                        pred_y,
                        pred_theta,
                        pred_width,
                        width,
                        height,
                        minimum_width_diagonal=args.minimum_width_diagonal,
                    )
                    coordinate_top1_correct += sum(
                        int(predicted == int(target))
                        for predicted, target in zip(
                            predicted_tokens, primary_tokens, strict=True
                        )
                    )
                except (TypeError, ValueError):
                    pass
            gt_x, gt_y, gt_theta, gt_width = best_gt_parameters
            corrected_angle = corrected_angle_error_degrees(
                pred_theta, gt_theta
            )
            buggy_angle = official_buggy_angle_error_degrees(pred_theta, gt_theta)
            corrected = int(best_iou > 0.25 and corrected_angle < 30.0)
            buggy = int(best_iou > 0.25 and buggy_angle < 30.0)
            center_error = math.hypot(pred_x - gt_x, pred_y - gt_y)
            width_error = abs(pred_width - gt_width)
            image_diagonal = math.hypot(width, height)
            center_error_normalized = center_error / image_diagonal
            width_error_normalized = width_error / image_diagonal
            corner_error = _corner_error_with_polygon_symmetry(
                decoded.points8, best_gt_points
            )
            outside_ratio = _outside_ratio(decoded.points8, width, height)

            valid_ious.append(best_iou)
            strict_iou_sum += best_iou
            corrected_valid.append(corrected)
            corrected_strict_sum += corrected
            buggy_valid.append(buggy)
            if min(gt_theta % 180.0, 180.0 - (gt_theta % 180.0)) <= 5.0:
                seam_corrected.append(corrected)
            center_errors.append(center_error)
            angle_errors.append(corrected_angle)
            width_errors.append(width_error)
            center_errors_normalized.append(center_error_normalized)
            width_errors_normalized.append(width_error_normalized)
            corner_errors.append(corner_error)
            outside_ratios.append(outside_ratio)
            result.update(
                {
                    "prediction_parameters_pixels": list(
                        decoded.parameters_pixels
                    ),
                    "prediction_points8_pixels": list(decoded.points8),
                    "matched_gt_points8_pixels": list(best_gt_points),
                    "iou": best_iou,
                    "angle_error_corrected_degrees": corrected_angle,
                    "angle_error_official_buggy_degrees": buggy_angle,
                    "center_error_pixels": center_error,
                    "width_error_pixels": width_error,
                    "center_error_diagonal_normalized": center_error_normalized,
                    "width_error_diagonal_normalized": width_error_normalized,
                    "corner_error_pixels_with_polygon_symmetry": corner_error,
                    "outside_ratio_2d_geometry": outside_ratio,
                    "gacc_corrected": corrected,
                    "gacc_official_buggy": buggy,
                }
            )
            output_handle.write(json.dumps(result) + "\n")
            results.append(result)

    none_precision = true_none_count / max(1, predicted_none_count)
    none_recall = true_none_count / max(1, negative_count)
    none_f1 = (
        2.0 * none_precision * none_recall / (none_precision + none_recall)
        if none_precision + none_recall > 0.0
        else 0.0
    )
    representation_oracle_miou = representation_oracle_iou_sum / max(
        1, positive_count
    )
    miou_oracle_ratio = min(
        1.0,
        strict_iou_sum / representation_oracle_iou_sum
        if representation_oracle_iou_sum > 0.0
        else 0.0,
    )
    metrics = {
        "realvlg_official_commit": REALVLG_OFFICIAL_COMMIT,
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "format_valid_rate": format_valid_count / max(1, positive_count),
        "positive_grasp_output_rate": grasp_output_count / max(1, positive_count),
        "width_valid_rate": width_valid_count / max(1, positive_count),
        "complete_six_slot_rate": complete_six_slot_count
        / max(1, positive_count),
        "coordinate_top1_accuracy": coordinate_top1_correct
        / max(1, 4 * positive_count),
        "mIoU_valid": mean(valid_ious) if valid_ious else 0.0,
        "mIoU_strict": strict_iou_sum / max(1, positive_count),
        "representation_oracle_mIoU_strict": representation_oracle_miou,
        "miou_oracle_ratio": miou_oracle_ratio,
        "gAcc_corrected_valid": (
            mean(corrected_valid) if corrected_valid else 0.0
        ),
        "gAcc_corrected_strict": corrected_strict_sum / max(1, positive_count),
        "gAcc_official_buggy_valid": mean(buggy_valid) if buggy_valid else 0.0,
        "angle_seam_subset_gAcc": (
            mean(seam_corrected) if seam_corrected else None
        ),
        "gacc_corrected_strict": corrected_strict_sum / max(1, positive_count),
        "miou_strict": strict_iou_sum / max(1, positive_count),
        "center_error_pixels": _metric_summary(center_errors),
        "center_error_diagonal_normalized": _metric_summary(
            center_errors_normalized
        ),
        "angle_error_degrees_mod_180": _metric_summary(angle_errors),
        "width_error_pixels": _metric_summary(width_errors),
        "width_error_diagonal_normalized": _metric_summary(
            width_errors_normalized
        ),
        "corner_error_pixels_with_polygon_symmetry": _metric_summary(
            corner_errors
        ),
        "outside_ratio_2d_geometry": _metric_summary(outside_ratios),
        "geometry_invalid_rate": geometry_invalid_count / max(1, positive_count),
        "structure_invalid_rate": structure_invalid_count / max(1, positive_count),
        "decode_fallback_rate": fallback_count / max(1, len(results)),
        "mean_latency_seconds": (
            mean(generation_latencies) if generation_latencies else None
        ),
        "false_none_rate_positive": false_none_count / max(1, positive_count),
        "none_precision": none_precision,
        "none_recall": none_recall,
        "none_f1": none_f1,
        "prediction_format": args.prediction_format,
        "gripper_depth_pixels": args.gripper_depth_pixels,
        "minimum_width_diagonal": args.minimum_width_diagonal,
    }
    metrics_path = (
        args.metrics.expanduser().resolve()
        if args.metrics is not None
        else output_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
