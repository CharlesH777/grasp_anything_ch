#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from locate_anything_service.collision_2d import evaluate_collision_2d  # noqa: E402
from locate_anything_service.grasp_geometry import (  # noqa: E402
    angular_error_degrees,
    derive_grasp_geometry,
    grasp_rectangle,
    polygon_area,
    polygon_inside_image_area,
    polygon_iou,
)
from locate_anything_service.parser import parse_grasp_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LocateAnything contact-point predictions."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path)
    source.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--generation-mode", choices=("fast", "slow", "hybrid"), default="fast"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--coord-mass-threshold", type=float, default=1e-4)
    parser.add_argument("--rectangle-thickness", type=float, default=80.0)
    parser.add_argument("--collision-threshold", type=float, default=0.0)
    parser.add_argument("--outside-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def official_buggy_angular_error_degrees(
    pred_degrees: float, gt_degrees: float
) -> float:
    """Reproduce RealVLG-R1's unit-guessing bug for old-log reconciliation."""
    if abs(pred_degrees) <= math.pi and abs(gt_degrees) <= math.pi:
        pred_radians, gt_radians = pred_degrees, gt_degrees
    else:
        pred_radians = math.radians(pred_degrees)
        gt_radians = math.radians(gt_degrees)
    cosine = max(
        -1.0,
        min(
            1.0,
            math.cos(pred_radians) * math.cos(gt_radians)
            + math.sin(pred_radians) * math.sin(gt_radians),
        ),
    )
    difference = math.degrees(math.acos(cosine))
    return 180.0 - difference if difference > 90.0 else difference


def _swap_invariant_endpoint_error(
    prediction: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    px1, py1, px2, py2 = prediction
    tx1, ty1, tx2, ty2 = target
    identity = (
        math.hypot(px1 - tx1, py1 - ty1)
        + math.hypot(px2 - tx2, py2 - ty2)
    ) * 0.5
    swapped = (
        math.hypot(px1 - tx2, py1 - ty2)
        + math.hypot(px2 - tx1, py2 - ty1)
    ) * 0.5
    return min(identity, swapped)


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p95": p95,
        "max": ordered[-1],
    }


def _decode_output(output: Any, input_ids: Any, processor: Any) -> str:
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
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self.coord_mass_threshold = coord_mass_threshold
        self.model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=True
        )

    def __call__(self, image_path: Path, description: str) -> tuple[str, float]:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        prompt = (
            "Predict one plausible two-finger 2D contact pair for the target "
            f"described as: {description}."
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
        kwargs = {
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
            "geometry_type": "contact",
            "image_size": image.size,
            "contact_coord_mass_threshold": self.coord_mass_threshold,
        }
        if self.generation_mode in {"fast", "hybrid"}:
            kwargs["n_future_tokens"] = 6
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**kwargs)
        return (
            _decode_output(output, inputs["input_ids"], self.processor),
            time.perf_counter() - started,
        )


def _prediction_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {row["sample_id"]: row for row in _load_jsonl(path)}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    coord_mass_threshold = float(getattr(args, "coord_mass_threshold", 1e-4))
    for name, value in (
        ("coord_mass_threshold", coord_mass_threshold),
        ("collision_threshold", args.collision_threshold),
        ("outside_threshold", args.outside_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    annotations = _load_jsonl(args.annotations)
    if args.limit:
        annotations = annotations[: args.limit]
    data_root = args.data_root.expanduser().resolve()
    predictions = _prediction_map(args.predictions)
    predictor = (
        ModelPredictor(
            args.model_path.expanduser().resolve(),
            args.generation_mode,
            args.max_new_tokens,
            coord_mass_threshold,
        )
        if args.model_path
        else None
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    valid_ious: list[float] = []
    valid_corrected_acc: list[int] = []
    valid_official_buggy_acc: list[int] = []
    strict_iou_sum = 0.0
    strict_corrected_sum = 0
    positive_count = 0
    negative_count = 0
    positive_format_valid_count = 0
    negative_format_valid_count = 0
    none_tp = none_fp = none_fn = 0
    collision_valid_count = collision_count = collision_unknown_count = 0
    collision_aware_correct = 0
    collision_ratio_sum = outside_ratio_sum = 0.0
    endpoint_errors: list[float] = []
    angle_errors: list[float] = []
    center_errors: list[float] = []
    width_errors: list[float] = []
    geometric_outside_ratios: list[float] = []
    geometric_outside_count = 0

    for annotation in annotations:
        sample_id = annotation["sample_id"]
        is_positive = annotation.get("task_type") == "grasp_contact"
        image_path = _resolve(data_root, annotation.get("image"))
        if image_path is None:
            raise ValueError(f"sample {sample_id} has no image")
        width = int(annotation["image_width"])
        height = int(annotation["image_height"])
        if predictor is not None:
            raw_output, latency = predictor(
                image_path, str(annotation.get("description", ""))
            )
        else:
            row = predictions.get(sample_id, {})
            raw_output = str(row.get("raw_output", ""))
            latency = float(row.get("latency_seconds", 0.0))

        parsed = parse_grasp_output(raw_output, width, height)
        syntax_valid = parsed.status != "invalid"
        if is_positive:
            positive_format_valid_count += int(syntax_valid)
        else:
            negative_count += 1
            negative_format_valid_count += int(syntax_valid)
        predicted_none = parsed.status == "none"
        if predicted_none and not is_positive:
            none_tp += 1
        elif predicted_none and is_positive:
            none_fp += 1
        elif not predicted_none and not is_positive:
            none_fn += 1

        result: dict[str, Any] = {
            "sample_id": sample_id,
            "raw_output": raw_output,
            "status": parsed.status,
            "parse_error": parsed.error,
            "latency_seconds": latency,
            "iou": 0.0 if is_positive else None,
            "angle_error_corrected_degrees": None,
            "angle_error_official_buggy_degrees": None,
            "gacc_corrected": 0 if is_positive else None,
            "gacc_official_buggy": 0 if is_positive else None,
        }
        if not is_positive:
            results.append(result)
            continue

        positive_count += 1
        if parsed.status != "ok":
            collision_unknown_count += 1
            results.append(result)
            continue

        prediction = parsed.grasps[0]
        pred_geometry = derive_grasp_geometry(
            prediction.contacts_1000, width, height
        )
        pred_polygon = grasp_rectangle(
            pred_geometry.contacts_pixels_float, args.rectangle_thickness
        )
        full_polygon_area = polygon_area(pred_polygon)
        inside_polygon_area = polygon_inside_image_area(
            pred_polygon, width, height
        )
        geometric_outside_ratio = (
            min(
                1.0,
                max(
                    0.0,
                    (full_polygon_area - inside_polygon_area) / full_polygon_area,
                ),
            )
            if full_polygon_area > 1e-9
            else 0.0
        )
        geometric_outside_ratios.append(geometric_outside_ratio)
        geometric_outside_count += int(
            geometric_outside_ratio > args.outside_threshold
        )
        gt_candidates = (
            annotation.get("evaluation_contact_candidates_pixels")
            or annotation.get("contact_candidates_pixels")
            or []
        )
        if not gt_candidates:
            raise ValueError(f"positive sample {sample_id} has no pixel GT contacts")

        best_iou = -1.0
        best_gt: tuple[float, float, float, float] | None = None
        best_gt_geometry = None
        for raw_gt in gt_candidates:
            gt = tuple(float(value) for value in raw_gt)
            gt_token_space = (
                gt[0] * 1000 / width,
                gt[1] * 1000 / height,
                gt[2] * 1000 / width,
                gt[3] * 1000 / height,
            )
            gt_geometry = derive_grasp_geometry(gt_token_space, width, height)
            gt_polygon = grasp_rectangle(
                gt_geometry.contacts_pixels_float, args.rectangle_thickness
            )
            overlap = polygon_iou(pred_polygon, gt_polygon)
            if overlap > best_iou:
                best_iou = overlap
                best_gt = gt
                best_gt_geometry = gt_geometry
        if best_gt is None or best_gt_geometry is None:
            raise ValueError(f"sample {sample_id} has no valid GT contacts")

        corrected_angle = angular_error_degrees(
            pred_geometry.angle_radians_image,
            best_gt_geometry.angle_radians_image,
        )
        official_buggy_angle = official_buggy_angular_error_degrees(
            math.degrees(pred_geometry.angle_radians_image),
            math.degrees(best_gt_geometry.angle_radians_image),
        )
        corrected = int(best_iou > 0.25 and corrected_angle < 30.0)
        official_buggy = int(
            best_iou > 0.25 and official_buggy_angle < 30.0
        )
        valid_ious.append(best_iou)
        valid_corrected_acc.append(corrected)
        valid_official_buggy_acc.append(official_buggy)
        strict_iou_sum += best_iou
        strict_corrected_sum += corrected

        pred_pixels = pred_geometry.contacts_pixels_float
        gt_center = (
            (best_gt[0] + best_gt[2]) * 0.5,
            (best_gt[1] + best_gt[3]) * 0.5,
        )
        center_error = math.hypot(
            pred_geometry.center_pixels_float[0] - gt_center[0],
            pred_geometry.center_pixels_float[1] - gt_center[1],
        )
        endpoint_error = _swap_invariant_endpoint_error(pred_pixels, best_gt)
        width_error = abs(
            pred_geometry.opening_width_pixels
            - best_gt_geometry.opening_width_pixels
        )
        endpoint_errors.append(endpoint_error)
        angle_errors.append(corrected_angle)
        center_errors.append(center_error)
        width_errors.append(width_error)
        result.update(
            {
                "prediction_contacts_pixels": list(pred_pixels),
                "matched_gt_contacts_pixels": list(best_gt),
                "iou": best_iou,
                "angle_error_corrected_degrees": corrected_angle,
                "angle_error_official_buggy_degrees": official_buggy_angle,
                "gacc_corrected": corrected,
                "gacc_official_buggy": official_buggy,
                "endpoint_error_pixels": endpoint_error,
                "center_error_pixels": center_error,
                "width_error_pixels": width_error,
                "outside_ratio_2d_geometry": geometric_outside_ratio,
            }
        )

        collision_valid = bool(annotation.get("collision_valid"))
        obstacle_path = _resolve(data_root, annotation.get("obstacle_mask"))
        if collision_valid and obstacle_path and obstacle_path.is_file():
            with Image.open(obstacle_path) as source:
                obstacle = source.convert("L")
            collision = evaluate_collision_2d(
                pred_geometry,
                obstacle,
                width,
                height,
                thickness_pixels=args.rectangle_thickness,
                collision_threshold=args.collision_threshold,
                outside_threshold=args.outside_threshold,
            )
            collision_valid_count += 1
            is_collision = collision.status == "collision"
            collision_count += int(is_collision)
            collision_ratio_sum += collision.collision_ratio or 0.0
            outside_ratio_sum += collision.outside_ratio or 0.0
            collision_aware_correct += int(corrected and not is_collision)
            result.update(
                {
                    "collision_status": collision.status,
                    "collision_ratio_2d": collision.collision_ratio,
                    "outside_ratio_2d": collision.outside_ratio,
                    "clearance_pixels_2d": collision.clearance_pixels,
                }
            )
        else:
            collision_unknown_count += 1
            result["collision_status"] = "unknown"
        results.append(result)

    valid_count = len(valid_ious)
    none_precision = none_tp / max(1, none_tp + none_fp)
    none_recall = none_tp / max(1, none_tp + none_fn)
    metrics = {
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "format_valid_rate": (
            positive_format_valid_count / max(1, positive_count)
        ),
        "positive_grasp_output_rate": valid_count / max(1, positive_count),
        "negative_format_valid_rate": (
            negative_format_valid_count / max(1, negative_count)
        ),
        "overall_format_valid_rate": (
            (positive_format_valid_count + negative_format_valid_count)
            / max(1, positive_count + negative_count)
        ),
        "miou_valid": sum(valid_ious) / max(1, valid_count),
        "miou_strict": strict_iou_sum / max(1, positive_count),
        "gacc_corrected_valid": sum(valid_corrected_acc) / max(1, valid_count),
        "gacc_official_buggy_valid": (
            sum(valid_official_buggy_acc) / max(1, valid_count)
        ),
        "gacc_corrected_strict": strict_corrected_sum / max(1, positive_count),
        "none_precision": none_precision,
        "none_recall": none_recall,
        "none_f1": (
            2 * none_precision * none_recall / max(1e-12, none_precision + none_recall)
        ),
        "collision_valid_samples": collision_valid_count,
        "collision_evaluable_rate": collision_valid_count / max(1, positive_count),
        "collision_unknown_rate": collision_unknown_count / max(1, positive_count),
        "collision_rate_2d": (
            collision_count / collision_valid_count
            if collision_valid_count
            else None
        ),
        "mean_collision_ratio_2d": (
            collision_ratio_sum / collision_valid_count
            if collision_valid_count
            else None
        ),
        "mean_outside_ratio_2d": (
            outside_ratio_sum / collision_valid_count
            if collision_valid_count
            else None
        ),
        "collision_aware_gacc_valid": (
            collision_aware_correct / collision_valid_count
            if collision_valid_count
            else None
        ),
        "collision_aware_gacc_strict": (
            collision_aware_correct / positive_count
            if collision_valid_count and positive_count
            else None
        ),
        "swap_invariant_endpoint_error_pixels": _metric_summary(endpoint_errors),
        "swap_invariant_angle_error_degrees": _metric_summary(angle_errors),
        "center_error_pixels": _metric_summary(center_errors),
        "width_error_pixels": _metric_summary(width_errors),
        "outside_ratio_2d_geometry": _metric_summary(
            geometric_outside_ratios
        ),
        "outside_rate_2d_geometry": (
            geometric_outside_count / len(geometric_outside_ratios)
            if geometric_outside_ratios
            else None
        ),
        "mean_latency_seconds": sum(
            float(row["latency_seconds"]) for row in results
        )
        / max(1, len(results)),
        "rectangle_thickness_pixels": args.rectangle_thickness,
        "collision_threshold": args.collision_threshold,
        "outside_threshold": args.outside_threshold,
        "coord_mass_threshold": coord_mass_threshold,
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path = args.metrics.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass
    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
