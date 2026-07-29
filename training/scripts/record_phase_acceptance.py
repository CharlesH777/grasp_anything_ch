#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _rate(metrics: dict[str, Any], name: str, source: Path) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{source} has invalid {name}={value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{source} has out-of-range {name}={value!r}")
    return value


def _positive_count(metrics: dict[str, Any], source: Path) -> int:
    value = metrics.get("positive_samples")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{source} has invalid positive_samples={value!r}")
    return value


def build_acceptance(
    checkpoint: Path,
    phase: str,
    metric_paths: dict[str, Path],
    *,
    min_format_valid_rate: float,
    min_positive_output_rate: float,
    min_gacc_strict: float,
    task: str = "contact",
    min_coordinate_top1_accuracy: float = 0.95,
    min_overfit_miou_ratio: float = 0.95,
    state_filename: str | None = None,
) -> dict[str, Any]:
    trainer_state = _load_json(checkpoint / "trainer_state.json")
    global_step = trainer_state.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step <= 0
    ):
        raise ValueError("checkpoint trainer_state.json needs a positive global_step")

    if state_filename is None:
        state_filename = (
            "grasp_rect_trainer_state.json"
            if task == "grasp_rect"
            else "grasp_contact_trainer_state.json"
        )
    task_state = _load_json(checkpoint / state_filename)
    if task_state.get("training_phase") != phase:
        raise ValueError(
            "checkpoint phase does not match acceptance phase: "
            f"saved={task_state.get('training_phase')!r}, requested={phase!r}"
        )

    split_metrics: dict[str, dict[str, Any]] = {}
    total_positive = 0
    weighted_format = 0.0
    weighted_output = 0.0
    weighted_gacc = 0.0
    weighted_coordinate_top1 = 0.0
    coordinate_top1_positive = 0
    weighted_optional = {
        "width_valid_rate": 0.0,
        "complete_six_slot_rate": 0.0,
        "miou_strict": 0.0,
        "representation_oracle_miou_strict": 0.0,
        "miou_oracle_ratio": 0.0,
    }
    optional_positive = {name: 0 for name in weighted_optional}
    for split, path in metric_paths.items():
        metrics = _load_json(path)
        positive = _positive_count(metrics, path)
        format_rate = _rate(metrics, "format_valid_rate", path)
        output_rate = _rate(metrics, "positive_grasp_output_rate", path)
        gacc_name = (
            "gacc_corrected_strict"
            if "gacc_corrected_strict" in metrics
            else "gAcc_corrected_strict"
        )
        gacc = _rate(metrics, gacc_name, path)
        split_metrics[split] = {
            "positive_samples": positive,
            "format_valid_rate": format_rate,
            "positive_grasp_output_rate": output_rate,
            "gacc_corrected_strict": gacc,
            "miou_strict": metrics.get("miou_strict"),
            "swap_invariant_endpoint_error_pixels": metrics.get(
                "swap_invariant_endpoint_error_pixels"
            ),
            "swap_invariant_angle_error_degrees": metrics.get(
                "swap_invariant_angle_error_degrees"
            ),
        }
        coordinate_top1 = metrics.get("coordinate_top1_accuracy")
        if coordinate_top1 is not None:
            coordinate_top1 = _rate(metrics, "coordinate_top1_accuracy", path)
            split_metrics[split]["coordinate_top1_accuracy"] = coordinate_top1
            weighted_coordinate_top1 += positive * coordinate_top1
            coordinate_top1_positive += positive
        for name in weighted_optional:
            source_name = name
            if name == "miou_strict" and source_name not in metrics:
                source_name = "mIoU_strict"
            elif name == "representation_oracle_miou_strict":
                source_name = "representation_oracle_mIoU_strict"
            if source_name in metrics:
                value = _rate(metrics, source_name, path)
                split_metrics[split][name] = value
                weighted_optional[name] += positive * value
                optional_positive[name] += positive
        total_positive += positive
        weighted_format += positive * format_rate
        weighted_output += positive * output_rate
        weighted_gacc += positive * gacc

    if not split_metrics:
        raise ValueError("at least one split metric must be provided")
    aggregate = {
        "positive_samples": total_positive,
        "format_valid_rate": weighted_format / total_positive,
        "positive_grasp_output_rate": weighted_output / total_positive,
        "gacc_corrected_strict": weighted_gacc / total_positive,
        "minimum_split_format_valid_rate": min(
            item["format_valid_rate"] for item in split_metrics.values()
        ),
        "minimum_split_positive_grasp_output_rate": min(
            item["positive_grasp_output_rate"] for item in split_metrics.values()
        ),
    }
    if coordinate_top1_positive:
        aggregate["coordinate_top1_accuracy"] = (
            weighted_coordinate_top1 / coordinate_top1_positive
        )
    for name, total in weighted_optional.items():
        if optional_positive[name]:
            aggregate[name] = total / optional_positive[name]
    thresholds = {
        "minimum_split_format_valid_rate": min_format_valid_rate,
        "minimum_split_positive_grasp_output_rate": min_positive_output_rate,
        "aggregate_gacc_corrected_strict": min_gacc_strict,
    }
    if phase == "overfit":
        thresholds["coordinate_top1_accuracy"] = min_coordinate_top1_accuracy
        if task == "grasp_rect":
            thresholds.update(
                width_valid_rate=1.0,
                complete_six_slot_rate=0.99,
                miou_oracle_ratio=min_overfit_miou_ratio,
            )
    failures = []
    if aggregate["minimum_split_format_valid_rate"] < min_format_valid_rate:
        failures.append("format_valid_rate")
    if (
        aggregate["minimum_split_positive_grasp_output_rate"]
        < min_positive_output_rate
    ):
        failures.append("positive_grasp_output_rate")
    if aggregate["gacc_corrected_strict"] < min_gacc_strict:
        failures.append("gacc_corrected_strict")
    if phase == "overfit":
        coordinate_top1 = aggregate.get("coordinate_top1_accuracy")
        if (
            coordinate_top1 is None
            or coordinate_top1 < min_coordinate_top1_accuracy
        ):
            failures.append("coordinate_top1_accuracy")
        if task == "grasp_rect":
            for name, threshold in (
                ("width_valid_rate", 1.0),
                ("complete_six_slot_rate", 0.99),
                ("miou_oracle_ratio", min_overfit_miou_ratio),
            ):
                value = aggregate.get(name)
                if value is None or value < threshold:
                    failures.append(name)

    metrics_payload: dict[str, Any] = {
        "aggregate": aggregate,
        "splits": split_metrics,
    }
    if phase == "overfit":
        metrics_payload.update(
            format_valid_rate=aggregate["format_valid_rate"],
            positive_grasp_output_rate=aggregate[
                "positive_grasp_output_rate"
            ],
            coordinate_top1_accuracy=aggregate.get(
                "coordinate_top1_accuracy"
            ),
            gacc_corrected_strict=aggregate["gacc_corrected_strict"],
            width_valid_rate=aggregate.get("width_valid_rate"),
            complete_six_slot_rate=aggregate.get("complete_six_slot_rate"),
            miou_strict=aggregate.get("miou_strict"),
            representation_oracle_miou_strict=aggregate.get(
                "representation_oracle_miou_strict"
            ),
            miou_oracle_ratio=aggregate.get("miou_oracle_ratio"),
        )

    return {
        "phase": phase,
        "task": task,
        "accepted": not failures,
        "checkpoint_step": global_step,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "metrics": metrics_payload,
        "failures": failures,
    }


def build_joint_acceptance(
    checkpoint: Path,
    phase: str,
    contact_metric_paths: dict[str, Path],
    grasp_rect_metric_paths: dict[str, Path],
    grounding_metrics_path: Path,
    *,
    min_format_valid_rate: float,
    min_positive_output_rate: float,
    min_gacc_strict: float,
    min_coordinate_top1_accuracy: float = 0.95,
    min_overfit_miou_ratio: float = 0.95,
    min_grounding_retention: float = 0.98,
) -> dict[str, Any]:
    common = {
        "min_format_valid_rate": min_format_valid_rate,
        "min_positive_output_rate": min_positive_output_rate,
        "min_gacc_strict": min_gacc_strict,
        "min_coordinate_top1_accuracy": min_coordinate_top1_accuracy,
        "min_overfit_miou_ratio": min_overfit_miou_ratio,
        "state_filename": "joint_trainer_state.json",
    }
    contact = build_acceptance(
        checkpoint,
        phase,
        contact_metric_paths,
        task="contact",
        **common,
    )
    grasp_rect = build_acceptance(
        checkpoint,
        phase,
        grasp_rect_metric_paths,
        task="grasp_rect",
        **common,
    )
    grounding = _load_json(grounding_metrics_path)
    retention = grounding.get("retention_ratio")
    if retention is None:
        baseline = grounding.get("baseline_score")
        score = grounding.get("score")
        if (
            isinstance(baseline, bool)
            or not isinstance(baseline, int | float)
            or baseline <= 0.0
            or isinstance(score, bool)
            or not isinstance(score, int | float)
        ):
            raise ValueError(
                "grounding metrics need retention_ratio or positive "
                "baseline_score plus score"
            )
        retention = float(score) / float(baseline)
    if (
        isinstance(retention, bool)
        or not isinstance(retention, int | float)
        or not math.isfinite(float(retention))
        or float(retention) < 0.0
    ):
        raise ValueError("grounding retention_ratio must be finite and non-negative")
    retention = float(retention)

    failures = [f"contact:{item}" for item in contact["failures"]]
    failures.extend(
        f"grasp_rect:{item}" for item in grasp_rect["failures"]
    )
    if retention < min_grounding_retention:
        failures.append("grounding:retention_ratio")
    return {
        "phase": phase,
        "task": "joint",
        "accepted": not failures,
        "checkpoint_step": contact["checkpoint_step"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "contact": contact["thresholds"],
            "grasp_rect": grasp_rect["thresholds"],
            "grounding_retention_ratio": min_grounding_retention,
        },
        "metrics": {
            "contact": contact["metrics"],
            "grasp_rect": grasp_rect["metrics"],
            "grounding_retention_ratio": retention,
            "grounding": grounding,
        },
        "failures": failures,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record an evaluated contact-training phase transition."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "overfit",
            "sft",
            "pair",
            "pose_r0",
            "pose",
            "geometry",
            "multigt",
            "negative",
            "collision",
            "structured_r0",
            "structured",
        ),
        required=True,
    )
    parser.add_argument(
        "--task", choices=("contact", "grasp_rect", "joint"), default="contact"
    )
    parser.add_argument(
        "--metrics",
        action="append",
        metavar="SPLIT=PATH",
        help="Evaluator metrics JSON; may be repeated for multiple splits.",
    )
    parser.add_argument("--contact-metrics", action="append", metavar="SPLIT=PATH")
    parser.add_argument(
        "--grasp-rect-metrics", action="append", metavar="SPLIT=PATH"
    )
    parser.add_argument("--grounding-metrics", type=Path)
    parser.add_argument("--min-format-valid-rate", type=float, default=0.98)
    parser.add_argument("--min-positive-output-rate", type=float, default=0.98)
    parser.add_argument("--min-gacc-strict", type=float, default=0.30)
    parser.add_argument(
        "--min-coordinate-top1-accuracy", type=float, default=0.95
    )
    parser.add_argument("--min-overfit-miou-ratio", type=float, default=0.95)
    parser.add_argument("--min-grounding-retention", type=float, default=0.98)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    def parse_metric_paths(items: list[str] | None, option: str) -> dict[str, Path]:
        metric_paths: dict[str, Path] = {}
        for item in items or []:
            if "=" not in item:
                raise SystemExit(f"invalid {option} value: {item!r}")
            split, raw_path = item.split("=", 1)
            if not split or split in metric_paths:
                raise SystemExit(
                    f"invalid or duplicate {option} split: {split!r}"
                )
            metric_paths[split] = Path(raw_path).expanduser().resolve()
        return metric_paths

    metric_paths = parse_metric_paths(args.metrics, "--metrics")
    contact_metric_paths = parse_metric_paths(
        args.contact_metrics, "--contact-metrics"
    )
    grasp_rect_metric_paths = parse_metric_paths(
        args.grasp_rect_metrics, "--grasp-rect-metrics"
    )
    if args.task == "joint":
        if not contact_metric_paths or not grasp_rect_metric_paths:
            raise SystemExit(
                "joint acceptance requires --contact-metrics and "
                "--grasp-rect-metrics"
            )
        if args.grounding_metrics is None:
            raise SystemExit("joint acceptance requires --grounding-metrics")
    elif not metric_paths:
        raise SystemExit("--metrics is required for single-task acceptance")

    for name, value in (
        ("min_format_valid_rate", args.min_format_valid_rate),
        ("min_positive_output_rate", args.min_positive_output_rate),
        ("min_gacc_strict", args.min_gacc_strict),
        ("min_coordinate_top1_accuracy", args.min_coordinate_top1_accuracy),
        ("min_overfit_miou_ratio", args.min_overfit_miou_ratio),
        ("min_grounding_retention", args.min_grounding_retention),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")

    try:
        if args.task == "joint":
            payload = build_joint_acceptance(
                args.checkpoint.expanduser().resolve(),
                args.phase,
                contact_metric_paths,
                grasp_rect_metric_paths,
                args.grounding_metrics.expanduser().resolve(),
                min_format_valid_rate=args.min_format_valid_rate,
                min_positive_output_rate=args.min_positive_output_rate,
                min_gacc_strict=args.min_gacc_strict,
                min_coordinate_top1_accuracy=(
                    args.min_coordinate_top1_accuracy
                ),
                min_overfit_miou_ratio=args.min_overfit_miou_ratio,
                min_grounding_retention=args.min_grounding_retention,
            )
        else:
            payload = build_acceptance(
                args.checkpoint.expanduser().resolve(),
                args.phase,
                metric_paths,
                min_format_valid_rate=args.min_format_valid_rate,
                min_positive_output_rate=args.min_positive_output_rate,
                min_gacc_strict=args.min_gacc_strict,
                task=args.task,
                min_coordinate_top1_accuracy=(
                    args.min_coordinate_top1_accuracy
                ),
                min_overfit_miou_ratio=args.min_overfit_miou_ratio,
            )
    except ValueError as error:
        print(f"Phase acceptance failed: {error}")
        return 1

    if args.report:
        _atomic_write(args.report.expanduser().resolve(), payload)
    if not payload["accepted"]:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 1

    destination = args.checkpoint.expanduser().resolve() / "phase_acceptance.json"
    _atomic_write(destination, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
