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
) -> dict[str, Any]:
    trainer_state = _load_json(checkpoint / "trainer_state.json")
    global_step = trainer_state.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step <= 0
    ):
        raise ValueError("checkpoint trainer_state.json needs a positive global_step")

    contact_state = _load_json(checkpoint / "grasp_contact_trainer_state.json")
    if contact_state.get("training_phase") != phase:
        raise ValueError(
            "checkpoint phase does not match acceptance phase: "
            f"saved={contact_state.get('training_phase')!r}, requested={phase!r}"
        )

    split_metrics: dict[str, dict[str, Any]] = {}
    total_positive = 0
    weighted_format = 0.0
    weighted_output = 0.0
    weighted_gacc = 0.0
    for split, path in metric_paths.items():
        metrics = _load_json(path)
        positive = _positive_count(metrics, path)
        format_rate = _rate(metrics, "format_valid_rate", path)
        output_rate = _rate(metrics, "positive_grasp_output_rate", path)
        gacc = _rate(metrics, "gacc_corrected_strict", path)
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
    thresholds = {
        "minimum_split_format_valid_rate": min_format_valid_rate,
        "minimum_split_positive_grasp_output_rate": min_positive_output_rate,
        "aggregate_gacc_corrected_strict": min_gacc_strict,
    }
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

    return {
        "phase": phase,
        "accepted": not failures,
        "checkpoint_step": global_step,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "metrics": {"aggregate": aggregate, "splits": split_metrics},
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
        choices=("sft", "pair", "geometry", "negative", "multigt"),
        required=True,
    )
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="Evaluator metrics JSON; may be repeated for multiple splits.",
    )
    parser.add_argument("--min-format-valid-rate", type=float, default=0.98)
    parser.add_argument("--min-positive-output-rate", type=float, default=0.98)
    parser.add_argument("--min-gacc-strict", type=float, default=0.30)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metric_paths: dict[str, Path] = {}
    for item in args.metrics:
        if "=" not in item:
            raise SystemExit(f"invalid --metrics value: {item!r}")
        split, raw_path = item.split("=", 1)
        if not split or split in metric_paths:
            raise SystemExit(f"invalid or duplicate metrics split: {split!r}")
        metric_paths[split] = Path(raw_path).expanduser().resolve()

    for name, value in (
        ("min_format_valid_rate", args.min_format_valid_rate),
        ("min_positive_output_rate", args.min_positive_output_rate),
        ("min_gacc_strict", args.min_gacc_strict),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")

    try:
        payload = build_acceptance(
            args.checkpoint.expanduser().resolve(),
            args.phase,
            metric_paths,
            min_format_valid_rate=args.min_format_valid_rate,
            min_positive_output_rate=args.min_positive_output_rate,
            min_gacc_strict=args.min_gacc_strict,
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
