#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import convert_realvlg_grasp as converter
from PIL import Image, ImageDraw

MIN_RANDOM_VISUALIZATIONS = 200
MIN_BOUNDARY_VISUALIZATIONS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 0 RealVLG Grasp audit."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--camera", default="kinect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-visualizations", type=int, default=200)
    parser.add_argument("--boundary-visualizations", type=int, default=50)
    parser.add_argument(
        "--expected-hash",
        action="append",
        default=[],
        metavar="SPLIT=SHA256",
    )
    parser.add_argument(
        "--confirm-visual-review",
        action="store_true",
        help=(
            "Mark Phase 0 accepted only after a human reviewed the generated "
            "random and boundary visualizations."
        ),
    )
    return parser.parse_args()


def _expected_hashes(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --expected-hash value: {value!r}")
        split, digest = value.split("=", 1)
        if split not in {"train", "seen", "similar", "novel"}:
            raise ValueError(f"unsupported hash split: {split!r}")
        if split in result or len(digest) != 64:
            raise ValueError(f"invalid or duplicate hash for split {split!r}")
        int(digest, 16)
        result[split] = digest.lower()
    return result


def _conversion_args(
    args: argparse.Namespace,
    split: str,
    expected_hash: str | None,
) -> argparse.Namespace:
    official = split != "train"
    return argparse.Namespace(
        data_root=args.data_root,
        metadata_dir=args.metadata_dir,
        output=args.output_dir / f"{split}.jsonl",
        stats=args.output_dir / f"{split}.stats.json",
        dataset_name="GraspNet_VLG",
        camera=args.camera,
        split="all" if split == "train" else split,
        max_candidates=8,
        minimum_width_diagonal=1e-4,
        gripper_depth_pixels=40.0,
        official_graspnet_eval=official,
        scene_start=0 if split == "train" else None,
        scene_end_exclusive=100 if split == "train" else None,
        expected_sample_id_sha256=expected_hash,
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _render_row(row: dict[str, Any], data_root: Path, output: Path) -> None:
    image_path = Path(str(row["image"]))
    if not image_path.is_absolute():
        image_path = data_root / image_path
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    rectangles = (
        row.get("evaluation_grasp_rectangles_pixels")
        or row.get("grasp_rectangles_pixels")
        or []
    )
    if not rectangles:
        sample_id = row.get("sample_id")
        raise ValueError(f"sample has no renderable GT rectangle: {sample_id}")
    points = rectangles[0]
    polygon = [
        (float(points[index]), float(points[index + 1]))
        for index in range(0, 8, 2)
    ]
    draw = ImageDraw.Draw(image)
    draw.line([*polygon, polygon[0]], fill=(220, 30, 30), width=3)
    center_x = sum(point[0] for point in polygon) / 4.0
    center_y = sum(point[1] for point in polygon) / 4.0
    draw.line(
        (center_x - 6, center_y, center_x + 6, center_y),
        fill=(20, 150, 40),
        width=2,
    )
    draw.line(
        (center_x, center_y - 6, center_x, center_y + 6),
        fill=(20, 150, 40),
        width=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.metadata_dir = (
        args.metadata_dir.expanduser().resolve()
        if args.metadata_dir is not None
        else None
    )
    if not args.data_root.is_dir():
        raise ValueError(f"data root not found: {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "phase0_audit.json"
    completion_marker = args.output_dir / ".phase0_complete"
    completion_marker.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    if args.random_visualizations < MIN_RANDOM_VISUALIZATIONS:
        raise ValueError(
            f"Phase 0 requires at least {MIN_RANDOM_VISUALIZATIONS} random "
            "visualizations"
        )
    if args.boundary_visualizations < MIN_BOUNDARY_VISUALIZATIONS:
        raise ValueError(
            f"Phase 0 requires at least {MIN_BOUNDARY_VISUALIZATIONS} boundary "
            "visualizations"
        )
    supplied_hashes = _expected_hashes(args.expected_hash)
    required_hashes = {"seen", "similar", "novel"}
    unexpected_hashes = sorted(set(supplied_hashes) - required_hashes)
    if unexpected_hashes:
        raise ValueError(
            "unsupported frozen hash splits: " + ", ".join(unexpected_hashes)
        )
    expected = dict(converter.FROZEN_OFFICIAL_SPLIT_HASHES)
    for split, digest in supplied_hashes.items():
        if digest != expected[split]:
            raise ValueError(
                f"supplied {split} hash does not match the pinned official "
                f"snapshot: expected={expected[split]}, supplied={digest}"
            )

    split_stats = {}
    for split in ("train", "seen", "similar", "novel"):
        stats = converter.convert(
            _conversion_args(args, split, expected.get(split))
        )
        counts = stats["statistics"]
        if counts.get("positive_samples", 0) <= 0:
            raise ValueError(f"Phase 0 split {split} has no positive samples")
        if counts.get("missing_images", 0) != 0:
            raise ValueError(f"Phase 0 split {split} has missing images")
        if split in required_hashes:
            expected_count = converter.FROZEN_OFFICIAL_SPLIT_COUNTS[split]
            actual_count = counts.get("positive_samples", 0)
            if actual_count != expected_count:
                raise ValueError(
                    f"Phase 0 split {split} sample count mismatch: "
                    f"expected={expected_count}, actual={actual_count}"
                )
        split_stats[split] = stats

    train_rows = _read_rows(args.output_dir / "train.jsonl")
    rng = random.Random(args.seed)
    random_rows = rng.sample(
        train_rows, min(args.random_visualizations, len(train_rows))
    )
    boundary_rows = sorted(
        train_rows,
        key=lambda row: (
            float((row.get("candidate_outside_2d") or [0.0])[0]),
            float((row.get("grasp_rect_candidates") or [[0, 0, 0, 0]])[0][3]),
            str(row.get("sample_id", "")),
        ),
        reverse=True,
    )[: args.boundary_visualizations]
    if len(random_rows) < MIN_RANDOM_VISUALIZATIONS:
        raise ValueError(
            f"Phase 0 produced only {len(random_rows)} random visualizations"
        )
    if len(boundary_rows) < MIN_BOUNDARY_VISUALIZATIONS:
        raise ValueError(
            f"Phase 0 produced only {len(boundary_rows)} boundary visualizations"
        )
    for group, rows in (("random", random_rows), ("boundary", boundary_rows)):
        for index, row in enumerate(rows):
            _render_row(
                row,
                args.data_root,
                args.output_dir / "visualizations" / group / f"{index:04d}.jpg",
            )

    manifest = {
        "phase": "phase0",
        "accepted": bool(args.confirm_visual_review),
        "snapshot_verified": True,
        "visual_review_confirmed": bool(args.confirm_visual_review),
        "expected_hashes": expected,
        "realvlg_official_commit": converter.REALVLG_OFFICIAL_COMMIT,
        "splits": {
            split: {
                "positive_samples": stats["statistics"]["positive_samples"],
                "sample_id_sha256": stats["protocol"]["sample_id_sha256"],
                "filter_reasons": stats["filter_reasons"],
                "geometry": stats["geometry"],
            }
            for split, stats in split_stats.items()
        },
        "visualizations": {
            "random": len(random_rows),
            "boundary": len(boundary_rows),
        },
    }
    _atomic_write(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    if manifest["accepted"]:
        _atomic_write(completion_marker, converter.REALVLG_OFFICIAL_COMMIT + "\n")
    return manifest


def main() -> int:
    try:
        manifest = audit(parse_args())
    except ValueError as error:
        print(f"Phase 0 audit failed: {error}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
