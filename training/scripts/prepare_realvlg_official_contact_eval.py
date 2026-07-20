#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

SPLIT_RANGES = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}


def prepare_split(data_root: Path, split: str, output: Path) -> dict[str, int]:
    rows: list[dict] = []
    images: set[str] = set()
    empty_contacts = 0
    raw_contacts = 0
    for scene_id in SPLIT_RANGES[split]:
        scene = f"scene_{scene_id:04d}"
        metadata_path = data_root / "metadata" / "kinect" / scene / "0000.json"
        objects = json.loads(metadata_path.read_text(encoding="utf-8"))
        for object_index, obj in enumerate(objects):
            if not obj.get("grasps"):
                continue
            image_value = str(obj["image_path"])
            image_path = data_root / image_value
            raw_size = obj.get("image_size_hw")
            if (
                isinstance(raw_size, list | tuple)
                and len(raw_size) == 2
                and all(
                    isinstance(value, int | float) and value > 0 for value in raw_size
                )
            ):
                height, width = (int(raw_size[0]), int(raw_size[1]))
            else:
                with Image.open(image_path) as source:
                    width, height = source.size
            contacts = obj.get("contact_points", [])
            if not contacts:
                empty_contacts += 1
            raw_contacts += len(contacts)
            object_id = obj.get("object_id", "")
            rows.append(
                {
                    "sample_id": (
                        f"RealVLG-official:{scene}:0000:{object_id}:{object_index}"
                    ),
                    "dataset": "GraspNet_VLG",
                    "task_type": "grasp_contact",
                    "image_width": width,
                    "image_height": height,
                    "image": image_value,
                    "scene": scene,
                    "object_id": object_id,
                    "source_object_index": object_index,
                    "description": str(obj.get("description", "")),
                    "evaluation_protocol": "realvlg_released_evaluator_exact",
                    "evaluation_only": True,
                    "evaluation_contact_candidates_pixels": contacts,
                    "contact_candidates_pixels": contacts,
                    "collision_valid": False,
                    "obstacle_mask": None,
                }
            )
            images.add(image_value)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "samples": len(rows),
        "images": len(images),
        "raw_contacts": raw_contacts,
        "empty_contact_samples": empty_contacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the exact sample/GT protocol used by eval_contact.py."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    summary = {}
    for split in SPLIT_RANGES:
        summary[split] = prepare_split(
            data_root,
            split,
            args.output_dir / f"contact_{split}_official_exact.jsonl",
        )
    summary_path = args.output_dir / "official_exact_stats.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
