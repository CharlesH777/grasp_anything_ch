#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic scene- and frame-stratified JSONL sample."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_scene(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[str(row.get("image", ""))].append(row)
    images = list(by_image)
    rng.shuffle(images)
    for image in images:
        rng.shuffle(by_image[image])

    selected: list[dict] = []
    while images and len(selected) < count:
        next_images = []
        for image in images:
            candidates = by_image[image]
            if candidates:
                selected.append(candidates.pop())
                if len(selected) == count:
                    break
            if candidates:
                next_images.append(image)
        images = next_images
    return selected


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    rows = load_rows(args.input)
    if args.count > len(rows):
        raise SystemExit(f"requested {args.count} rows from only {len(rows)}")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[str(row.get("scene", "unknown"))].append(row)
    scenes = sorted(by_scene)
    base, remainder = divmod(args.count, len(scenes))
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for index, scene in enumerate(scenes):
        quota = base + int(index < remainder)
        selected.extend(sample_scene(by_scene[scene], quota, rng))

    if len(selected) != args.count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {args.count}")
    selected.sort(key=lambda row: str(row.get("sample_id", "")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"input": len(rows), "selected": len(selected), "scenes": len(scenes)}
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
