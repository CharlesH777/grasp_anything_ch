#!/usr/bin/env python3
"""Render contact predictions and matched GT contacts as image grids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _rows(path: Path) -> dict[str, dict]:
    return {
        row["sample_id"]: row
        for row in (json.loads(line) for line in path.open(encoding="utf-8"))
        if row.get("sample_id")
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _draw_line(draw: ImageDraw.ImageDraw, points: list[float], color: str) -> None:
    if len(points) != 4:
        return
    x1, y1, x2, y2 = (float(value) for value in points)
    draw.line((x1, y1, x2, y2), fill=color, width=5)
    radius = 7
    for x, y in ((x1, y1), (x2, y2)):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=9)
    args = parser.parse_args()

    annotations = _rows(args.annotations)
    predictions = _rows(args.predictions)
    selected = list(predictions.values())[: max(1, args.limit)]
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for prediction in selected:
        annotation = annotations.get(prediction["sample_id"])
        if annotation is None:
            continue
        image_path = _resolve(args.data_root, str(annotation["image"]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        _draw_line(draw, prediction.get("matched_gt_contacts_pixels", []), "#23a455")
        _draw_line(draw, prediction.get("prediction_contacts_pixels", []), "#e53935")
        label = (
            f"{prediction['sample_id'].split(':')[-1]} "
            f"gAcc={prediction.get('gacc_corrected', 0)} "
            f"IoU={prediction.get('iou', 0.0):.2f} "
            f"ang={prediction.get('angle_error_corrected_degrees', 0.0):.1f}"
        )
        draw.rectangle((0, 0, image.width, 20), fill="black")
        draw.text((4, 4), label, fill="white", font=font)
        scale = min(1.0, 512.0 / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale))
            )
        tiles.append(image)

    if not tiles:
        raise SystemExit("no matching prediction/annotation rows")
    columns = 3
    rows = math.ceil(len(tiles) / columns)
    tile_width = max(image.width for image in tiles)
    tile_height = max(image.height for image in tiles)
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202020")
    for index, tile in enumerate(tiles):
        grid.paste(
            tile,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output, quality=92)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
