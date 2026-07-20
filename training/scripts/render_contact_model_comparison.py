#!/usr/bin/env python3
# ruff: noqa: E501
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SPLITS = ("seen", "similar", "novel")
COLORS = {
    "background": "#15181d",
    "surface": "#20252b",
    "text": "#f4f6f8",
    "muted": "#aeb8c2",
    "bbox": "#9be15d",
    "gt": "#2ed4e6",
    "ours": "#ff5d68",
    "realvlg": "#ffb020",
    "pass": "#69d391",
    "fail": "#ff7a82",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render GT, Ours, and RealVLG Contact predictions side by side."
    )
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--ours-dir", type=Path, required=True)
    parser.add_argument("--realvlg-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=560)
    parser.add_argument("--overview-count", type=int, default=12)
    parser.add_argument("--limit-per-split", type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/lato") / name,
        Path("/usr/share/fonts/truetype/dejavu") / "DejaVuSans.ttf",
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def valid_points(values: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(values, list | tuple) or len(values) != 4:
        return None
    try:
        points = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    return points if all(math.isfinite(value) for value in points) else None


def draw_contact(
    draw: ImageDraw.ImageDraw,
    values: Any,
    color: str | tuple[int, int, int, int],
    *,
    width: int,
    radius: int,
    outline: str | tuple[int, int, int, int] | None = None,
) -> None:
    points = valid_points(values)
    if points is None:
        return
    x1, y1, x2, y2 = points
    if outline is not None:
        draw.line((x1, y1, x2, y2), fill=outline, width=width + 4)
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    for x, y in ((x1, y1), (x2, y2)):
        bounds = (x - radius, y - radius, x + radius, y + radius)
        if outline is not None:
            draw.ellipse(bounds, fill=outline)
            inset = max(1, round(radius * 0.35))
            bounds = (
                bounds[0] + inset,
                bounds[1] + inset,
                bounds[2] - inset,
                bounds[3] - inset,
            )
        draw.ellipse(bounds, fill=color)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Any, scale: float = 1.0) -> None:
    points = valid_points(bbox)
    if points is None:
        return
    width = max(2, round(3 * scale))
    draw.rectangle(points, outline=COLORS["bbox"], width=width)


def overlay_gt(
    image: Image.Image, annotation: dict[str, Any], bbox: Any
) -> Image.Image:
    panel = image.convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_bbox(draw, bbox)
    for candidate in annotation.get("evaluation_contact_candidates_pixels", []):
        draw_contact(
            draw,
            candidate,
            (46, 212, 230, 225),
            width=4,
            radius=5,
            outline=(12, 15, 18, 190),
        )
    return Image.alpha_composite(panel, overlay).convert("RGB")


def overlay_prediction(
    image: Image.Image,
    prediction: dict[str, Any],
    bbox: Any,
    color: str,
) -> Image.Image:
    panel = image.convert("RGBA")
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_bbox(draw, bbox)
    draw_contact(
        draw,
        prediction.get("matched_gt_contacts_pixels"),
        (46, 212, 230, 210),
        width=4,
        radius=5,
        outline=(15, 18, 22, 230),
    )
    draw_contact(
        draw,
        prediction.get("prediction_contacts_pixels"),
        color,
        width=7,
        radius=9,
        outline=(255, 255, 255, 235),
    )
    return Image.alpha_composite(panel, overlay).convert("RGB")


def fit_panel(image: Image.Image, width: int) -> Image.Image:
    height = round(width * image.height / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def comparison_crop_box(
    image: Image.Image,
    bbox: Any,
    *predictions: Any,
) -> tuple[float, float, float, float]:
    target = valid_points(bbox)
    if target is None:
        return (0.0, 0.0, float(image.width), float(image.height))
    x1, y1, x2, y2 = target
    xs = [x1, x2]
    ys = [y1, y2]
    for raw in predictions:
        points = valid_points(raw)
        if points is None:
            continue
        xs.extend(
            (
                min(max(points[0], 0.0), image.width),
                min(max(points[2], 0.0), image.width),
            )
        )
        ys.extend(
            (
                min(max(points[1], 0.0), image.height),
                min(max(points[3], 0.0), image.height),
            )
        )
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    crop_width = max(512.0, (max_x - min_x) * 1.35, abs(x2 - x1) * 2.2)
    crop_height = max(288.0, (max_y - min_y) * 1.35, abs(y2 - y1) * 2.2)
    aspect = 16.0 / 9.0
    if crop_width / crop_height < aspect:
        crop_width = crop_height * aspect
    else:
        crop_height = crop_width / aspect
    crop_width = min(crop_width, float(image.width))
    crop_height = min(crop_height, float(image.height))
    left = min(max(0.0, center_x - crop_width * 0.5), image.width - crop_width)
    top = min(max(0.0, center_y - crop_height * 0.5), image.height - crop_height)
    return (left, top, left + crop_width, top + crop_height)


def format_float(value: Any, digits: int = 3) -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_prediction(prediction: dict[str, Any]) -> str:
    points = valid_points(prediction.get("prediction_contacts_pixels"))
    coords = "n/a"
    if points is not None:
        coords = f"({points[0]:.0f},{points[1]:.0f})-({points[2]:.0f},{points[3]:.0f})"
    passed = "PASS" if prediction.get("gacc_corrected") == 1 else "FAIL"
    return (
        f"{coords}  IoU {format_float(prediction.get('iou'))}  "
        f"angle {format_float(prediction.get('angle_error_corrected_degrees'), 1)}  "
        f"{passed}"
    )


def wrap_pixels(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for index, word in enumerate(words):
        candidate = word if not current else f"{current} {word}"
        if font.getlength(candidate) <= max_width:
            current = candidate
            continue
        if len(lines) == max_lines - 1:
            remainder = " ".join(([current] if current else []) + words[index:])
            while remainder and font.getlength(f"{remainder}...") > max_width:
                remainder = remainder[:-1]
            lines.append(f"{remainder.rstrip()}...")
            return lines
        if current:
            lines.append(current)
        current = word
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def metadata_object(
    data_root: Path,
    annotation: dict[str, Any],
    cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scene = str(annotation["scene"])
    if scene not in cache:
        path = data_root / "metadata" / "kinect" / scene / "0000.json"
        cache[scene] = json.loads(path.read_text(encoding="utf-8"))
    return cache[scene][int(annotation["source_object_index"])]


def in_frame_count(annotation: dict[str, Any]) -> int:
    width = float(annotation["image_width"])
    height = float(annotation["image_height"])
    count = 0
    for raw in annotation.get("evaluation_contact_candidates_pixels", []):
        points = valid_points(raw)
        if points is not None and all(
            0 <= value <= bound
            for value, bound in zip(points, (width, height, width, height), strict=True)
        ):
            count += 1
    return count


def panel_block(
    title: str,
    image: Image.Image,
    footer: str,
    accent: str,
    panel_width: int,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> Image.Image:
    title_height = 38
    footer_height = 48
    block = Image.new(
        "RGB",
        (panel_width, title_height + image.height + footer_height),
        COLORS["surface"],
    )
    draw = ImageDraw.Draw(block)
    draw.rectangle((0, 0, 6, title_height), fill=accent)
    draw.text((16, 9), title, font=title_font, fill=COLORS["text"])
    block.paste(image, (0, title_height))
    footer_y = title_height + image.height
    draw.line((0, footer_y, panel_width, footer_y), fill="#343b43", width=1)
    lines = wrap_pixels(footer, body_font, panel_width - 24, 2)
    for index, line in enumerate(lines):
        draw.text(
            (12, footer_y + 7 + index * 17),
            line,
            font=body_font,
            fill=COLORS["muted"],
        )
    return block


def render_item(
    annotation: dict[str, Any],
    ours: dict[str, Any],
    realvlg: dict[str, Any],
    source: Image.Image,
    bbox: Any,
    panel_width: int,
) -> Image.Image:
    title_font = load_font("Lato-Semibold.ttf", 17)
    body_font = load_font("Lato-Regular.ttf", 14)
    header_font = load_font("Lato-Semibold.ttf", 20)
    description_font = load_font("Lato-Regular.ttf", 16)

    crop_box = comparison_crop_box(
        source,
        bbox,
        ours.get("prediction_contacts_pixels"),
        realvlg.get("prediction_contacts_pixels"),
    )
    gt_image = fit_panel(
        overlay_gt(source, annotation, bbox).crop(crop_box), panel_width
    )
    ours_image = fit_panel(
        overlay_prediction(source, ours, bbox, COLORS["ours"]).crop(crop_box),
        panel_width,
    )
    real_image = fit_panel(
        overlay_prediction(source, realvlg, bbox, COLORS["realvlg"]).crop(crop_box),
        panel_width,
    )
    candidate_count = len(annotation.get("evaluation_contact_candidates_pixels", []))
    panels = (
        panel_block(
            "IMAGE + GT LABELS",
            gt_image,
            f"{candidate_count} raw GT pairs  |  {in_frame_count(annotation)} fully in frame",
            COLORS["gt"],
            panel_width,
            title_font,
            body_font,
        ),
        panel_block(
            "OURS / CHECKPOINT-10501",
            ours_image,
            format_prediction(ours),
            COLORS["ours"],
            panel_width,
            title_font,
            body_font,
        ),
        panel_block(
            "REALVLG-R1 GRPO CONTACT 3B",
            real_image,
            format_prediction(realvlg),
            COLORS["realvlg"],
            panel_width,
            title_font,
            body_font,
        ),
    )
    header_height = 88
    gap = 8
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + gap * 2, header_height + panels[0].height),
        COLORS["background"],
    )
    draw = ImageDraw.Draw(canvas)
    heading = (
        f"{annotation['scene']}  |  object {annotation['object_id']}  |  "
        f"{annotation['sample_id']}"
    )
    draw.text((14, 12), heading, font=header_font, fill=COLORS["text"])
    description = str(annotation.get("description", ""))
    lines = wrap_pixels(description, description_font, canvas.width - 28, 2)
    for index, line in enumerate(lines):
        draw.text(
            (14, 43 + index * 20),
            line,
            font=description_font,
            fill=COLORS["muted"],
        )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * (panel_width + gap), header_height))
    return canvas


def outcome(ours: dict[str, Any], realvlg: dict[str, Any]) -> str:
    ours_correct = ours.get("gacc_corrected") == 1
    real_correct = realvlg.get("gacc_corrected") == 1
    if ours_correct and real_correct:
        return "both"
    if ours_correct:
        return "ours_only"
    if real_correct:
        return "real_only"
    return "neither"


def make_overview(
    split: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    count: int,
) -> Path:
    groups = {name: [] for name in ("ours_only", "real_only", "both", "neither")}
    for record in records:
        groups[record["outcome"]].append(record)
    selected: list[dict[str, Any]] = []
    per_group = max(1, count // len(groups))
    for name in groups:
        ranked = sorted(
            groups[name], key=lambda row: abs(row["iou_delta"]), reverse=True
        )
        selected.extend(ranked[:per_group])
    if len(selected) < count:
        selected_ids = {row["sample_id"] for row in selected}
        remaining = sorted(records, key=lambda row: abs(row["iou_delta"]), reverse=True)
        selected.extend(
            row for row in remaining if row["sample_id"] not in selected_ids
        )
    selected = selected[:count]

    thumbnails: list[Image.Image] = []
    for record in selected:
        with Image.open(output_dir / record["image"]) as source:
            image = source.convert("RGB")
        width = 840
        height = round(width * image.height / image.width)
        thumbnails.append(image.resize((width, height), Image.Resampling.LANCZOS))
    columns = 2
    gap = 8
    rows = math.ceil(len(thumbnails) / columns)
    cell_width = max(image.width for image in thumbnails)
    cell_height = max(image.height for image in thumbnails)
    grid = Image.new(
        "RGB",
        (
            columns * cell_width + (columns - 1) * gap,
            rows * cell_height + (rows - 1) * gap,
        ),
        COLORS["background"],
    )
    for index, image in enumerate(thumbnails):
        grid.paste(
            image,
            (
                (index % columns) * (cell_width + gap),
                (index // columns) * (cell_height + gap),
            ),
        )
    path = output_dir / f"{split}_overview.jpg"
    grid.save(path, quality=90, optimize=True)
    return path


def write_index(records: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for record in records:
        search = " ".join(
            (
                record["split"],
                record["scene"],
                str(record["object_id"]),
                record["description"],
            )
        ).lower()
        rows.append(
            f"""
            <article class="sample" data-split="{record["split"]}"
              data-outcome="{record["outcome"]}" data-search="{html.escape(search)}"
              data-delta="{record["iou_delta"]:.9f}">
              <a href="{html.escape(record["image"])}" target="_blank">
                <img src="{html.escape(record["image"])}" loading="lazy"
                  alt="{html.escape(record["sample_id"])}">
              </a>
              <div class="meta">
                <strong>{html.escape(record["scene"])} / object {html.escape(str(record["object_id"]))}</strong>
                <span>Ours {record["ours_iou"]:.3f} / {record["ours_gacc"]} &nbsp; RealVLG {record["real_iou"]:.3f} / {record["real_gacc"]}</span>
              </div>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Contact Model Visual Comparison</title>
  <style>
    :root {{ color-scheme: dark; font-family: Lato, Inter, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #12151a; color: #f4f6f8; }}
    header {{ position: sticky; top: 0; z-index: 3; background: #191d23; border-bottom: 1px solid #343b43; padding: 14px 20px; }}
    h1 {{ font-size: 20px; margin: 0 0 12px; letter-spacing: 0; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .segments {{ display: flex; border: 1px solid #46505a; border-radius: 6px; overflow: hidden; }}
    button {{ border: 0; border-right: 1px solid #46505a; background: #242a31; color: #dbe1e7; padding: 8px 12px; cursor: pointer; }}
    button:last-child {{ border-right: 0; }}
    button.active {{ background: #e8edf2; color: #16191d; }}
    input {{ min-width: 260px; border: 1px solid #46505a; border-radius: 6px; background: #101318; color: #fff; padding: 8px 11px; }}
    select {{ border: 1px solid #46505a; border-radius: 6px; background: #242a31; color: #fff; padding: 8px 10px; }}
    #count {{ color: #aeb8c2; margin-left: auto; }}
    main {{ padding: 18px; display: grid; grid-template-columns: repeat(auto-fill,minmax(560px,1fr)); gap: 14px; }}
    .sample {{ min-width: 0; border: 1px solid #303740; border-radius: 6px; background: #1b2026; overflow: hidden; }}
    .sample > a {{ display: block; overflow-x: auto; }}
    .sample img {{ width: 100%; aspect-ratio: 1696 / 489; object-fit: cover; display: block; }}
    .meta {{ display: flex; justify-content: space-between; gap: 12px; padding: 9px 11px; font-size: 13px; color: #aeb8c2; }}
    .meta strong {{ color: #f4f6f8; font-weight: 600; }}
    .hidden {{ display: none; }}
    @media (max-width: 720px) {{ header {{ padding: 12px; }} .toolbar {{ display: grid; grid-template-columns: 1fr; }} .segments {{ width: 100%; }} .segments button {{ flex: 1 1 0; min-width: 0; padding: 8px 4px; }} input, select {{ width: 100%; min-width: 0; max-width: 100%; }} main {{ grid-template-columns: 1fr; padding: 10px; }} .sample img {{ width: 900px; max-width: none; }} .meta {{ flex-direction: column; }} #count {{ width: 100%; margin-left: 0; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Contact Labels / Ours / RealVLG-R1</h1>
    <div class="toolbar">
      <div class="segments" id="splits">
        <button class="active" data-value="all">All</button><button data-value="seen">Seen</button><button data-value="similar">Similar</button><button data-value="novel">Novel</button>
      </div>
      <select id="outcome" aria-label="Outcome">
        <option value="all">All outcomes</option><option value="ours_only">Ours only correct</option><option value="real_only">RealVLG only correct</option><option value="both">Both correct</option><option value="neither">Neither correct</option>
      </select>
      <input id="search" type="search" placeholder="Search scene, object, description">
      <select id="sort" aria-label="Sort">
        <option value="default">Dataset order</option><option value="delta_desc">IoU delta: Ours first</option><option value="delta_asc">IoU delta: RealVLG first</option>
      </select>
      <span id="count"></span>
    </div>
  </header>
  <main id="samples">{"".join(rows)}</main>
  <script>
    const container = document.querySelector('#samples');
    const cards = [...document.querySelectorAll('.sample')];
    let split = 'all';
    document.querySelectorAll('#splits button').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('#splits button').forEach(item => item.classList.remove('active'));
      button.classList.add('active'); split = button.dataset.value; update();
    }}));
    document.querySelector('#outcome').addEventListener('change', update);
    document.querySelector('#search').addEventListener('input', update);
    document.querySelector('#sort').addEventListener('change', update);
    function update() {{
      const outcome = document.querySelector('#outcome').value;
      const query = document.querySelector('#search').value.trim().toLowerCase();
      const sort = document.querySelector('#sort').value;
      let shown = 0;
      cards.forEach(card => {{
        const visible = (split === 'all' || card.dataset.split === split) && (outcome === 'all' || card.dataset.outcome === outcome) && (!query || card.dataset.search.includes(query));
        card.classList.toggle('hidden', !visible); shown += visible ? 1 : 0;
      }});
      const direction = sort === 'delta_asc' ? 1 : -1;
      if (sort !== 'default') cards.sort((a,b) => direction * (Number(a.dataset.delta) - Number(b.dataset.delta)));
      else cards.sort((a,b) => Number(a.dataset.order) - Number(b.dataset.order));
      cards.forEach(card => container.appendChild(card));
      document.querySelector('#count').textContent = `${{shown}} / ${{cards.length}} samples`;
    }}
    cards.forEach((card,index) => card.dataset.order = index);
    update();
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.panel_width < 320:
        raise SystemExit("--panel-width must be at least 320")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_cache: dict[str, list[dict[str, Any]]] = {}
    manifest: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    global_index = 0
    for split in SPLITS:
        annotations = load_jsonl(
            args.annotations_dir / f"contact_{split}_official_exact.jsonl"
        )
        ours_rows = load_jsonl(args.ours_dir / f"{split}.predictions.jsonl")
        real_rows = load_jsonl(args.realvlg_dir / f"{split}.predictions.jsonl")
        if args.limit_per_split:
            annotations = annotations[: args.limit_per_split]
            ours_rows = ours_rows[: args.limit_per_split]
            real_rows = real_rows[: args.limit_per_split]
        if not (
            [row["sample_id"] for row in annotations]
            == [row["sample_id"] for row in ours_rows]
            == [row["sample_id"] for row in real_rows]
        ):
            raise RuntimeError(f"sample ordering mismatch for {split}")
        split_dir = args.output_dir / "items" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for local_index, (annotation, ours, realvlg) in enumerate(
            zip(annotations, ours_rows, real_rows, strict=True)
        ):
            obj = metadata_object(args.data_root, annotation, metadata_cache)
            image_path = resolve(args.data_root, str(annotation["image"]))
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            comparison = render_item(
                annotation,
                ours,
                realvlg,
                image,
                obj.get("bbox"),
                args.panel_width,
            )
            object_id = str(annotation.get("object_id", "unknown")).replace("/", "_")
            filename = f"{local_index:04d}_{annotation['scene']}_obj_{object_id}.jpg"
            path = split_dir / filename
            comparison.save(path, quality=90, optimize=True)
            relative_path = path.relative_to(args.output_dir).as_posix()
            record = {
                "index": global_index,
                "split": split,
                "sample_id": annotation["sample_id"],
                "scene": annotation["scene"],
                "object_id": annotation.get("object_id"),
                "description": annotation.get("description", ""),
                "image": relative_path,
                "outcome": outcome(ours, realvlg),
                "ours_iou": float(ours.get("iou") or 0.0),
                "real_iou": float(realvlg.get("iou") or 0.0),
                "iou_delta": float(ours.get("iou") or 0.0)
                - float(realvlg.get("iou") or 0.0),
                "ours_gacc": int(ours.get("gacc_corrected") or 0),
                "real_gacc": int(realvlg.get("gacc_corrected") or 0),
                "gt_candidates": len(
                    annotation.get("evaluation_contact_candidates_pixels", [])
                ),
                "gt_candidates_in_frame": in_frame_count(annotation),
            }
            manifest.append(record)
            split_records[split].append(record)
            global_index += 1

    for split, records in split_records.items():
        make_overview(
            split,
            records,
            args.output_dir,
            min(args.overview_count, len(records)),
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_index(manifest, args.output_dir)
    print(
        json.dumps(
            {
                "samples": len(manifest),
                "output": str(args.output_dir.resolve()),
                "index": str((args.output_dir / "index.html").resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
