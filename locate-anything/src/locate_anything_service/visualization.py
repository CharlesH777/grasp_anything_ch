from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw, ImageFont

from .schemas import Box, Point


def annotate_image(image: Image.Image, boxes: list[Box], points: list[Point]) -> str:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for box in boxes:
        draw.rectangle(box.pixels, outline="#00ff66", width=3)
        if box.label:
            x1, y1, _, _ = box.pixels
            label_box = draw.textbbox((x1, y1), box.label, font=font)
            draw.rectangle(label_box, fill="#00ff66")
            draw.text((x1, y1), box.label, fill="black", font=font)

    for point in points:
        x, y = point.pixels
        radius = 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#ff3355")
        if point.label:
            draw.text((x + radius + 2, y), point.label, fill="#ff3355", font=font)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
