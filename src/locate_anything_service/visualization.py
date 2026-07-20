from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw, ImageFont

from .grasp_geometry import grasp_rectangle
from .schemas import Box, GraspContact, GraspRectangle, Point


def annotate_image(
    image: Image.Image,
    boxes: list[Box],
    points: list[Point],
    grasps: list[GraspContact] | None = None,
    grasp_rectangles: list[GraspRectangle] | None = None,
) -> str:
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

    for grasp in grasps or []:
        status_colors = {
            "free": "#00b86b",
            "collision": "#e5484d",
            "unknown": "#f0a020",
        }
        color = status_colors[grasp.collision_2d_status]
        thickness = grasp.collision_proxy_thickness_pixels or 80.0
        polygon = grasp_rectangle(grasp.contacts_pixels, thickness)
        draw.polygon(polygon, outline=color, width=2)

        x1, y1, x2, y2 = grasp.contacts_pixels
        draw.line((x1, y1, x2, y2), fill=color, width=3)
        for x, y in ((x1, y1), (x2, y2)):
            radius = 6
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline="white",
                width=1,
            )
        center_x, center_y = grasp.center_pixels
        radius = 3
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill="white",
            outline=color,
            width=1,
        )
        label = grasp.label or "grasp"
        draw.text(
            (center_x + 6, center_y + 6),
            f"{label}: {grasp.collision_2d_status}",
            fill=color,
            font=font,
        )

    for rectangle in grasp_rectangles or []:
        status_colors = {
            "free": "#00a878",
            "collision": "#d64045",
            "unknown": "#3478c8",
        }
        color = status_colors[rectangle.collision_2d_status]
        values = rectangle.rectangle_points_pixels_float
        polygon = tuple(
            (values[index], values[index + 1])
            for index in range(0, len(values), 2)
        )
        draw.polygon(polygon, outline=color, width=3)
        center_x, center_y = rectangle.center_pixels
        radius = 4
        draw.line(
            (center_x - radius, center_y, center_x + radius, center_y),
            fill=color,
            width=2,
        )
        draw.line(
            (center_x, center_y - radius, center_x, center_y + radius),
            fill=color,
            width=2,
        )
        draw.text(
            (center_x + 6, center_y + 6),
            f"{rectangle.label or 'grasp rect'}: {rectangle.collision_2d_status}",
            fill=color,
            font=font,
        )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
