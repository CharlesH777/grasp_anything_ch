from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import Box, Point

OBJECT_PATTERN = re.compile(
    r"<(?P<tag>ref|object|c)>(.*?)</(?P=tag)>", re.DOTALL
)
GEOMETRY_PATTERN = re.compile(r"<box>((?:<-?\d+(?:\.\d+)?>){2,4})</box>")
NUMBER_PATTERN = re.compile(r"<(-?\d+(?:\.\d+)?)>")


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    boxes: list[Box]
    points: list[Point]


def _label_before(text: str, position: int) -> str | None:
    matches = list(OBJECT_PATTERN.finditer(text, 0, position))
    if not matches:
        return None
    label = matches[-1].group(2).strip()
    return label or None


def _to_normalized(value: float) -> float:
    normalized = value / 1000.0
    return min(1.0, max(0.0, normalized))


def parse_output(text: str, image_width: int, image_height: int) -> ParsedOutput:
    boxes: list[Box] = []
    points: list[Point] = []

    for match in GEOMETRY_PATTERN.finditer(text):
        payload = match.group(1)
        values = [float(value) for value in NUMBER_PATTERN.findall(payload)]
        label = _label_before(text, match.start())

        if len(values) >= 4:
            raw = tuple(values[:4])
            normalized = tuple(_to_normalized(value) for value in raw)
            x1, y1, x2, y2 = normalized
            boxes.append(
                Box(
                    label=label,
                    coordinates_1000=raw,
                    normalized=normalized,
                    pixels=(
                        round(x1 * image_width),
                        round(y1 * image_height),
                        round(x2 * image_width),
                        round(y2 * image_height),
                    ),
                )
            )
        elif len(values) >= 2:
            raw_point = tuple(values[:2])
            normalized_point = tuple(_to_normalized(value) for value in raw_point)
            x, y = normalized_point
            points.append(
                Point(
                    label=label,
                    coordinates_1000=raw_point,
                    normalized=normalized_point,
                    pixels=(round(x * image_width), round(y * image_height)),
                )
            )

    return ParsedOutput(boxes=boxes, points=points)
