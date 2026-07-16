from __future__ import annotations

import re
from dataclasses import dataclass

from .grasp_geometry import derive_grasp_geometry
from .schemas import Box, GraspContact, Point

OBJECT_PATTERN = re.compile(
    r"<(?P<tag>ref|object|c)>(.*?)</(?P=tag)>", re.DOTALL
)
GEOMETRY_PATTERN = re.compile(r"<box>((?:<-?\d+(?:\.\d+)?>){2,4})</box>")
NUMBER_PATTERN = re.compile(r"<(-?\d+(?:\.\d+)?)>")
GRASP_START_PATTERN = re.compile(r"<grasp>", re.IGNORECASE)
GRASP_END_PATTERN = re.compile(r"</grasp>", re.IGNORECASE)
GRASP_PAYLOAD_PATTERN = re.compile(r"<grasp>(.*?)</grasp>", re.IGNORECASE | re.DOTALL)
GRASP_COORD_PATTERN = re.compile(
    r"\s*<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
    r"<\s*(-?\d+(?:\.\d+)?)\s*>\s*"
)
NONE_PATTERN = re.compile(r"\s*none\s*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    boxes: list[Box]
    points: list[Point]


@dataclass(frozen=True, slots=True)
class ParsedGraspOutput:
    status: str
    grasps: list[GraspContact]
    error: str | None = None


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


def _invalid_grasp(error: str) -> ParsedGraspOutput:
    return ParsedGraspOutput(status="invalid", grasps=[], error=error)


def parse_grasp_output(
    text: str, image_width: int, image_height: int
) -> ParsedGraspOutput:
    payload_matches = list(GRASP_PAYLOAD_PATTERN.finditer(text))
    if re.search(r"</?box>", text, re.IGNORECASE):
        return _invalid_grasp("legacy <box> blocks are not valid grasp output")
    if (
        len(payload_matches) != 1
        or len(GRASP_START_PATTERN.findall(text)) != 1
        or len(GRASP_END_PATTERN.findall(text)) != 1
    ):
        return _invalid_grasp("expected exactly one complete <grasp> block")

    match = payload_matches[0]
    outside_block = text[: match.start()] + text[match.end() :]
    outside_block = OBJECT_PATTERN.sub("", outside_block)
    outside_block = re.sub(
        r"<\|(?:im_end|endoftext|eot_id)\|>|</s>",
        "",
        outside_block,
        flags=re.IGNORECASE,
    )
    if NUMBER_PATTERN.search(outside_block):
        return _invalid_grasp("coordinate token found outside the <grasp> block")

    payload = match.group(1)
    if NONE_PATTERN.fullmatch(payload):
        return ParsedGraspOutput(status="none", grasps=[])

    coordinate_match = GRASP_COORD_PATTERN.fullmatch(payload)
    if coordinate_match is None:
        return _invalid_grasp(
            "expected <grasp><x1><y1><x2><y2></grasp> or <grasp>none</grasp>"
        )

    values = tuple(float(value) for value in coordinate_match.groups())
    try:
        geometry = derive_grasp_geometry(values, image_width, image_height)
    except ValueError as error:
        return _invalid_grasp(str(error))

    label = _label_before(text, match.start())
    grasp = GraspContact(
        label=label,
        contacts_1000=geometry.contacts_1000,
        contacts_normalized=geometry.contacts_normalized,
        contacts_pixels=geometry.contacts_pixels,
        center_1000=geometry.center_1000,
        center_pixels=geometry.center_pixels,
        angle_radians_image=geometry.angle_radians_image,
        opening_width_pixels=geometry.opening_width_pixels,
        opening_width_diagonal_normalized=(
            geometry.opening_width_diagonal_normalized
        ),
        collision_2d_status="unknown",
        collision_detail="no reliable obstacle mask",
    )
    return ParsedGraspOutput(status="ok", grasps=[grasp])
