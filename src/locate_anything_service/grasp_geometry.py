from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Point2D = tuple[float, float]
Polygon2D = tuple[Point2D, ...]
ContactCoordinates = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GraspGeometry:
    contacts_1000: ContactCoordinates
    contacts_normalized: ContactCoordinates
    contacts_pixels_float: ContactCoordinates
    center_1000: Point2D
    center_pixels_float: Point2D
    angle_radians_image: float
    opening_width_pixels: float
    opening_width_diagonal_normalized: float

    @property
    def contacts_pixels(self) -> tuple[int, int, int, int]:
        return tuple(round(value) for value in self.contacts_pixels_float)

    @property
    def center_pixels(self) -> tuple[int, int]:
        return tuple(round(value) for value in self.center_pixels_float)


def derive_grasp_geometry(
    contacts_1000: Iterable[float],
    image_width: int,
    image_height: int,
) -> GraspGeometry:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size: {image_width}x{image_height}")

    values = tuple(float(value) for value in contacts_1000)
    if len(values) != 4:
        raise ValueError(f"expected four contact coordinates, got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("contact coordinates must be finite")
    if not all(0.0 <= value <= 1000.0 for value in values):
        raise ValueError("contact coordinates must be in [0, 1000]")

    x1, y1, x2, y2 = values
    normalized = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
    pixels = (
        normalized[0] * image_width,
        normalized[1] * image_height,
        normalized[2] * image_width,
        normalized[3] * image_height,
    )
    dx = pixels[2] - pixels[0]
    dy = pixels[3] - pixels[1]
    width_pixels = math.hypot(dx, dy)
    if width_pixels <= 1e-9:
        raise ValueError("contact points must not coincide")

    image_diagonal = math.hypot(image_width, image_height)
    return GraspGeometry(
        contacts_1000=values,
        contacts_normalized=normalized,
        contacts_pixels_float=pixels,
        center_1000=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        center_pixels_float=(
            (pixels[0] + pixels[2]) * 0.5,
            (pixels[1] + pixels[3]) * 0.5,
        ),
        angle_radians_image=math.atan2(dy, dx) % math.pi,
        opening_width_pixels=width_pixels,
        opening_width_diagonal_normalized=width_pixels / image_diagonal,
    )


def grasp_rectangle(
    contacts_pixels: Iterable[float],
    thickness_pixels: float = 80.0,
) -> Polygon2D:
    values = tuple(float(value) for value in contacts_pixels)
    if len(values) != 4:
        raise ValueError("contacts_pixels must contain four values")
    if not math.isfinite(thickness_pixels) or thickness_pixels <= 0:
        raise ValueError("thickness_pixels must be positive and finite")

    x1, y1, x2, y2 = values
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("cannot construct a rectangle from coincident contacts")

    normal_x = -dy / length
    normal_y = dx / length
    offset_x = normal_x * thickness_pixels * 0.5
    offset_y = normal_y * thickness_pixels * 0.5
    return (
        (x1 - offset_x, y1 - offset_y),
        (x1 + offset_x, y1 + offset_y),
        (x2 + offset_x, y2 + offset_y),
        (x2 - offset_x, y2 - offset_y),
    )


def polygon_area(polygon: Iterable[Point2D]) -> float:
    points = tuple(polygon)
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points, points[1:] + points[:1], strict=True
            )
        )
    ) * 0.5


def _signed_polygon_area(polygon: Polygon2D) -> float:
    return sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(
            polygon, polygon[1:] + polygon[:1], strict=True
        )
    ) * 0.5


def _inside(point: Point2D, start: Point2D, end: Point2D, sign: float) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (
        end[1] - start[1]
    ) * (point[0] - start[0])
    return cross * sign >= -1e-9


def _line_intersection(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> Point2D:
    x1, y1 = first_start
    x2, y2 = first_end
    x3, y3 = second_start
    x4, y4 = second_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) <= 1e-12:
        return first_end
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    return (
        (
            determinant_first * (x3 - x4)
            - (x1 - x2) * determinant_second
        )
        / denominator,
        (
            determinant_first * (y3 - y4)
            - (y1 - y2) * determinant_second
        )
        / denominator,
    )


def convex_polygon_intersection(
    subject_polygon: Iterable[Point2D],
    clip_polygon: Iterable[Point2D],
) -> Polygon2D:
    output = tuple(subject_polygon)
    clip = tuple(clip_polygon)
    if len(output) < 3 or len(clip) < 3:
        return ()

    sign = 1.0 if _signed_polygon_area(clip) >= 0 else -1.0
    for clip_start, clip_end in zip(clip, clip[1:] + clip[:1], strict=True):
        input_points = output
        output_list: list[Point2D] = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _inside(previous, clip_start, clip_end, sign)
        for current in input_points:
            current_inside = _inside(current, clip_start, clip_end, sign)
            if current_inside:
                if not previous_inside:
                    output_list.append(
                        _line_intersection(previous, current, clip_start, clip_end)
                    )
                output_list.append(current)
            elif previous_inside:
                output_list.append(
                    _line_intersection(previous, current, clip_start, clip_end)
                )
            previous = current
            previous_inside = current_inside
        output = tuple(output_list)
    return output


def polygon_iou(first: Iterable[Point2D], second: Iterable[Point2D]) -> float:
    first_polygon = tuple(first)
    second_polygon = tuple(second)
    intersection = polygon_area(
        convex_polygon_intersection(first_polygon, second_polygon)
    )
    union = polygon_area(first_polygon) + polygon_area(second_polygon) - intersection
    return intersection / union if union > 1e-12 else 0.0


def polygon_inside_image_area(
    polygon: Iterable[Point2D], image_width: int, image_height: int
) -> float:
    image_polygon: Polygon2D = (
        (0.0, 0.0),
        (float(image_width), 0.0),
        (float(image_width), float(image_height)),
        (0.0, float(image_height)),
    )
    return polygon_area(convex_polygon_intersection(tuple(polygon), image_polygon))


def angular_error_degrees(first_radians: float, second_radians: float) -> float:
    delta = (first_radians - second_radians + math.pi * 0.5) % math.pi
    return math.degrees(abs(delta - math.pi * 0.5))
