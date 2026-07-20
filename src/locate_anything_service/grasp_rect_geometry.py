from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .grasp_geometry import Polygon2D

GraspRectTokens = tuple[float, float, float, float]
GraspRectPixels = tuple[float, float, float, float]
Points8 = tuple[float, float, float, float, float, float, float, float]

ANGLE_BIN_COUNT = 1001
TOKEN_MAX = 1000
DEFAULT_GRIPPER_DEPTH_PIXELS = 40.0
DEFAULT_MINIMUM_WIDTH_DIAGONAL = 1e-4


@dataclass(frozen=True, slots=True)
class GraspRectangleGeometry:
    parameters_1000: GraspRectTokens
    center_1000: tuple[float, float]
    center_normalized: tuple[float, float]
    center_pixels_float: tuple[float, float]
    angle_token: float
    angle_degrees_image: float
    angle_radians_image: float
    opening_width_token: float
    opening_width_pixels: float
    opening_width_diagonal_normalized: float
    gripper_depth_pixels: float
    rectangle_points_pixels_float: Points8

    @property
    def center_pixels(self) -> tuple[int, int]:
        return tuple(round(value) for value in self.center_pixels_float)

    @property
    def rectangle_points_pixels(self) -> tuple[int, ...]:
        return tuple(round(value) for value in self.rectangle_points_pixels_float)

    @property
    def polygon_pixels_float(self) -> Polygon2D:
        values = self.rectangle_points_pixels_float
        return tuple(
            (values[index], values[index + 1])
            for index in range(0, len(values), 2)
        )


def _finite_values(
    values: Iterable[float], expected: int, name: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _validate_image_size(image_width: int, image_height: int) -> float:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size: {image_width}x{image_height}")
    return math.hypot(image_width, image_height)


def _validate_minimum_width(minimum_width_diagonal: float) -> float:
    value = float(minimum_width_diagonal)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("minimum_width_diagonal must be finite and non-negative")
    return value


def canonical_angle_degrees(theta_degrees: float) -> float:
    value = float(theta_degrees)
    if not math.isfinite(value):
        raise ValueError("theta_degrees must be finite")
    return value % 180.0


def encode_angle_token(theta_degrees: float) -> int:
    theta_180 = canonical_angle_degrees(theta_degrees)
    return int(math.floor(theta_180 * ANGLE_BIN_COUNT / 180.0 + 0.5)) % (
        ANGLE_BIN_COUNT
    )


def decode_angle_token(angle_token: float) -> float:
    value = float(angle_token)
    if not math.isfinite(value) or not 0.0 <= value <= TOKEN_MAX:
        raise ValueError("angle token must be in [0, 1000]")
    return value * 180.0 / ANGLE_BIN_COUNT


def rect_to_points8(
    center_x: float,
    center_y: float,
    theta_degrees: float,
    width_pixels: float,
    gripper_depth_pixels: float = DEFAULT_GRIPPER_DEPTH_PIXELS,
) -> Points8:
    center_x, center_y, width_pixels, gripper_depth_pixels = _finite_values(
        (center_x, center_y, width_pixels, gripper_depth_pixels),
        4,
        "grasp rectangle parameters",
    )
    if width_pixels <= 0.0:
        raise ValueError("grasp rectangle width must be positive")
    if gripper_depth_pixels <= 0.0:
        raise ValueError("gripper_depth_pixels must be positive")

    theta_radians = math.radians(canonical_angle_degrees(theta_degrees))
    unit_x = math.cos(theta_radians)
    unit_y = math.sin(theta_radians)
    normal_x = -unit_y
    normal_y = unit_x
    half_width = width_pixels * 0.5
    half_depth = gripper_depth_pixels * 0.5

    return (
        center_x - half_width * unit_x + half_depth * normal_x,
        center_y - half_width * unit_y + half_depth * normal_y,
        center_x + half_width * unit_x + half_depth * normal_x,
        center_y + half_width * unit_y + half_depth * normal_y,
        center_x + half_width * unit_x - half_depth * normal_x,
        center_y + half_width * unit_y - half_depth * normal_y,
        center_x - half_width * unit_x - half_depth * normal_x,
        center_y - half_width * unit_y - half_depth * normal_y,
    )


def points8_to_rect(points8: Iterable[float]) -> GraspRectPixels:
    values = _finite_values(points8, 8, "points8")
    corners = tuple(
        (values[index], values[index + 1]) for index in range(0, 8, 2)
    )
    center_x = sum(point[0] for point in corners) * 0.25
    center_y = sum(point[1] for point in corners) * 0.25
    left_center = (
        (corners[0][0] + corners[3][0]) * 0.5,
        (corners[0][1] + corners[3][1]) * 0.5,
    )
    right_center = (
        (corners[1][0] + corners[2][0]) * 0.5,
        (corners[1][1] + corners[2][1]) * 0.5,
    )
    width_pixels = math.hypot(
        right_center[0] - left_center[0],
        right_center[1] - left_center[1],
    )
    if width_pixels <= 1e-9:
        raise ValueError("grasp rectangle width must be positive")
    theta_degrees = canonical_angle_degrees(
        math.degrees(
            math.atan2(
                corners[1][1] - corners[0][1],
                corners[1][0] - corners[0][0],
            )
        )
    )
    return center_x, center_y, theta_degrees, width_pixels


def encode_grasp_rectangle_pixels(
    center_x: float,
    center_y: float,
    theta_degrees: float,
    width_pixels: float,
    image_width: int,
    image_height: int,
    *,
    minimum_width_diagonal: float = DEFAULT_MINIMUM_WIDTH_DIAGONAL,
) -> tuple[int, int, int, int]:
    center_x, center_y, width_pixels = _finite_values(
        (center_x, center_y, width_pixels), 3, "grasp rectangle parameters"
    )
    image_diagonal = _validate_image_size(image_width, image_height)
    minimum = _validate_minimum_width(minimum_width_diagonal)
    if not 0.0 <= center_x <= image_width or not 0.0 <= center_y <= image_height:
        raise ValueError("grasp rectangle center must be inside the image")
    width_diagonal = width_pixels / image_diagonal
    if width_diagonal <= minimum:
        raise ValueError("grasp rectangle width must exceed minimum_width_diagonal")

    center_x_token = int(math.floor(center_x * TOKEN_MAX / image_width + 0.5))
    center_y_token = int(math.floor(center_y * TOKEN_MAX / image_height + 0.5))
    width_token = int(math.floor(width_diagonal * TOKEN_MAX + 0.5))
    if not 0 <= width_token <= TOKEN_MAX:
        raise ValueError("grasp rectangle width is outside the token representation")
    return (
        center_x_token,
        center_y_token,
        encode_angle_token(theta_degrees),
        width_token,
    )


def derive_grasp_rectangle_geometry(
    parameters_1000: Iterable[float],
    image_width: int,
    image_height: int,
    *,
    gripper_depth_pixels: float = DEFAULT_GRIPPER_DEPTH_PIXELS,
    minimum_width_diagonal: float = DEFAULT_MINIMUM_WIDTH_DIAGONAL,
) -> GraspRectangleGeometry:
    values = _finite_values(parameters_1000, 4, "grasp rectangle parameters")
    image_diagonal = _validate_image_size(image_width, image_height)
    minimum = _validate_minimum_width(minimum_width_diagonal)
    if not all(0.0 <= value <= TOKEN_MAX for value in values):
        raise ValueError("grasp rectangle token values must be in [0, 1000]")
    depth = float(gripper_depth_pixels)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("gripper_depth_pixels must be positive and finite")

    center_x_token, center_y_token, angle_token, width_token = values
    center_normalized = (
        center_x_token / TOKEN_MAX,
        center_y_token / TOKEN_MAX,
    )
    center_pixels = (
        center_normalized[0] * image_width,
        center_normalized[1] * image_height,
    )
    angle_degrees = decode_angle_token(angle_token)
    width_diagonal = width_token / TOKEN_MAX
    if width_diagonal <= minimum:
        raise ValueError("grasp rectangle width must exceed minimum_width_diagonal")
    width_pixels = width_diagonal * image_diagonal
    points8 = rect_to_points8(
        center_pixels[0],
        center_pixels[1],
        angle_degrees,
        width_pixels,
        depth,
    )
    return GraspRectangleGeometry(
        parameters_1000=values,
        center_1000=(center_x_token, center_y_token),
        center_normalized=center_normalized,
        center_pixels_float=center_pixels,
        angle_token=angle_token,
        angle_degrees_image=angle_degrees,
        angle_radians_image=math.radians(angle_degrees),
        opening_width_token=width_token,
        opening_width_pixels=width_pixels,
        opening_width_diagonal_normalized=width_diagonal,
        gripper_depth_pixels=depth,
        rectangle_points_pixels_float=points8,
    )
