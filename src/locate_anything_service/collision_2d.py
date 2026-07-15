from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageChops, ImageDraw

from .grasp_geometry import (
    GraspGeometry,
    grasp_rectangle,
    polygon_area,
    polygon_inside_image_area,
)


@dataclass(frozen=True, slots=True)
class CollisionMasks:
    obstacle_mask: Image.Image | None
    valid: bool
    detail: str | None = None


class CollisionMaskProvider(Protocol):
    def __call__(self, image: Image.Image, query: str) -> CollisionMasks: ...


@dataclass(frozen=True, slots=True)
class Collision2DResult:
    status: str
    thickness_pixels: float | None = None
    collision_ratio: float | None = None
    outside_ratio: float | None = None
    clearance_pixels: float | None = None
    detail: str | None = None


def unknown_collision(detail: str) -> Collision2DResult:
    return Collision2DResult(status="unknown", detail=detail)


def _nonzero_pixels(image: Image.Image) -> int:
    histogram = image.histogram()
    return sum(histogram[1:])


def _binary_mask(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size != size:
        raise ValueError(
            f"collision mask size {image.size} does not match image size {size}"
        )
    return image.convert("L").point(lambda value: 255 if value > 0 else 0, mode="1")


def _clearance_pixels(
    polygon_mask: Image.Image, obstacle_mask: Image.Image
) -> float | None:
    if _nonzero_pixels(obstacle_mask) == 0:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    obstacle = np.asarray(obstacle_mask.convert("L")) > 0
    polygon = np.asarray(polygon_mask.convert("L")) > 0
    if not polygon.any():
        return None
    free = (~obstacle).astype("uint8")
    distances = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    return float(distances[polygon].min())


def evaluate_collision_2d(
    geometry: GraspGeometry,
    obstacle_mask: Image.Image | None,
    image_width: int,
    image_height: int,
    *,
    thickness_pixels: float = 80.0,
    collision_threshold: float = 0.0,
    outside_threshold: float = 0.0,
    valid: bool = True,
    detail: str | None = None,
) -> Collision2DResult:
    if not valid or obstacle_mask is None:
        return unknown_collision(detail or "no reliable obstacle mask")
    if not 0.0 <= collision_threshold <= 1.0:
        raise ValueError("collision_threshold must be in [0, 1]")
    if not 0.0 <= outside_threshold <= 1.0:
        raise ValueError("outside_threshold must be in [0, 1]")

    polygon = grasp_rectangle(
        geometry.contacts_pixels_float, thickness_pixels=thickness_pixels
    )
    full_area = polygon_area(polygon)
    if full_area <= 1e-9 or not math.isfinite(full_area):
        raise ValueError("invalid collision polygon")

    size = (image_width, image_height)
    obstacle = _binary_mask(obstacle_mask, size)
    polygon_mask = Image.new("1", size, 0)
    ImageDraw.Draw(polygon_mask).polygon(polygon, fill=1)
    polygon_pixels = _nonzero_pixels(polygon_mask)
    if polygon_pixels == 0:
        raise ValueError("collision polygon has no rasterized pixels")
    intersection = ImageChops.logical_and(polygon_mask, obstacle)
    intersection_pixels = _nonzero_pixels(intersection)

    collision_ratio = intersection_pixels / polygon_pixels
    inside_area = polygon_inside_image_area(polygon, image_width, image_height)
    outside_ratio = min(1.0, max(0.0, (full_area - inside_area) / full_area))
    collides = (
        collision_ratio > collision_threshold
        or outside_ratio > outside_threshold
    )
    clearance = 0.0 if intersection_pixels else _clearance_pixels(
        polygon_mask, obstacle
    )
    return Collision2DResult(
        status="collision" if collides else "free",
        thickness_pixels=thickness_pixels,
        collision_ratio=collision_ratio,
        outside_ratio=outside_ratio,
        clearance_pixels=clearance,
        detail=detail,
    )
