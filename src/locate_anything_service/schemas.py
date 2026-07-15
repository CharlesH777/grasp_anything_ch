from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Box(BaseModel):
    label: str | None = None
    coordinates_1000: tuple[float, float, float, float]
    normalized: tuple[float, float, float, float]
    pixels: tuple[int, int, int, int]


class Point(BaseModel):
    label: str | None = None
    coordinates_1000: tuple[float, float]
    normalized: tuple[float, float]
    pixels: tuple[int, int]


class GraspContact(BaseModel):
    label: str | None = None
    contacts_1000: tuple[float, float, float, float]
    contacts_normalized: tuple[float, float, float, float]
    contacts_pixels: tuple[int, int, int, int]
    center_1000: tuple[float, float]
    center_pixels: tuple[int, int]
    angle_radians_image: float
    opening_width_pixels: float
    opening_width_diagonal_normalized: float
    collision_2d_status: Literal["free", "collision", "unknown"] = "unknown"
    collision_proxy_thickness_pixels: float | None = None
    collision_ratio_2d: float | None = None
    outside_ratio_2d: float | None = None
    clearance_pixels_2d: float | None = None
    collision_detail: str | None = None


class LocateResponse(BaseModel):
    model: str
    mode: Literal[
        "raw",
        "detect",
        "ground_single",
        "ground_multi",
        "ground_text",
        "detect_text",
        "gui_box",
        "gui_point",
        "point",
        "grasp_contact",
    ]
    generation_mode: Literal["fast", "hybrid", "slow"]
    prompt: str
    raw_output: str
    image_width: int
    image_height: int
    boxes: list[Box] = Field(default_factory=list)
    points: list[Point] = Field(default_factory=list)
    grasps: list[GraspContact] = Field(default_factory=list)
    grasp_status: Literal["ok", "none", "invalid"] | None = None
    grasp_parse_error: str | None = None
    generation_stats: dict[str, float | int] | None = None
    annotated_image_base64: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str
    device: str
    detail: str | None = None
