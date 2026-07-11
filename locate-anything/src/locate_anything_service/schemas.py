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
    ]
    generation_mode: Literal["fast", "hybrid", "slow"]
    prompt: str
    raw_output: str
    image_width: int
    image_height: int
    boxes: list[Box] = Field(default_factory=list)
    points: list[Point] = Field(default_factory=list)
    generation_stats: dict[str, float | int] | None = None
    annotated_image_base64: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str
    device: str
    detail: str | None = None
