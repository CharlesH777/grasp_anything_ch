from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    model_id: str = "nvidia/LocateAnything-3B"
    model_revision: str = "c32291ca5e996f5a7a485845b4f57a233936bba0"
    device: str = "cuda"
    hf_token: str | None = None
    hf_home: str | None = None
    load_model_on_startup: bool = True
    allow_cpu: bool = False
    require_grasp_checkpoint: bool = False
    generation_mode: str = "hybrid"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    contact_decode_coord_mass_threshold: float = 1e-4
    collision_thickness_pixels: float = 80.0
    collision_threshold: float = 0.0
    collision_outside_threshold: float = 0.0
    max_upload_mb: int = 25
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    def __post_init__(self) -> None:
        for name, value in (
            (
                "contact_decode_coord_mass_threshold",
                self.contact_decode_coord_mass_threshold,
            ),
            ("collision_threshold", self.collision_threshold),
            ("collision_outside_threshold", self.collision_outside_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.collision_thickness_pixels <= 0:
            raise ValueError("collision_thickness_pixels must be positive")

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            model_id=os.getenv("LOCATE_MODEL_ID", defaults.model_id),
            model_revision=os.getenv(
                "LOCATE_MODEL_REVISION", defaults.model_revision
            ),
            device=os.getenv("LOCATE_DEVICE", defaults.device),
            hf_token=os.getenv("HF_TOKEN"),
            hf_home=os.getenv("HF_HOME"),
            load_model_on_startup=_as_bool(
                os.getenv("LOCATE_LOAD_MODEL_ON_STARTUP"), True
            ),
            allow_cpu=_as_bool(os.getenv("LOCATE_ALLOW_CPU"), False),
            require_grasp_checkpoint=_as_bool(
                os.getenv("LOCATE_REQUIRE_GRASP_CHECKPOINT"), False
            ),
            generation_mode=os.getenv("LOCATE_GENERATION_MODE", "hybrid"),
            max_new_tokens=int(os.getenv("LOCATE_MAX_NEW_TOKENS", "2048")),
            temperature=float(os.getenv("LOCATE_TEMPERATURE", "0.7")),
            contact_decode_coord_mass_threshold=float(
                os.getenv("LOCATE_CONTACT_COORD_MASS_THRESHOLD", "0.0001")
            ),
            collision_thickness_pixels=float(
                os.getenv("LOCATE_COLLISION_THICKNESS_PIXELS", "80")
            ),
            collision_threshold=float(
                os.getenv("LOCATE_COLLISION_THRESHOLD", "0")
            ),
            collision_outside_threshold=float(
                os.getenv("LOCATE_COLLISION_OUTSIDE_THRESHOLD", "0")
            ),
            max_upload_mb=int(os.getenv("LOCATE_MAX_UPLOAD_MB", "25")),
            host=os.getenv("LOCATE_HOST", defaults.host),
            port=int(os.getenv("LOCATE_PORT", "8000")),
            log_level=os.getenv("LOCATE_LOG_LEVEL", defaults.log_level),
        )
