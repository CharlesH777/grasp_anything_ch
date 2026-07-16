from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from PIL import Image

from .collision_2d import (
    CollisionMaskProvider,
    evaluate_collision_2d,
    unknown_collision,
)
from .config import Settings
from .grasp_geometry import derive_grasp_geometry
from .parser import parse_grasp_output, parse_output
from .prompts import PromptMode, build_prompt
from .schemas import LocateResponse
from .visualization import annotate_image


class LocateAnythingRuntime:
    def __init__(
        self,
        settings: Settings,
        collision_mask_provider: CollisionMaskProvider | None = None,
    ) -> None:
        self.settings = settings
        self._worker = None
        self._processor = None
        self._tokenizer = None
        self._device = settings.device
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self._collision_mask_provider = collision_mask_provider

    @property
    def loaded(self) -> bool:
        return (
            self._worker is not None
            and self._processor is not None
            and self._tokenizer is not None
        )

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if self.settings.hf_home:
                os.environ.setdefault("HF_HOME", self.settings.hf_home)

            try:
                import torch
                from transformers import AutoModel, AutoProcessor, AutoTokenizer

                device = self._device
                if (
                    self.settings.device.startswith("cuda")
                    and not torch.cuda.is_available()
                ):
                    if not self.settings.allow_cpu:
                        raise RuntimeError(
                            "CUDA is unavailable. Set LOCATE_ALLOW_CPU=1 only for "
                            "debugging; production inference requires an NVIDIA GPU."
                        )
                    device = "cpu"

                model_dtype = (
                    torch.bfloat16
                    if device.startswith("cuda")
                    else torch.float32
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_id,
                    revision=self.settings.model_revision,
                    token=self.settings.hf_token,
                    trust_remote_code=True,
                )
                processor = AutoProcessor.from_pretrained(
                    self.settings.model_id,
                    revision=self.settings.model_revision,
                    token=self.settings.hf_token,
                    trust_remote_code=True,
                )
                worker = AutoModel.from_pretrained(
                    self.settings.model_id,
                    revision=self.settings.model_revision,
                    token=self.settings.hf_token,
                    torch_dtype=model_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to(device).eval()

                self._device = device
                self._tokenizer = tokenizer
                self._processor = processor
                self._worker = worker
                self._load_error = None
            except Exception as error:
                self._load_error = str(error)
                raise

    def predict(
        self,
        image_path: Path,
        query: str,
        mode: PromptMode = "ground_single",
        generation_mode: str | None = None,
        annotate: bool = False,
    ) -> LocateResponse:
        import torch

        if not self.loaded:
            self.load()

        prompt = build_prompt(query, mode)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self._processor.process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self._device)
        pixel_dtype = (
            torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(pixel_dtype)

        selected_generation_mode = generation_mode or self.settings.generation_mode
        generate_kwargs = {
            "pixel_values": inputs["pixel_values"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws"),
            "tokenizer": self._tokenizer,
            "max_new_tokens": self.settings.max_new_tokens,
            "use_cache": True,
            "generation_mode": selected_generation_mode,
            "temperature": self.settings.temperature,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "verbose": False,
        }
        if mode == "grasp_contact":
            generate_kwargs.update(
                do_sample=False,
                temperature=0.0,
                top_p=None,
            )
            generate_kwargs["geometry_type"] = "contact"
            generate_kwargs["image_size"] = (width, height)
            generate_kwargs["contact_coord_mass_threshold"] = (
                self.settings.contact_decode_coord_mass_threshold
            )
            if selected_generation_mode in {"fast", "hybrid"}:
                generate_kwargs["n_future_tokens"] = 6

        with self._lock, torch.no_grad():
            generation_started = time.perf_counter()
            response = self._worker.generate(**generate_kwargs)
            generation_seconds = time.perf_counter() - generation_started
        raw_output = response[0] if isinstance(response, tuple) else response
        if isinstance(raw_output, list):
            raw_output = raw_output[0]
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)
        generation_stats = {
            "generation_seconds": round(generation_seconds, 6),
            "box_count": raw_output.count("<box>"),
            "grasp_count": raw_output.count("<grasp>"),
        }

        boxes = []
        points = []
        grasps = []
        grasp_status = None
        grasp_parse_error = None
        if mode == "grasp_contact":
            parsed_grasp = parse_grasp_output(raw_output, width, height)
            grasp_status = parsed_grasp.status
            grasp_parse_error = parsed_grasp.error
            grasps = parsed_grasp.grasps
            if grasps:
                collision_started = time.perf_counter()
                collision = unknown_collision("no collision mask provider configured")
                if self._collision_mask_provider is not None:
                    try:
                        masks = self._collision_mask_provider(image, query)
                        geometry = derive_grasp_geometry(
                            grasps[0].contacts_1000, width, height
                        )
                        collision = evaluate_collision_2d(
                            geometry,
                            masks.obstacle_mask,
                            width,
                            height,
                            thickness_pixels=(
                                self.settings.collision_thickness_pixels
                            ),
                            collision_threshold=self.settings.collision_threshold,
                            outside_threshold=(
                                self.settings.collision_outside_threshold
                            ),
                            valid=masks.valid,
                            detail=masks.detail,
                        )
                    except Exception as error:
                        collision = unknown_collision(
                            f"collision mask provider failed: {error}"
                        )
                grasps[0] = grasps[0].model_copy(
                    update={
                        "collision_2d_status": collision.status,
                        "collision_proxy_thickness_pixels": (
                            collision.thickness_pixels
                        ),
                        "collision_ratio_2d": collision.collision_ratio,
                        "outside_ratio_2d": collision.outside_ratio,
                        "clearance_pixels_2d": collision.clearance_pixels,
                        "collision_detail": collision.detail,
                    }
                )
                generation_stats["collision_check_seconds"] = round(
                    time.perf_counter() - collision_started, 6
                )
        else:
            parsed = parse_output(raw_output, width, height)
            boxes = parsed.boxes
            points = parsed.points

        annotated = (
            annotate_image(image, boxes, points, grasps) if annotate else None
        )
        return LocateResponse(
            model=self.settings.model_id,
            mode=mode,
            generation_mode=selected_generation_mode,
            prompt=prompt,
            raw_output=raw_output,
            image_width=width,
            image_height=height,
            boxes=boxes,
            points=points,
            grasps=grasps,
            grasp_status=grasp_status,
            grasp_parse_error=grasp_parse_error,
            generation_stats=generation_stats,
            annotated_image_base64=annotated,
        )
