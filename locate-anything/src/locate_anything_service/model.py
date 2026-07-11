from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from PIL import Image

from .config import Settings
from .parser import parse_output
from .prompts import PromptMode, build_prompt
from .schemas import LocateResponse
from .visualization import annotate_image


class LocateAnythingRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._worker = None
        self._processor = None
        self._tokenizer = None
        self._device = settings.device
        self._lock = threading.Lock()
        self._load_error: str | None = None

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
        if self.loaded:
            return
        if self.settings.hf_home:
            os.environ.setdefault("HF_HOME", self.settings.hf_home)

        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            if (
                self.settings.device.startswith("cuda")
                and not torch.cuda.is_available()
            ):
                if not self.settings.allow_cpu:
                    raise RuntimeError(
                        "CUDA is unavailable. Set LOCATE_ALLOW_CPU=1 only for "
                        "debugging; production inference requires an NVIDIA GPU."
                    )
                self._device = "cpu"

            model_dtype = (
                torch.bfloat16
                if self._device.startswith("cuda")
                else torch.float32
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_id,
                revision=self.settings.model_revision,
                token=self.settings.hf_token,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.settings.model_id,
                revision=self.settings.model_revision,
                token=self.settings.hf_token,
                trust_remote_code=True,
            )
            self._worker = AutoModel.from_pretrained(
                self.settings.model_id,
                revision=self.settings.model_revision,
                token=self.settings.hf_token,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self._device).eval()
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
        with self._lock, torch.no_grad():
            generation_started = time.perf_counter()
            response = self._worker.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self._tokenizer,
                max_new_tokens=self.settings.max_new_tokens,
                use_cache=True,
                generation_mode=selected_generation_mode,
                temperature=self.settings.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                verbose=False,
            )
            generation_seconds = time.perf_counter() - generation_started
        raw_output = response[0] if isinstance(response, tuple) else response
        if isinstance(raw_output, list):
            raw_output = raw_output[0]
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)
        generation_stats = {
            "generation_seconds": round(generation_seconds, 6),
            "box_count": raw_output.count("<box>"),
        }

        parsed = parse_output(raw_output, width, height)
        annotated = (
            annotate_image(image, parsed.boxes, parsed.points) if annotate else None
        )
        return LocateResponse(
            model=self.settings.model_id,
            mode=mode,
            generation_mode=selected_generation_mode,
            prompt=prompt,
            raw_output=raw_output,
            image_width=width,
            image_height=height,
            boxes=parsed.boxes,
            points=parsed.points,
            generation_stats=generation_stats,
            annotated_image_base64=annotated,
        )
