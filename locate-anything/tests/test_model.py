from pathlib import Path

import torch
from PIL import Image

from locate_anything_service.config import Settings
from locate_anything_service.model import LocateAnythingRuntime


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    def py_apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def process_vision_info(self, _messages):
        return [], []

    def __call__(self, **_kwargs):
        return FakeBatch(
            pixel_values=torch.zeros((1, 1), dtype=torch.float32),
            input_ids=torch.zeros((1, 1), dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
        )


class FakeWorker:
    def __init__(self) -> None:
        self.generation_mode = None

    def generate(self, **kwargs):
        self.generation_mode = kwargs["generation_mode"]
        return "<ref>object</ref><box><1><1><500><500></box>"


def test_runtime_uses_setting_default_and_structured_stats(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (1000, 1000), "white").save(image_path)

    runtime = LocateAnythingRuntime(Settings(device="cpu", generation_mode="slow"))
    worker = FakeWorker()
    runtime._worker = worker
    runtime._processor = FakeProcessor()
    runtime._tokenizer = object()

    result = runtime.predict(image_path, query="object", generation_mode=None)

    assert worker.generation_mode == "slow"
    assert result.generation_mode == "slow"
    assert result.generation_stats is not None
    assert result.generation_stats["generation_seconds"] >= 0
    assert result.generation_stats["box_count"] == 1
    assert result.boxes[0].normalized == (0.001, 0.001, 0.5, 0.5)
