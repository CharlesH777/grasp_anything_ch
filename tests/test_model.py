import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from locate_anything_service.collision_2d import CollisionMasks
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


class FakeGraspWorker(FakeWorker):
    def generate(self, **kwargs):
        self.generation_mode = kwargs["generation_mode"]
        assert kwargs["do_sample"] is False
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] is None
        assert kwargs["geometry_type"] == "contact"
        assert kwargs["n_future_tokens"] == 6
        assert kwargs["image_size"] == (100, 100)
        assert kwargs["contact_coord_mass_threshold"] > 0
        return "<ref>grasp</ref><grasp><100><500><900><500></grasp>"


class _RecordingLock:
    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, *_args) -> None:
        return None


class _LoadedWorker:
    def __init__(self, grasp_task_token_ids=None) -> None:
        self.config = types.SimpleNamespace(
            grasp_task_token_ids=grasp_task_token_ids
        )

    def to(self, _device):
        return self

    def eval(self):
        return self


def _fake_transformers(
    *, fail_model: bool = False, grasp_task_token_ids=None
) -> types.ModuleType:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            if fail_model:
                raise RuntimeError("model load failed")
            return _LoadedWorker(grasp_task_token_ids)

    module.AutoTokenizer = AutoTokenizer
    module.AutoProcessor = AutoProcessor
    module.AutoModel = AutoModel
    return module


def test_load_is_locked_and_commits_complete_state(monkeypatch) -> None:
    runtime = LocateAnythingRuntime(Settings(device="cpu"))
    lock = _RecordingLock()
    runtime._lock = lock
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())

    runtime.load()

    assert lock.enter_count == 1
    assert runtime.loaded
    assert runtime.load_error is None


def test_failed_load_does_not_publish_partial_state(monkeypatch) -> None:
    runtime = LocateAnythingRuntime(Settings(device="cpu"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers(fail_model=True),
    )

    with pytest.raises(RuntimeError, match="model load failed"):
        runtime.load()

    assert runtime.loaded is False
    assert runtime._tokenizer is None
    assert runtime._processor is None
    assert runtime._worker is None
    assert runtime.load_error == "model load failed"


def test_required_grasp_checkpoint_rejects_base_model(monkeypatch) -> None:
    runtime = LocateAnythingRuntime(
        Settings(device="cpu", require_grasp_checkpoint=True)
    )
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())

    with pytest.raises(RuntimeError, match="grasp_task_token_ids"):
        runtime.load()

    assert runtime.loaded is False


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


def test_runtime_returns_grasp_and_collision_status(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    obstacle = Image.new("L", (100, 100), 0)
    for x in range(45, 56):
        for y in range(45, 56):
            obstacle.putpixel((x, y), 255)

    runtime = LocateAnythingRuntime(
        Settings(device="cpu", generation_mode="fast"),
        collision_mask_provider=lambda _image, _query: CollisionMasks(
            obstacle_mask=obstacle, valid=True
        ),
    )
    runtime._worker = FakeGraspWorker()
    runtime._processor = FakeProcessor()
    runtime._tokenizer = object()

    result = runtime.predict(image_path, query="object", mode="grasp_contact")

    assert result.grasp_status == "ok"
    assert len(result.grasps) == 1
    assert result.grasps[0].contacts_pixels == (10, 50, 90, 50)
    assert result.grasps[0].collision_2d_status == "collision"
    assert result.boxes == []
    assert result.points == []
