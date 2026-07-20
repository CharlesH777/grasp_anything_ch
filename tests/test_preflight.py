from __future__ import annotations

import json
from pathlib import Path

from locate_anything_service.config import Settings
from locate_anything_service.preflight import format_results, run_preflight


def _checkpoint(path: Path, *, grasp: bool, grasp_rect: bool = False) -> None:
    path.mkdir()
    config = {"grasp_task_token_ids": [101, 102]} if grasp else {}
    if grasp_rect:
        config["grasp_rect_task_token_ids"] = [103, 104]
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in (
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    ):
        (path / name).write_text("{}", encoding="utf-8")


def test_preflight_accepts_complete_grasp_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint, grasp=True)

    results = run_preflight(
        Settings(model_id=str(checkpoint), require_grasp_checkpoint=True),
        check_cuda=False,
    )

    assert all(item.ok for item in results)
    assert "[OK] grasp_checkpoint" in format_results(results)


def test_preflight_rejects_base_model_when_grasp_is_required(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint, grasp=False)

    results = run_preflight(
        Settings(model_id=str(checkpoint), require_grasp_checkpoint=True),
        check_cuda=False,
    )

    assert not all(item.ok for item in results)
    assert any(item.name == "grasp_checkpoint" and not item.ok for item in results)


def test_preflight_accepts_complete_grasp_rect_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint, grasp=False, grasp_rect=True)

    results = run_preflight(
        Settings(
            model_id=str(checkpoint),
            require_grasp_rect_checkpoint=True,
        ),
        check_cuda=False,
    )

    assert all(item.ok for item in results)
    assert "[OK] grasp_rect_checkpoint" in format_results(results)
