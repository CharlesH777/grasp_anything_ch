from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acceptance = _load(
    "record_phase_acceptance",
    "training/scripts/record_phase_acceptance.py",
)
sampler = _load("sample_contact_eval", "training/scripts/sample_contact_eval.py")


def test_phase_acceptance_uses_positive_weighted_metrics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 6000}), encoding="utf-8"
    )
    (checkpoint / "grasp_contact_trainer_state.json").write_text(
        json.dumps({"training_phase": "pair"}), encoding="utf-8"
    )
    metric_paths = {}
    for split, count, gacc in (("seen", 3, 0.6), ("novel", 1, 0.2)):
        path = tmp_path / f"{split}.json"
        path.write_text(
            json.dumps(
                {
                    "positive_samples": count,
                    "format_valid_rate": 1.0,
                    "positive_grasp_output_rate": 1.0,
                    "gacc_corrected_strict": gacc,
                    "miou_strict": 0.4,
                }
            ),
            encoding="utf-8",
        )
        metric_paths[split] = path

    report = acceptance.build_acceptance(
        checkpoint,
        "pair",
        metric_paths,
        min_format_valid_rate=0.98,
        min_positive_output_rate=0.98,
        min_gacc_strict=0.3,
    )

    assert report["accepted"] is True
    assert report["checkpoint_step"] == 6000
    assert report["metrics"]["aggregate"]["gacc_corrected_strict"] == pytest.approx(
        0.5
    )


def test_scene_sampler_round_robins_images() -> None:
    rows = [
        {"sample_id": f"sample-{index}", "image": f"image-{index // 2}.png"}
        for index in range(8)
    ]

    selected = sampler.sample_scene(rows, 4, __import__("random").Random(7))

    assert len(selected) == 4
    assert len({row["image"] for row in selected}) == 4


def test_phase_launchers_do_not_embed_machine_paths() -> None:
    for name in ("start_phase1_overfit.sh", "start_phase2_sft.sh"):
        source = (ROOT / "training" / "scripts" / name).read_text(encoding="utf-8")
        assert "/data2/" not in source
        assert "/home/" not in source
        assert '"${META_PATH:-}"' in source
