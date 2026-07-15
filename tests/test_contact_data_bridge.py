from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.train.locany_finetune_magi_stream import (  # noqa: E402
    IGNORE_TOKEN_ID,
    LazySupervisedDatasetMTP,
    StreamPackingMTPTrainer,
    export_locany_inference_files,
    packed_collate_fn_mtp,
)


class _Tokenizer:
    ids = {"<box>": 10, "</box>": 11, "<0>": 100, "<1000>": 1100}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]


def _dataset() -> LazySupervisedDatasetMTP:
    dataset = LazySupervisedDatasetMTP.__new__(LazySupervisedDatasetMTP)
    dataset.processor = SimpleNamespace(tokenizer=_Tokenizer())
    dataset.block_size = 6
    dataset.task_type = "grasp_contact"
    dataset.max_contact_candidates = 3
    return dataset


def test_contact_mtp_mask_marks_only_the_four_coordinate_labels() -> None:
    dataset = _dataset()
    labels = torch.tensor(
        [
            IGNORE_TOKEN_ID,
            IGNORE_TOKEN_ID,
            1,
            2,
            3,
            IGNORE_TOKEN_ID,
            10,
            200,
            300,
            400,
            500,
            11,
        ]
    )

    mask = dataset._contact_mtp_coordinate_mask(
        labels, original_length=5, task_type="grasp_contact"
    )

    assert mask.nonzero(as_tuple=False).flatten().tolist() == [7, 8, 9, 10]


def test_contact_mtp_mask_rejects_missing_joint_block() -> None:
    dataset = _dataset()
    labels = torch.full((12,), IGNORE_TOKEN_ID)

    with pytest.raises(ValueError, match="exactly one"):
        dataset._contact_mtp_coordinate_mask(
            labels, original_length=5, task_type="grasp_contact"
        )


def test_contact_fields_are_fixed_shape_and_keep_collision_unknown() -> None:
    dataset = _dataset()
    fields = dataset._contact_fields(
        {
            "task_type": "grasp_contact",
            "image_width": 640,
            "image_height": 480,
            "contact_candidates": [[100, 200, 300, 400], [200, 300, 400, 500]],
            "candidate_collision_2d": [0.0, 0.2],
            "candidate_outside_2d": [0.0, 0.1],
            "collision_valid": False,
            "conversations": [
                {
                    "from": "gpt",
                    "value": "<box><100><200><300><400></box>",
                }
            ],
        }
    )

    assert fields["contact_candidates"].shape == (1, 3, 4)
    assert fields["contact_candidate_mask"].tolist() == [[True, True, False]]
    assert fields["contact_positive_mask"].tolist() == [True]
    assert fields["collision_valid"].tolist() == [False]
    assert torch.isnan(fields["candidate_collision_2d"][0, 2])
    assert fields["candidate_outside_2d"][0, :2].tolist() == pytest.approx(
        [0.0, 0.1]
    )


def test_packed_collator_preserves_contact_rows_and_label_alignment() -> None:
    length = 12
    contact_mask = torch.zeros(length, dtype=torch.bool)
    contact_mask[7:11] = True
    feature = {
        "input_ids": torch.arange(length),
        "labels": torch.tensor(
            [IGNORE_TOKEN_ID, *range(1, length)], dtype=torch.long
        ),
        "position_ids": torch.arange(length),
        "pixel_values": torch.zeros((4, 3, 14, 14)),
        "image_flags": torch.tensor([1]),
        "image_grid_hws": np.array([[2, 2]]),
        "sub_sample_lengths": torch.tensor([length]),
        "contact_mtp_coord_mask": contact_mask,
        "contact_candidates": torch.tensor([[[100, 200, 300, 400]]]),
        "contact_candidate_mask": torch.tensor([[True]]),
        "contact_positive_mask": torch.tensor([True]),
        "contact_image_size": torch.tensor([[640.0, 480.0]]),
        "candidate_collision_2d": torch.tensor([[0.0]]),
        "candidate_outside_2d": torch.tensor([[0.0]]),
        "collision_valid": torch.tensor([True]),
        "contact_task_code": torch.tensor([1]),
    }

    batch = packed_collate_fn_mtp([feature])

    assert batch["contact_mtp_coord_mask"].shape == (1, length)
    assert batch["contact_candidates"].shape == (1, 1, 4)
    assert batch["contact_task_code"].tolist() == [1]


def test_shifted_label_count_excludes_each_packed_sample_first_label() -> None:
    labels = torch.tensor([[99, 1, 2, IGNORE_TOKEN_ID, 88, 3, 4, 5]])
    batch = {
        "labels": labels,
        "sub_sample_lengths": [torch.tensor([4, 4])],
    }

    count = StreamPackingMTPTrainer._shifted_label_count(batch)

    assert count.item() == 5


class _Accelerator:
    @staticmethod
    def unwrap_model(model):
        return model


def test_accumulation_window_counts_are_shared_with_every_microbatch() -> None:
    trainer = StreamPackingMTPTrainer.__new__(StreamPackingMTPTrainer)
    trainer.args = SimpleNamespace(
        average_tokens_across_devices=True, world_size=1
    )
    trainer.accelerator = _Accelerator()
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(
            contact_geometry_start_blocks=1,
            contact_geometry_ramp_blocks=2,
        )
    )
    trainer._seen_contact_blocks = 2
    first = {
        "labels": torch.tensor([[IGNORE_TOKEN_ID, 1, 2, 3]]),
        "sub_sample_lengths": [torch.tensor([4])],
        "contact_positive_mask": torch.tensor([True]),
        "collision_valid": torch.tensor([False]),
        "contact_task_code": torch.tensor([1]),
    }
    second = {
        "labels": torch.tensor([[IGNORE_TOKEN_ID, 4, 5]]),
        "sub_sample_lengths": [torch.tensor([3])],
        "contact_positive_mask": torch.tensor([False]),
        "collision_valid": torch.tensor([False]),
        "contact_task_code": torch.tensor([0]),
    }

    batches, ce_count = trainer.get_batch_samples(
        iter((first, second)), num_batches=2, device=torch.device("cpu")
    )

    assert ce_count.item() == 5
    assert trainer._seen_contact_blocks == 3
    assert trainer._last_window_counts.tolist() == [5, 1, 0, 0, 1, 2]
    assert trainer._last_geometry_scale == 0.5
    for batch in batches:
        assert batch["global_ce_tokens_in_window"].item() == 5
        assert batch["global_contact_count_in_window"].item() == 1
        assert batch["geometry_loss_scale"].item() == pytest.approx(0.5)


def test_checkpoint_export_installs_remote_code_and_auto_map(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")

    export_locany_inference_files(str(checkpoint))

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    assert (checkpoint / "modeling_locateanything.py").is_file()
    assert (checkpoint / "generate_utils.py").is_file()
    assert config["auto_map"]["AutoModel"].startswith("modeling_locateanything")
