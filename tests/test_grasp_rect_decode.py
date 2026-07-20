from __future__ import annotations

import sys
from pathlib import Path

import torch

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.utils.locany.generate_utils import (  # noqa: E402
    constrain_grasp_rect_ar_token,
    decode_grasp_rectangle,
    handle_pattern,
    sample_tokens,
    structured_decode_failure_pattern,
)

TOKEN_IDS = {
    "box_start_token_id": 1,
    "box_end_token_id": 2,
    "grasp_rect_start_token_id": 7,
    "grasp_rect_end_token_id": 8,
    "null_token_id": 3,
    "none_token_id": 4,
    "im_end_token_id": 5,
    "ref_end_token_id": 6,
    "coord_start_token_id": 10,
    "coord_end_token_id": 1010,
}


def _probabilities() -> torch.Tensor:
    probs = torch.zeros((6, 1020), dtype=torch.float32)
    probs[0, TOKEN_IDS["grasp_rect_start_token_id"]] = 0.9
    probs[5, TOKEN_IDS["grasp_rect_end_token_id"]] = 0.9
    return probs


def test_joint_decoder_replaces_degenerate_top1_width() -> None:
    probs = _probabilities()
    for position, value in enumerate((500, 500, 0), start=1):
        probs[position, 10 + value] = 0.8
        probs[position, 10 + ((value + 1) % 1001)] = 0.1
    probs[4, 10] = 0.4
    probs[4, 110] = 0.35

    decoded = decode_grasp_rectangle(
        torch.log(probs.clamp_min(1e-30)),
        probs,
        TOKEN_IDS,
        keep_k=2,
        image_size=(100, 100),
        minimum_width_diagonal=0.05,
    )

    assert decoded is not None
    assert (decoded[1:5] - 10).tolist() == [500, 500, 0, 100]


def test_width_must_strictly_exceed_minimum() -> None:
    probs = _probabilities()
    for position, value in enumerate((500, 500, 0, 100), start=1):
        probs[position, 10 + value] = 0.8

    decoded = decode_grasp_rectangle(
        torch.zeros_like(probs),
        probs,
        TOKEN_IDS,
        keep_k=1,
        image_size=(100, 100),
        minimum_width_diagonal=0.1,
    )

    assert decoded is None


def test_decoder_does_not_add_a_physical_maximum_width() -> None:
    probs = _probabilities()
    for position, value in enumerate((500, 500, 0, 1000), start=1):
        probs[position, 10 + value] = 0.8

    decoded = decode_grasp_rectangle(
        torch.zeros_like(probs),
        probs,
        TOKEN_IDS,
        keep_k=1,
        image_size=(100, 100),
        minimum_width_diagonal=1e-4,
    )

    assert decoded is not None
    assert decoded[4].item() == 1010


def test_decoder_preserves_none_frame() -> None:
    probs = _probabilities()
    probs[1, TOKEN_IDS["none_token_id"]] = 0.8
    probs[2, TOKEN_IDS["grasp_rect_end_token_id"]] = 0.8
    probs[3, TOKEN_IDS["null_token_id"]] = 0.8
    probs[4, TOKEN_IDS["null_token_id"]] = 0.8

    decoded = decode_grasp_rectangle(
        torch.zeros_like(probs), probs, TOKEN_IDS, image_size=(100, 100)
    )

    assert decoded is not None
    assert decoded[:3].tolist() == [7, 4, 8]


def test_grasp_rect_pattern_is_distinct() -> None:
    tokens = torch.tensor([7, 510, 510, 10, 410, 8])

    pattern = handle_pattern(
        tokens,
        TOKEN_IDS,
        generation_mode="hybrid",
        geometry_type="grasp_rect",
    )

    assert pattern["type"] == "grasp_rect_box"
    assert pattern["tokens"] == tokens.tolist()


def test_grasp_rect_ar_constraint_uses_independent_frame() -> None:
    logits = torch.zeros((1, 1, 1020), dtype=torch.float32)
    logits[0, 0, 133] = 1.0

    out_type, token = constrain_grasp_rect_ar_token(
        logits,
        torch.tensor([[TOKEN_IDS["ref_end_token_id"]]]),
        TOKEN_IDS,
    )
    assert (out_type, token.item()) == ("continue_ar", 7)

    sequence = torch.tensor([[7, 510, 510, 10, 410]])
    out_type, token = constrain_grasp_rect_ar_token(logits, sequence, TOKEN_IDS)
    assert (out_type, token.item()) == ("box_end_ar", 8)


def test_joint_decode_failure_cannot_fall_back_to_raw_rect_argmax() -> None:
    logits = torch.full((1, 6, 1020), -20.0)
    logits[0, 0, TOKEN_IDS["grasp_rect_start_token_id"]] = 20.0
    logits[0, 5, TOKEN_IDS["grasp_rect_end_token_id"]] = 20.0
    for position, value in enumerate((500, 500, 0, 0), start=1):
        logits[0, position, TOKEN_IDS["coord_start_token_id"] + value] = 20.0

    _, _, raw_argmax, _, decode_failed = sample_tokens(
        logits,
        torch.tensor([[TOKEN_IDS["ref_end_token_id"]]]),
        TOKEN_IDS,
        geometry_type="grasp_rect",
        grasp_rect_keep_k=1,
        grasp_rect_minimum_width_diagonal=0.1,
        grasp_rect_coord_mass_threshold=0.0,
        generation_mode="hybrid",
    )

    assert decode_failed.tolist() == [True]
    assert raw_argmax[0, 4].item() == TOKEN_IDS["coord_start_token_id"]
    hybrid = structured_decode_failure_pattern(
        TOKEN_IDS, "hybrid", "grasp_rect"
    )
    fast = structured_decode_failure_pattern(TOKEN_IDS, "fast", "grasp_rect")
    assert hybrid["type"] == "error_box"
    assert hybrid["tokens"] == [TOKEN_IDS["grasp_rect_start_token_id"]]
    assert fast["type"] == "grasp_rect_decode_error"
