from __future__ import annotations

import sys
from pathlib import Path

import torch

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.utils.locany.generate_utils import (  # noqa: E402
    decode_contact_pair,
    handle_pattern,
)

TOKEN_IDS = {
    "box_start_token_id": 1,
    "box_end_token_id": 2,
    "null_token_id": 3,
    "none_token_id": 4,
    "im_end_token_id": 5,
    "ref_end_token_id": 6,
    "coord_start_token_id": 10,
    "coord_end_token_id": 1010,
}


def _probabilities() -> torch.Tensor:
    probs = torch.zeros((6, 1020), dtype=torch.float32)
    probs[0, TOKEN_IDS["box_start_token_id"]] = 0.9
    probs[5, TOKEN_IDS["box_end_token_id"]] = 0.9
    return probs


def test_contact_decoder_jointly_avoids_degenerate_top1_pair() -> None:
    probs = _probabilities()
    preferred = [500, 500, 500, 500]
    alternatives = [200, 400, 800, 600]
    for position, (first, second) in enumerate(
        zip(preferred, alternatives, strict=True), start=1
    ):
        probs[position, 10 + first] = 0.40
        probs[position, 10 + second] = 0.35

    decoded = decode_contact_pair(
        torch.log(probs.clamp_min(1e-30)),
        probs,
        TOKEN_IDS,
        keep_k=2,
        image_size=(1000, 100),
        minimum_width_diagonal=0.05,
    )

    assert decoded is not None
    values = decoded[1:5] - TOKEN_IDS["coord_start_token_id"]
    dx = float(values[2] - values[0]) * 1000 / 1000
    dy = float(values[3] - values[1]) * 100 / 1000
    assert (dx * dx + dy * dy) ** 0.5 >= 0.05 * (1000**2 + 100**2) ** 0.5


def test_contact_decoder_preserves_none_frame() -> None:
    probs = _probabilities()
    probs[1, TOKEN_IDS["none_token_id"]] = 0.8
    probs[2, TOKEN_IDS["box_end_token_id"]] = 0.8
    probs[3, TOKEN_IDS["null_token_id"]] = 0.8
    probs[4, TOKEN_IDS["null_token_id"]] = 0.8

    decoded = decode_contact_pair(
        torch.zeros_like(probs), probs, TOKEN_IDS, image_size=(640, 480)
    )

    assert decoded is not None
    assert decoded[:3].tolist() == [1, 4, 2]


def test_contact_decoder_rejects_non_box_frame() -> None:
    probs = _probabilities()
    probs[0].zero_()
    probs[0, 7] = 0.9
    for position, value in enumerate((100, 200, 800, 200), start=1):
        probs[position, 10 + value] = 0.8

    decoded = decode_contact_pair(
        torch.zeros_like(probs), probs, TOKEN_IDS, image_size=(640, 480)
    )

    assert decoded is None


def test_contact_decoder_rejects_zero_coordinate_mass() -> None:
    probs = _probabilities()

    decoded = decode_contact_pair(
        torch.zeros_like(probs), probs, TOKEN_IDS, image_size=(640, 480)
    )

    assert decoded is None


def test_contact_pattern_is_distinct_from_bbox_pattern() -> None:
    tokens = torch.tensor([1, 110, 210, 310, 410, 2])

    pattern = handle_pattern(
        tokens, TOKEN_IDS, generation_mode="hybrid", geometry_type="contact"
    )

    assert pattern["type"] == "contact_box"
    assert pattern["tokens"] == tokens.tolist()


def test_contact_pattern_rejects_legacy_point_block() -> None:
    point_tokens = torch.tensor([1, 110, 210, 2, 3, 3])

    hybrid = handle_pattern(
        point_tokens,
        TOKEN_IDS,
        generation_mode="hybrid",
        geometry_type="contact",
    )
    fast = handle_pattern(
        point_tokens,
        TOKEN_IDS,
        generation_mode="fast",
        geometry_type="contact",
    )

    assert hybrid["type"] == "error_box"
    assert hybrid["tokens"] == [1, 110, 210]
    assert fast["type"] == "empty_box"
    assert fast["tokens"] == [1, 4, 2]
