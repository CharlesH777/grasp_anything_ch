from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.train.grasp_contact import (  # noqa: E402
    compute_contact_auxiliary_losses,
    pi_periodic_angle_loss,
    validate_coordinate_token_range,
)

COORD_START = 2
COORD_END = 1002
VOCAB_SIZE = 1004


def _inputs(preferred_values: tuple[int, int, int, int]):
    hidden = torch.zeros((1, 5, 4), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        hidden[0, :4] = torch.eye(4)
    lm_head = torch.zeros((VOCAB_SIZE, 4), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for position, value in enumerate(preferred_values):
            lm_head[COORD_START + value, position] = 12.0
    return hidden, lm_head


def _loss(
    preferred_values: tuple[int, int, int, int],
    candidates: list[list[int]],
    *,
    collision_scores: list[float] | None = None,
    outside_scores: list[float] | None = None,
    collision_valid: bool = False,
):
    hidden, lm_head = _inputs(preferred_values)
    candidate_tensor = torch.tensor([candidates], dtype=torch.long)
    candidate_mask = torch.ones((1, len(candidates)), dtype=torch.bool)
    result = compute_contact_auxiliary_losses(
        hidden_states=hidden,
        lm_head_weight=lm_head,
        contact_mtp_coord_mask=torch.tensor(
            [[False, True, True, True, True]], dtype=torch.bool
        ),
        contact_candidates=candidate_tensor,
        contact_candidate_mask=candidate_mask,
        contact_positive_mask=torch.tensor([True]),
        contact_image_size=torch.tensor([[200.0, 100.0]]),
        coord_start_token_id=COORD_START,
        coord_end_token_id=COORD_END,
        candidate_collision_2d=(
            torch.tensor([collision_scores], dtype=torch.float32)
            if collision_scores is not None
            else None
        ),
        candidate_outside_2d=(
            torch.tensor([outside_scores], dtype=torch.float32)
            if outside_scores is not None
            else None
        ),
        collision_valid=torch.tensor([collision_valid]),
        coord_mass_threshold=0.0,
        coord_entropy_threshold=1.0,
    )
    return result, hidden, lm_head


def test_exchange_invariant_pair_loss_accepts_swapped_contacts() -> None:
    identity, _, _ = _loss((100, 200, 300, 400), [[100, 200, 300, 400]])
    swapped, hidden, lm_head = _loss(
        (300, 400, 100, 200), [[100, 200, 300, 400]]
    )

    assert swapped.pair_sum.item() == pytest.approx(identity.pair_sum.item())
    total = (
        swapped.pair_sum
        + swapped.center_sum
        + swapped.angle_sum
        + swapped.width_sum
    )
    total.backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(lm_head.grad).all()


def test_pi_periodic_angle_loss_matches_smooth_squared_cosine_formula() -> None:
    cosine = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True
    )

    loss = pi_periodic_angle_loss(cosine)

    assert loss.tolist() == pytest.approx([0.0, 0.75, 1.0, 0.75, 0.0])
    loss.sum().backward()
    assert torch.isfinite(cosine.grad).all()


def test_reliable_collision_filter_uses_safe_candidate_when_available() -> None:
    unfiltered, _, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
        collision_scores=[0.5, 0.0],
        collision_valid=False,
    )
    filtered, _, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
        collision_scores=[0.5, 0.0],
        collision_valid=True,
    )

    assert filtered.pair_sum > unfiltered.pair_sum + 5.0


def test_collision_filter_keeps_candidates_if_every_candidate_collides() -> None:
    baseline, _, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
        collision_scores=[0.5, 0.7],
        collision_valid=False,
    )
    all_colliding, _, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
        collision_scores=[0.5, 0.7],
        collision_valid=True,
    )

    assert all_colliding.pair_sum.item() == pytest.approx(baseline.pair_sum.item())


def test_outside_filter_does_not_require_obstacle_mask_validity() -> None:
    unfiltered, _, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
    )
    filtered, _, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
        outside_scores=[0.5, 0.0],
        collision_valid=False,
    )

    assert filtered.pair_sum > unfiltered.pair_sum + 5.0


def test_positive_without_exact_four_slot_block_is_rejected() -> None:
    hidden, lm_head = _inputs((100, 200, 300, 400))
    with pytest.raises(ValueError, match="multiple of four"):
        compute_contact_auxiliary_losses(
            hidden_states=hidden,
            lm_head_weight=lm_head,
            contact_mtp_coord_mask=torch.tensor(
                [[False, True, True, True, False]], dtype=torch.bool
            ),
            contact_candidates=torch.tensor([[[100, 200, 300, 400]]]),
            contact_candidate_mask=torch.tensor([[True]]),
            contact_positive_mask=torch.tensor([True]),
            contact_image_size=torch.tensor([[200.0, 100.0]]),
            coord_start_token_id=COORD_START,
            coord_end_token_id=COORD_END,
        )


def test_coordinate_token_range_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="1001 contiguous"):
        validate_coordinate_token_range(10, 1009)
