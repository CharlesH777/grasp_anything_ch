from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.train.grasp_contact import (  # noqa: E402
    combine_base_and_pair_ce,
    compute_contact_auxiliary_losses,
    compute_coordinate_token_metrics,
    compute_task_adapter_cross_entropy,
    configure_llm_lora,
    enable_task_token_training,
    initialize_missing_task_adapters,
    pi_periodic_angle_loss,
    validate_coordinate_token_range,
)
from eaglevl.utils.locany.grasp_adapter_utils import (  # noqa: E402
    apply_grasp_task_output_delta,
)

COORD_START = 2
COORD_END = 1002
VOCAB_SIZE = 1004


def test_pair_ce_replaces_four_tokens_without_changing_loss_scale() -> None:
    base_sum = torch.tensor(25.0)
    pair_block_mean_sum = torch.tensor(2.0)

    combined = combine_base_and_pair_ce(
        base_sum,
        pair_block_mean_sum,
        global_ce_token_count=29,
        pair_weight=1.0,
    )

    assert combined.item() == pytest.approx((25.0 + 4.0 * 2.0) / 29.0)


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


def test_small_predicted_width_keeps_width_recovery_loss_active() -> None:
    result, _, _ = _loss(
        (200, 200, 200, 200),
        [[100, 200, 300, 400]],
    )

    assert result.geometry_active_count.item() == 1
    assert result.width_sum.item() > 0
    assert result.angle_sum.item() == 0


def test_pi_periodic_angle_loss_is_exchange_invariant_with_90_degree_gradient() -> None:
    cosine = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True
    )

    loss = pi_periodic_angle_loss(cosine)

    assert loss.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])
    loss.sum().backward()
    assert torch.isfinite(cosine.grad).all()
    assert cosine.grad[2].abs().item() == pytest.approx(1.0)

    predicted_vector = torch.tensor([1.0, 0.0], requires_grad=True)
    target_vector = torch.tensor([0.0, 1.0])
    perpendicular_cosine = torch.nn.functional.cosine_similarity(
        predicted_vector.unsqueeze(0), target_vector.unsqueeze(0)
    )
    pi_periodic_angle_loss(perpendicular_cosine).backward()
    assert predicted_vector.grad.norm().item() > 0.0


def test_geometry_losses_keep_pixel_scale_before_normalization() -> None:
    result, _, _ = _loss(
        (100, 200, 300, 400),
        [[200, 200, 500, 400]],
    )

    # A 30 px center error and a substantial width error must not collapse to
    # the 1e-6 scale produced by normalized-input SmoothL1.
    assert result.center_sum.item() > 1e-2
    assert result.width_sum.item() > 1e-2


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


def test_collision_filter_skips_positive_with_no_safe_candidate() -> None:
    skipped, hidden, _ = _loss(
        (100, 200, 300, 400),
        [[100, 200, 300, 400], [600, 200, 800, 400]],
        collision_scores=[0.5, 0.7],
        collision_valid=True,
    )

    assert skipped.contact_count.item() == 0
    assert skipped.pair_sum.item() == 0
    skipped.pair_sum.backward()
    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))


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


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.base = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.lora = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.input_grads_enabled = False

    def named_parameters(self):
        return iter(
            (
                ("base.weight", self.base),
                ("layer.lora_A.default.weight", self.lora),
            )
        )

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True


class _FakeLoraModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"use_llm_lora": 0})()
        self.use_llm_lora = False
        self.language_model = _FakeLanguageModel()
        self.wrap_calls = 0

    def wrap_llm_lora(self, *, r, lora_alpha, lora_dropout) -> None:
        assert r == 32
        assert lora_alpha == 64
        assert lora_dropout == 0.05
        self.wrap_calls += 1
        self.use_llm_lora = True
        self.language_model.lora.requires_grad = True
        self.language_model.enable_input_require_grads()


def test_lora_configuration_reuses_checkpoint_adapter_after_freezing() -> None:
    model = _FakeLoraModel()

    assert configure_llm_lora(model, 32) is True
    assert model.wrap_calls == 1
    assert model.config.use_llm_lora == 32

    model.language_model.lora.requires_grad = False
    assert configure_llm_lora(model, 32) is False
    assert model.wrap_calls == 1
    assert model.language_model.lora.requires_grad is True


def test_lora_configuration_rejects_checkpoint_rank_change() -> None:
    model = _FakeLoraModel()
    configure_llm_lora(model, 32)

    with pytest.raises(ValueError, match="checkpoint rank"):
        configure_llm_lora(model, 16)


class _TaskTokenLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.lm_head = torch.nn.Linear(3, 8, bias=False)
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


class _TaskTokenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TaskTokenLanguageModel()
        self.language_model.embedding.weight.requires_grad = False
        self.config = type("Config", (), {"grasp_task_token_ids": [6, 7]})()
        self.register_buffer("_grasp_task_token_ids", torch.tensor([6, 7]))
        self.grasp_task_embedding_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.grasp_task_output_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self._grasp_task_embedding_hook = None
        self._grasp_task_output_hook = None

    def register_grasp_task_embedding_hook(self):
        if self._grasp_task_embedding_hook is not None:
            self._grasp_task_embedding_hook.remove()
        if self._grasp_task_output_hook is not None:
            self._grasp_task_output_hook.remove()

        def add_delta(_module, inputs, output):
            result = output
            for row, token_id in enumerate(self._grasp_task_token_ids):
                token_mask = (inputs[0] == token_id).unsqueeze(-1)
                result = torch.where(
                    token_mask,
                    result + self.grasp_task_embedding_delta[row],
                    result,
                )
            return result

        self._grasp_task_embedding_hook = (
            self.language_model.embedding.register_forward_hook(add_delta)
        )

        def add_logits(_module, inputs, output):
            return apply_grasp_task_output_delta(
                inputs[0],
                output,
                self._grasp_task_token_ids,
                self.grasp_task_output_delta,
            )

        self._grasp_task_output_hook = (
            self.language_model.lm_head.register_forward_hook(add_logits)
        )
        return (
            self.grasp_task_embedding_delta.numel()
            + self.grasp_task_output_delta.numel()
        )


def test_task_token_training_uses_compact_delta_and_keeps_base_frozen() -> None:
    model = _TaskTokenModel()

    assert enable_task_token_training(model, [6, 7]) == 12
    assert enable_task_token_training(model, [6, 7]) == 12
    input_loss = model.language_model.embedding(torch.tensor([[0, 6, 7]])).sum()
    output_loss = model.language_model.lm_head(torch.ones(1, 3)).sum()
    (input_loss + output_loss).backward()

    assert model.language_model.embedding.weight.grad is None
    assert torch.equal(
        model.grasp_task_embedding_delta.grad,
        torch.ones_like(model.grasp_task_embedding_delta),
    )
    assert torch.equal(
        model.grasp_task_output_delta.grad,
        torch.ones_like(model.grasp_task_output_delta),
    )


def test_training_ce_uses_inference_adapter_at_every_valid_position() -> None:
    model = _TaskTokenModel()
    enable_task_token_training(model, [6, 7])
    hidden = torch.tensor(
        [[1.0, 0.5, -0.25], [0.25, 1.0, 0.5]], requires_grad=True
    )
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        model.grasp_task_output_delta.copy_(
            torch.tensor([[0.4, -0.2, 0.1], [-0.1, 0.3, 0.2]])
        )

    base_logits = torch.nn.functional.linear(
        hidden, model.language_model.lm_head.weight
    )
    expected_logits = apply_grasp_task_output_delta(
        hidden,
        base_logits,
        model._grasp_task_token_ids,
        model.grasp_task_output_delta,
    )
    inference_logits = model.language_model.lm_head(hidden)
    assert torch.allclose(inference_logits, expected_logits)

    training_loss = compute_task_adapter_cross_entropy(
        model.language_model.lm_head, hidden, labels
    )
    expected_loss = torch.nn.functional.cross_entropy(
        inference_logits, labels, reduction="sum"
    )
    assert torch.allclose(training_loss, expected_loss)
    training_loss.backward()

    # Neither target is a grasp structure token. Both adapter rows must still
    # receive the negative-class gradient induced by the CE denominator.
    row_gradient_norms = model.grasp_task_output_delta.grad.abs().sum(dim=-1)
    assert torch.all(row_gradient_norms > 0)
    assert model.language_model.lm_head.weight.grad is None


def test_coordinate_metrics_use_the_same_hooked_logits() -> None:
    model = _TaskTokenModel()
    enable_task_token_training(model, [6, 7])
    hidden = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
    labels = torch.tensor([2, 3])
    with torch.no_grad():
        model.grasp_task_output_delta.copy_(
            torch.tensor([[0.5, 0.0, 0.0], [0.0, -0.5, 0.0]])
        )
        logits = model.language_model.lm_head(hidden).float()
        expected_ce = torch.nn.functional.cross_entropy(
            logits, labels, reduction="sum"
        )
        expected_correct = logits.argmax(dim=-1).eq(labels).sum()

    ce_sum, correct, count = compute_coordinate_token_metrics(
        model.language_model.lm_head, hidden, labels
    )

    assert torch.allclose(ce_sum, expected_ce)
    assert torch.equal(correct, expected_correct)
    assert count.item() == 2
    assert not ce_sum.requires_grad


def test_pair_loss_uses_hooked_task_token_logits_as_negative_classes() -> None:
    lm_head = torch.nn.Linear(3, 1005, bias=False)
    lm_head.weight.requires_grad = False
    task_token_ids = torch.tensor([1003, 1004])
    output_delta = torch.nn.Parameter(torch.zeros(2, 3))

    def add_logits(_module, inputs, output):
        return apply_grasp_task_output_delta(
            inputs[0], output, task_token_ids, output_delta
        )

    handle = lm_head.register_forward_hook(add_logits)
    hidden = torch.zeros((1, 5, 3), requires_grad=True)
    with torch.no_grad():
        hidden[0, 1:] = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
    result = compute_contact_auxiliary_losses(
        hidden_states=hidden,
        lm_head_weight=lm_head.weight,
        lm_head=lm_head,
        contact_mtp_coord_mask=torch.tensor(
            [[False, True, True, True, True]], dtype=torch.bool
        ),
        contact_candidates=torch.tensor([[[10, 20, 30, 40]]]),
        contact_candidate_mask=torch.tensor([[True]]),
        contact_positive_mask=torch.tensor([True]),
        contact_image_size=torch.tensor([[200.0, 100.0]]),
        coord_start_token_id=2,
        coord_end_token_id=1002,
        coord_mass_threshold=0.0,
        coord_entropy_threshold=1.0,
    )
    result.pair_sum.backward()
    handle.remove()

    # Pair targets are all coordinate tokens. A nonzero gradient on both grasp
    # rows proves their hooked logits participated as negative classes.
    assert torch.all(output_delta.grad.abs().sum(dim=-1) > 0)


def test_missing_task_adapters_are_zero_initialized_after_model_load() -> None:
    model = _TaskTokenModel()
    with torch.no_grad():
        model.grasp_task_embedding_delta.fill_(float("nan"))
        model.grasp_task_output_delta.fill_(float("nan"))

    initialized = initialize_missing_task_adapters(
        model,
        ["grasp_task_embedding_delta", "grasp_task_output_delta"],
    )

    assert initialized == 12
    assert torch.equal(
        model.grasp_task_embedding_delta,
        torch.zeros_like(model.grasp_task_embedding_delta),
    )
    assert torch.equal(
        model.grasp_task_output_delta,
        torch.zeros_like(model.grasp_task_output_delta),
    )


def test_loaded_task_adapters_are_preserved_and_validated() -> None:
    model = _TaskTokenModel()
    with torch.no_grad():
        model.grasp_task_embedding_delta.fill_(0.25)
        model.grasp_task_output_delta.fill_(-0.5)

    assert initialize_missing_task_adapters(model, []) == 0
    assert torch.all(model.grasp_task_embedding_delta == 0.25)
    assert torch.all(model.grasp_task_output_delta == -0.5)

    with torch.no_grad():
        model.grasp_task_output_delta[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="grasp_task_output_delta"):
        initialize_missing_task_adapters(model, [])


def test_task_embedding_hook_does_not_spread_nonfinite_unused_rows() -> None:
    model = _TaskTokenModel()
    enable_task_token_training(model, [6, 7])
    with torch.no_grad():
        model.grasp_task_embedding_delta[0].fill_(float("nan"))

    output = model.language_model.embedding(torch.tensor([[0, 1, 7]]))

    assert torch.isfinite(output).all()
