from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

EAGLE_ROOT = Path(__file__).resolve().parents[1] / "training" / "Eagle" / "Embodied"
sys.path.insert(0, str(EAGLE_ROOT))

from eaglevl.train.grasp_contact import (  # noqa: E402
    activate_task_token_adapter,
    register_all_task_token_hooks,
)
from eaglevl.train.grasp_rect import (  # noqa: E402
    circular_angle_neighbors,
    compute_grasp_rect_auxiliary_losses,
    double_angle_loss,
    wrapped_angle_marginal_nll,
)
from eaglevl.utils.locany.grasp_adapter_utils import (  # noqa: E402
    apply_grasp_task_output_delta,
)


def test_wrapped_radius_zero_equals_full_vocabulary_ce() -> None:
    logits = torch.randn(4, 1020, dtype=torch.float64)
    targets = torch.tensor([0, 1, 500, 1000])
    log_probs = logits.log_softmax(dim=-1)

    wrapped = wrapped_angle_marginal_nll(
        log_probs,
        targets,
        coord_start_token_id=10,
        radius=0,
    )
    expected = F.cross_entropy(logits, targets + 10, reduction="none")

    assert torch.allclose(wrapped, expected)


def test_wrapped_neighbors_cross_the_angle_seam() -> None:
    targets = torch.tensor([0, 1000])

    neighbors = circular_angle_neighbors(targets, radius=1)

    assert neighbors.tolist() == [[1000, 0, 1], [999, 1000, 0]]


def test_wrapped_loss_penalizes_probability_on_non_coordinate_tokens() -> None:
    coordinate_logits = torch.full((1, 1020), -20.0)
    coordinate_logits[0, 10] = 5.0
    wrong_token_logits = coordinate_logits.clone()
    wrong_token_logits[0, 0] = 10.0

    clean_loss = wrapped_angle_marginal_nll(
        coordinate_logits.log_softmax(dim=-1),
        torch.tensor([0]),
        coord_start_token_id=10,
        radius=1,
    )
    wrong_loss = wrapped_angle_marginal_nll(
        wrong_token_logits.log_softmax(dim=-1),
        torch.tensor([0]),
        coord_start_token_id=10,
        radius=1,
    )

    assert wrong_loss.item() > clean_loss.item() + 4.0


def test_double_angle_loss_treats_bins_zero_and_1000_as_neighbors() -> None:
    probabilities = torch.zeros((1, 1001))
    probabilities[0, 1000] = 1.0

    loss, resultant = double_angle_loss(probabilities, torch.tensor([0]))

    assert resultant.item() == pytest.approx(1.0)
    assert loss.item() < 2e-5


class _FixedHead(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[:2] == self.fixed_logits.shape[:2]
        return self.fixed_logits


def test_auxiliary_pose_loss_uses_four_slots_and_reports_top1() -> None:
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    logits = torch.full((1, 4, 1020), -12.0)
    target = torch.tensor([100, 200, 0, 300])
    for slot, value in enumerate(target):
        logits[0, slot, 10 + value] = 12.0
    mask = torch.tensor([[False, True, True, True, True, False]])

    output = compute_grasp_rect_auxiliary_losses(
        hidden_states=hidden,
        lm_head_weight=torch.empty(1020, 8),
        lm_head=_FixedHead(logits),
        grasp_rect_mtp_coord_mask=mask,
        grasp_rect_candidates=target.reshape(1, 1, 4),
        grasp_rect_candidate_mask=torch.tensor([[True]]),
        grasp_rect_positive_mask=torch.tensor([True]),
        grasp_rect_image_size=torch.tensor([[640.0, 480.0]]),
        coord_start_token_id=10,
        coord_end_token_id=1010,
        angle_wrap_radius=0,
        coord_mass_threshold=0.0,
        coord_entropy_threshold=1.0,
        angle_resultant_threshold=0.0,
    )

    assert output.grasp_count.item() == 1
    assert output.pose_sum.item() < 1e-6
    assert output.pose_four_slot_top1_correct.item() == 1
    assert output.theta_circular_within1_correct.item() == 1


class _DualTaskLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 3)
        self.lm_head = torch.nn.Linear(3, 10, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def replace_token_modules(self) -> None:
        self.embedding = torch.nn.Embedding(10, 3)
        self.lm_head = torch.nn.Linear(3, 10, bias=False)
        self.embedding.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False


class _DualTaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _DualTaskLanguageModel()
        self.language_model.embedding.weight.requires_grad = False
        self.language_model.lm_head.weight.requires_grad = False
        self.config = type(
            "Config",
            (),
            {
                "grasp_task_token_ids": [6, 7],
                "grasp_rect_task_token_ids": [8, 9],
            },
        )()
        self.register_buffer("_grasp_task_token_ids", torch.tensor([6, 7]))
        self.register_buffer(
            "_grasp_rect_task_token_ids", torch.tensor([8, 9])
        )
        self.grasp_task_embedding_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.grasp_task_output_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.grasp_rect_task_embedding_delta = torch.nn.Parameter(
            torch.zeros(2, 3)
        )
        self.grasp_rect_task_output_delta = torch.nn.Parameter(
            torch.zeros(2, 3)
        )
        self._grasp_task_embedding_hook = None
        self._grasp_task_output_hook = None
        self._grasp_rect_task_embedding_hook = None
        self._grasp_rect_task_output_hook = None

    def _register(self, prefix: str, token_ids: torch.Tensor) -> int:
        input_handle_name = f"_{prefix}_embedding_hook"
        output_handle_name = f"_{prefix}_output_hook"
        for handle_name in (input_handle_name, output_handle_name):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.remove()
        input_delta = getattr(self, f"{prefix}_embedding_delta")
        output_delta = getattr(self, f"{prefix}_output_delta")

        def add_input(_module, inputs, output):
            result = output
            for row, token_id in enumerate(token_ids):
                result = torch.where(
                    (inputs[0] == token_id).unsqueeze(-1),
                    result + input_delta[row],
                    result,
                )
            return result

        def add_output(_module, inputs, output):
            return apply_grasp_task_output_delta(
                inputs[0], output, token_ids, output_delta
            )

        setattr(
            self,
            input_handle_name,
            self.language_model.embedding.register_forward_hook(add_input),
        )
        setattr(
            self,
            output_handle_name,
            self.language_model.lm_head.register_forward_hook(add_output),
        )
        return input_delta.numel() + output_delta.numel()

    def register_grasp_task_embedding_hook(self):
        return self._register("grasp_task", self._grasp_task_token_ids)

    def register_grasp_rect_task_embedding_hook(self):
        return self._register(
            "grasp_rect_task", self._grasp_rect_task_token_ids
        )


def test_resize_rehooks_both_tasks_and_only_rect_adapter_remains_trainable() -> None:
    model = _DualTaskModel()
    register_all_task_token_hooks(model)
    model.language_model.replace_token_modules()
    with torch.no_grad():
        model.grasp_task_embedding_delta[0].fill_(1.0)
        model.grasp_rect_task_embedding_delta[0].fill_(2.0)

    activated = activate_task_token_adapter(model, "grasp_rect", [8, 9])
    embedded = model.language_model.embedding(torch.tensor([[6, 8]]))
    base = model.language_model.embedding.weight.detach()[[6, 8]]

    assert activated == 12
    assert torch.allclose(embedded[0, 0], base[0] + 1.0)
    assert torch.allclose(embedded[0, 1], base[1] + 2.0)
    assert model.grasp_task_embedding_delta.requires_grad is False
    assert model.grasp_task_output_delta.requires_grad is False
    assert model.grasp_rect_task_embedding_delta.requires_grad is True
    assert model.grasp_rect_task_output_delta.requires_grad is True
