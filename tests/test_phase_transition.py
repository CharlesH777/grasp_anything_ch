from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "scripts"
    / "validate_phase_transition.py"
)
TRAIN_SCRIPT = SCRIPT.with_name("train_realvlg_contact.sh")
EAGLE_TRAIN_SCRIPT = (
    SCRIPT.parents[1]
    / "Eagle"
    / "Embodied"
    / "eaglevl"
    / "train"
    / "locany_finetune_magi_stream.py"
)
SPEC = importlib.util.spec_from_file_location("validate_phase_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _write_checkpoint(
    path: Path,
    *,
    grasp: bool,
    accepted_phase: str = "pair",
    coordinate_accuracy: float = 0.96,
) -> None:
    path.mkdir()
    config = {"use_llm_lora": 32 if grasp else 0}
    if grasp:
        config["grasp_task_token_ids"] = [100, 101]
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if not grasp:
        return
    shard_name = "model-00001-of-00001.safetensors"
    (path / shard_name).write_bytes(b"weights")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({
            "weight_map": {
                "grasp_task_embedding_delta": shard_name,
                "grasp_task_output_delta": shard_name,
                "language_model.layers.0.lora_A.default.weight": shard_name,
                "language_model.layers.0.lora_B.default.weight": shard_name,
            }
        }),
        encoding="utf-8",
    )
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": 300}), encoding="utf-8"
    )
    (path / "grasp_contact_trainer_state.json").write_text(
        json.dumps({
            "seen_contact_blocks": 100,
            "training_phase": accepted_phase,
            "data_fingerprint": "test-data",
        }),
        encoding="utf-8",
    )
    (path / "phase_acceptance.json").write_text(
        json.dumps({
            "phase": accepted_phase,
            "accepted": True,
            "checkpoint_step": 300,
            "metrics": {
                "format_valid_rate": 1.0,
                "coordinate_top1_accuracy": coordinate_accuracy,
            },
        }),
        encoding="utf-8",
    )


def _write_meta(path: Path, annotation: str) -> None:
    path.write_text(
        json.dumps({"contact": {"annotation": annotation}}), encoding="utf-8"
    )


def test_geometry_phase_rejects_base_model_and_overfit_meta(tmp_path: Path) -> None:
    model = tmp_path / "base"
    _write_checkpoint(model, grasp=False)
    meta = tmp_path / "overfit64_meta.json"
    _write_meta(meta, "/tmp/contact_overfit64.jsonl")

    with pytest.raises(ValueError, match="not a grasp checkpoint"):
        validator.validate_phase_transition("geometry", model, meta)


def test_geometry_phase_requires_full_meta(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    _write_checkpoint(model, grasp=True)
    meta = tmp_path / "overfit64_meta.json"
    _write_meta(meta, "/tmp/contact_overfit64.jsonl")

    with pytest.raises(ValueError, match="cannot use overfit64"):
        validator.validate_phase_transition("geometry", model, meta)


def test_geometry_phase_accepts_grasp_checkpoint_and_full_meta(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    _write_checkpoint(model, grasp=True)
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    validator.validate_phase_transition("geometry", model, meta)


def test_sft_phase_also_requires_the_phase_one_checkpoint(tmp_path: Path) -> None:
    model = tmp_path / "base"
    _write_checkpoint(model, grasp=False)
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    with pytest.raises(ValueError, match="not a grasp checkpoint"):
        validator.validate_phase_transition("sft", model, meta)


def test_sft_phase_requires_overfit_metrics_to_pass(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    _write_checkpoint(
        model,
        grasp=True,
        accepted_phase="overfit",
        coordinate_accuracy=0.30,
    )
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    with pytest.raises(ValueError, match="coordinate_top1_accuracy"):
        validator.validate_phase_transition("sft", model, meta)


def test_sft_phase_accepts_real_weights_and_phase_one_metrics(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    _write_checkpoint(model, grasp=True, accepted_phase="overfit")
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    validator.validate_phase_transition("sft", model, meta)


def test_legacy_checkpoint_can_transition_after_explicit_acceptance(
    tmp_path: Path,
) -> None:
    model = tmp_path / "checkpoint"
    _write_checkpoint(model, grasp=True, accepted_phase="overfit")
    contact_state_path = model / "grasp_contact_trainer_state.json"
    contact_state = json.loads(contact_state_path.read_text(encoding="utf-8"))
    del contact_state["training_phase"]
    contact_state_path.write_text(json.dumps(contact_state), encoding="utf-8")
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    validator.validate_phase_transition("sft", model, meta)


def test_later_phase_rejects_config_only_checkpoint(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"use_llm_lora": 32, "grasp_task_token_ids": [1, 2]}),
        encoding="utf-8",
    )
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    with pytest.raises(ValueError, match="no inspectable model weights"):
        validator.validate_phase_transition("pair", model, meta)


def test_same_phase_resume_requires_matching_training_phase(tmp_path: Path) -> None:
    model = tmp_path / "base"
    _write_checkpoint(model, grasp=True, accepted_phase="overfit")
    resume = tmp_path / "resume"
    _write_checkpoint(resume, grasp=True, accepted_phase="sft")
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    validator.validate_phase_transition(
        "sft", model, meta, resume_from_checkpoint=resume
    )

    with pytest.raises(ValueError, match="expected 'pair'"):
        validator.validate_phase_transition(
            "pair", model, meta, resume_from_checkpoint=resume
        )


def test_same_phase_weight_restart_is_explicit_and_accepts_legacy_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, grasp=True, accepted_phase="sft")
    contact_state_path = checkpoint / "grasp_contact_trainer_state.json"
    contact_state = json.loads(contact_state_path.read_text(encoding="utf-8"))
    del contact_state["training_phase"]
    del contact_state["data_fingerprint"]
    contact_state_path.write_text(json.dumps(contact_state), encoding="utf-8")
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    with pytest.raises(ValueError, match="requires an accepted overfit"):
        validator.validate_phase_transition("sft", checkpoint, meta)

    validator.validate_phase_transition(
        "sft",
        checkpoint,
        meta,
        allow_same_phase_weight_restart=True,
    )


def test_same_phase_weight_restart_rejects_cross_phase_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint, grasp=True, accepted_phase="pair")
    meta = tmp_path / "full_meta.json"
    _write_meta(meta, "/tmp/contact_train_grasp_v2.jsonl")

    with pytest.raises(ValueError, match="same-phase weight restart"):
        validator.validate_phase_transition(
            "sft",
            checkpoint,
            meta,
            allow_same_phase_weight_restart=True,
        )


def test_overfit_phase_does_not_require_a_grasp_checkpoint(tmp_path: Path) -> None:
    validator.validate_phase_transition(
        "overfit", tmp_path / "missing-model", tmp_path / "missing-meta"
    )


def test_pair_geometry_and_multigt_use_distinct_candidate_curricula() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    pair = re.search(r"\n  pair\)(.*?)\n    ;;", source, re.DOTALL)
    geometry = re.search(r"\n  geometry\)(.*?)\n    ;;", source, re.DOTALL)
    later = re.search(r"\n  negative\|multigt\)(.*?)\n    ;;", source, re.DOTALL)

    assert pair is not None
    assert geometry is not None
    assert later is not None
    assert 'active_candidates="${CONTACT_PAIR_MAX_CANDIDATES:-1}"' in pair.group(1)
    assert "active_candidates=1" in geometry.group(1)
    assert "CONTACT_MAX_CANDIDATES" not in geometry.group(1)
    assert '[[ "${phase}" == "multigt" ]]' in later.group(1)
    assert 'active_candidates="${CONTACT_MAX_CANDIDATES:-8}"' in later.group(1)


def test_training_does_not_implicitly_resume_last_checkpoint() -> None:
    source = EAGLE_TRAIN_SCRIPT.read_text(encoding="utf-8")

    assert "training_args.resume_from_checkpoint or last_checkpoint" not in source
    assert "no explicit " in source
    assert "RESUME_FROM_CHECKPOINT was provided" in source


def test_packed_training_limits_supported_tensor_batch_size() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")

    assert 'per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"' in source
    assert '"${per_device_train_batch_size}" != "2"' in source
    assert "supports PER_DEVICE_TRAIN_BATCH_SIZE=1 or 2" in source


def test_runtime_training_controls_survive_config_sourcing() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")

    for name in (
        "WARMUP_RATIO",
        "WEIGHT_DECAY",
        "MAX_GRAD_NORM",
        "LR_SCHEDULER_TYPE",
        "LOGGING_STEPS",
        "SAVE_STEPS",
        "SAVE_TOTAL_LIMIT",
        "SEED",
    ):
        assert name in source.split("declare -A environment_overrides", 1)[0]


def test_grasp_only_mode_explicitly_disables_grounding_gate() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")

    assert 'grasp_only="${GRASP_ONLY:-0}"' in source
    assert "grounding replay is intentionally disabled" in source
    grasp_only = source.split('if [[ "${grasp_only}" == "1"', 1)[1]
    assert "min_grounding_samples=0" in grasp_only
    assert "min_grounding_fraction=0.0" in grasp_only


def test_training_uses_pinned_eagle_bootstrap() -> None:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")

    assert 'scripts/bootstrap_eagle.sh"' in source
    assert "--no-clone" in source
    assert "skipping Git patch check" not in source
