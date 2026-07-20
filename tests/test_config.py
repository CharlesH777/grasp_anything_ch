import pytest

from locate_anything_service.config import Settings


def test_settings_defaults_are_values(monkeypatch) -> None:
    monkeypatch.delenv("LOCATE_MODEL_ID", raising=False)
    monkeypatch.delenv("LOCATE_DEVICE", raising=False)

    settings = Settings.from_env()

    assert settings.model_id == "nvidia/LocateAnything-3B"
    assert settings.model_revision == "c32291ca5e996f5a7a485845b4f57a233936bba0"
    assert settings.device == "cuda"
    assert settings.require_grasp_checkpoint is False


def test_settings_reject_invalid_contact_threshold() -> None:
    with pytest.raises(ValueError, match="contact_decode_coord_mass_threshold"):
        Settings(contact_decode_coord_mass_threshold=-0.1)


def test_settings_reads_required_grasp_checkpoint(monkeypatch) -> None:
    monkeypatch.setenv("LOCATE_REQUIRE_GRASP_CHECKPOINT", "1")

    assert Settings.from_env().require_grasp_checkpoint is True


def test_settings_reads_grasp_rect_geometry(monkeypatch) -> None:
    monkeypatch.setenv("LOCATE_REQUIRE_GRASP_RECT_CHECKPOINT", "1")
    monkeypatch.setenv("LOCATE_GRASP_RECT_MINIMUM_WIDTH_DIAGONAL", "0.02")
    monkeypatch.setenv("LOCATE_GRASP_RECT_COORD_ENTROPY_THRESHOLD", "0.8")
    monkeypatch.setenv("LOCATE_GRASP_RECT_GRIPPER_DEPTH_PIXELS", "42")

    settings = Settings.from_env()

    assert settings.require_grasp_rect_checkpoint is True
    assert settings.grasp_rect_minimum_width_diagonal == 0.02
    assert settings.grasp_rect_decode_coord_entropy_threshold == 0.8
    assert settings.grasp_rect_gripper_depth_pixels == 42
