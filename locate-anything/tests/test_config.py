from locate_anything_service.config import Settings


def test_settings_defaults_are_values(monkeypatch) -> None:
    monkeypatch.delenv("LOCATE_MODEL_ID", raising=False)
    monkeypatch.delenv("LOCATE_DEVICE", raising=False)

    settings = Settings.from_env()

    assert settings.model_id == "nvidia/LocateAnything-3B"
    assert settings.model_revision == "c32291ca5e996f5a7a485845b4f57a233936bba0"
    assert settings.device == "cuda"
