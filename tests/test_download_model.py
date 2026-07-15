import importlib.util
from pathlib import Path


def _load_download_module():
    script_path = Path(__file__).parents[1] / "scripts" / "download_model.py"
    spec = importlib.util.spec_from_file_location("download_model", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_uses_pinned_default_revision(monkeypatch, capsys) -> None:
    download_model = _load_download_module()
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/tmp/model"

    monkeypatch.delenv("LOCATE_MODEL_ID", raising=False)
    monkeypatch.delenv("LOCATE_MODEL_REVISION", raising=False)
    monkeypatch.setattr(download_model, "snapshot_download", fake_snapshot_download)

    download_model.main()

    assert captured["repo_id"] == "nvidia/LocateAnything-3B"
    assert captured["revision"] == "c32291ca5e996f5a7a485845b4f57a233936bba0"
    assert capsys.readouterr().out.strip() == "/tmp/model"
