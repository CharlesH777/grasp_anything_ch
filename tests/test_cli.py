import sys

import locate_anything_service.cli as cli


def test_serve_preserves_port_zero_and_empty_host(monkeypatch) -> None:
    captured = {}

    def fake_run(app, **kwargs) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["grasp-anything", "serve", "--port", "0", "--host", ""]
    )

    cli.main()

    assert captured["port"] == 0
    assert captured["host"] == ""


def test_predict_generation_mode_defaults_to_none() -> None:
    args = cli._parser().parse_args(["predict", "image.png", "object"])

    assert args.generation_mode is None
