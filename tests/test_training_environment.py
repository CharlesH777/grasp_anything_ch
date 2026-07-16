from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "scripts"
    / "validate_training_environment.py"
)
SPEC = importlib.util.spec_from_file_location("validate_training_environment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def test_training_environment_accepts_local_pinned_stack() -> None:
    validator.validate_versions()


def test_training_environment_reports_all_drift(monkeypatch) -> None:
    versions = dict(validator.EXPECTED_VERSIONS)
    versions["torch"] = "9.9.9"
    monkeypatch.setattr(validator, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"torch==9\.9\.9"):
        validator.validate_versions()


def test_training_environment_rejects_mismatched_cuda_toolkit(
    monkeypatch, tmp_path: Path
) -> None:
    cuda_home = tmp_path / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 10.1'\n")
    nvcc.chmod(0o755)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))

    with pytest.raises(RuntimeError, match="does not match PyTorch CUDA"):
        validator.validate_cuda_toolkit()
