#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "peft": "0.12.0",
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.57.1",
}


def validate_versions() -> None:
    errors: list[str] = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            errors.append(f"{package} is not installed (expected {expected})")
            continue
        if Version(installed).base_version != expected:
            errors.append(f"{package}=={installed} (expected {expected})")
    if errors:
        detail = "; ".join(errors)
        raise RuntimeError(
            "unsupported training environment: "
            f"{detail}. Install the pinned project model dependencies."
        )


def validate_cuda_toolkit() -> None:
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA_HOME has no bin/nvcc: {cuda_home}")
    try:
        output = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot execute {nvcc}: {error}") from error
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        raise RuntimeError(f"cannot determine CUDA toolkit version from {nvcc}")

    import torch

    torch_cuda = torch.version.cuda
    if torch_cuda is not None and match.group(1) != torch_cuda:
        raise RuntimeError(
            f"CUDA toolkit {match.group(1)} at {cuda_home} does not match "
            f"PyTorch CUDA {torch_cuda}"
        )


def main() -> int:
    try:
        validate_versions()
        validate_cuda_toolkit()
    except RuntimeError as error:
        print(str(error), file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
