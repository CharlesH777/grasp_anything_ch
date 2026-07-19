from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _result(name: str, ok: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=ok, detail=detail)


def _is_local_reference(model_id: str) -> bool:
    path = Path(model_id).expanduser()
    return path.is_absolute() or model_id.startswith((".", "~")) or path.exists()


def _load_local_config(model_path: Path) -> tuple[dict[str, Any] | None, str]:
    config_path = model_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        return None, f"cannot read {config_path}: {error}"
    except json.JSONDecodeError as error:
        return None, f"invalid JSON in {config_path}: {error}"
    if not isinstance(payload, dict):
        return None, f"{config_path} must contain a JSON object"
    return payload, str(config_path)


def _check_local_model(model_path: Path, require_grasp: bool) -> list[CheckResult]:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        return [_result("model_path", False, f"directory not found: {model_path}")]

    required = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    has_weights = (model_path / "model.safetensors").is_file() or (
        model_path / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model.safetensors or model.safetensors.index.json")
    results = [
        _result(
            "model_files",
            not missing,
            f"checkpoint={model_path}"
            if not missing
            else f"missing from {model_path}: {', '.join(missing)}",
        )
    ]

    config, detail = _load_local_config(model_path)
    if config is None:
        results.append(_result("model_config", False, detail))
        return results
    results.append(_result("model_config", True, detail))

    if require_grasp:
        token_ids = config.get("grasp_task_token_ids")
        valid = (
            isinstance(token_ids, list)
            and len(token_ids) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in token_ids
            )
            and token_ids[0] != token_ids[1]
        )
        results.append(
            _result(
                "grasp_checkpoint",
                valid,
                "grasp task token IDs are present and distinct"
                if valid
                else "config.json needs two distinct integer grasp_task_token_ids",
            )
        )
    return results


def run_preflight(
    settings: Settings,
    *,
    require_grasp: bool | None = None,
    check_cuda: bool = True,
) -> list[CheckResult]:
    require_grasp = (
        settings.require_grasp_checkpoint
        if require_grasp is None
        else require_grasp
    )
    version = sys.version_info
    results = [
        _result(
            "python",
            (3, 10) <= version[:2] < (3, 13),
            f"{version.major}.{version.minor}.{version.micro} (supported: 3.10-3.12)",
        )
    ]

    imports = {
        "PIL": "Pillow",
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "torch": "torch",
        "transformers": "transformers",
    }
    missing = [
        package
        for module, package in imports.items()
        if importlib.util.find_spec(module) is None
    ]
    results.append(
        _result(
            "dependencies",
            not missing,
            "runtime dependencies are importable"
            if not missing
            else f"missing packages: {', '.join(missing)}",
        )
    )

    if _is_local_reference(settings.model_id):
        results.extend(
            _check_local_model(Path(settings.model_id), require_grasp=require_grasp)
        )
    else:
        results.append(
            _result(
                "model_reference",
                not require_grasp,
                f"remote model ID: {settings.model_id}"
                if not require_grasp
                else (
                    "grasp checkpoint validation requires a local checkpoint path; "
                    f"got remote ID {settings.model_id!r}"
                ),
            )
        )

    if check_cuda and settings.device.startswith("cuda") and not missing:
        try:
            import torch

            available = torch.cuda.is_available()
            ok = available or settings.allow_cpu
            detail = (
                f"CUDA available ({torch.cuda.get_device_name(0)})"
                if available
                else (
                    "CUDA unavailable; LOCATE_ALLOW_CPU permits debug fallback"
                    if settings.allow_cpu
                    else "CUDA unavailable and LOCATE_ALLOW_CPU is disabled"
                )
            )
            results.append(_result("device", ok, detail))
        except Exception as error:
            results.append(_result("device", False, f"cannot inspect CUDA: {error}"))
    return results


def format_results(results: list[CheckResult], *, json_output: bool = False) -> str:
    if json_output:
        return json.dumps(
            {
                "ok": all(item.ok for item in results),
                "checks": [asdict(item) for item in results],
            },
            indent=2,
            sort_keys=True,
        )
    return "\n".join(
        f"[{'OK' if item.ok else 'FAIL'}] {item.name}: {item.detail}"
        for item in results
    )
