#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _load_stats(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stats file must contain a JSON object: {path}")
    return payload


def merge(
    shards: list[Path],
    stats_shards: list[Path],
    output: Path,
    stats_output: Path,
) -> dict[str, Any]:
    if not shards or len(shards) != len(stats_shards):
        raise ValueError("shards and stats-shards must have the same non-zero length")
    for path in (*shards, *stats_shards):
        if not path.is_file():
            raise FileNotFoundError(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for shard in shards:
            with shard.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    temporary.replace(output)

    statistics: Counter[str] = Counter()
    filter_reasons: Counter[str] = Counter()
    configurations: list[dict[str, Any]] = []
    for path in stats_shards:
        payload = _load_stats(path)
        statistics.update(payload.get("statistics", {}))
        filter_reasons.update(payload.get("filter_reasons", {}))
        configuration = payload.get("configuration", {})
        if not isinstance(configuration, dict):
            raise ValueError(f"invalid configuration in {path}")
        configurations.append(configuration)

    common_configuration = {
        key: value
        for key, value in configurations[0].items()
        if key not in {"scene_start", "scene_end_exclusive"}
    }
    for configuration in configurations[1:]:
        comparable = {
            key: value
            for key, value in configuration.items()
            if key not in {"scene_start", "scene_end_exclusive"}
        }
        if comparable != common_configuration:
            raise ValueError("shard configurations do not match")
    common_configuration["scene_shards"] = [
        [config.get("scene_start"), config.get("scene_end_exclusive")]
        for config in configurations
    ]
    result = {
        "statistics": dict(sorted(statistics.items())),
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "configuration": common_configuration,
    }
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge ordered RealVLG JSONL shards.")
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--stats-shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.shards, args.stats_shards, args.output, args.stats_output)
    print(json.dumps(result["statistics"], ensure_ascii=False, sort_keys=True))
    print(f"Wrote: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
