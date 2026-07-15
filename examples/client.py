from __future__ import annotations

import argparse
import base64
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("query")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", default="ground_single")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.image.open("rb") as image_file:
        response = httpx.post(
            f"{args.url}/v1/locate",
            files={"image": (args.image.name, image_file, "image/jpeg")},
            data={
                "query": args.query,
                "mode": args.mode,
                "generation_mode": "hybrid",
                "annotate": str(args.output is not None).lower(),
            },
            timeout=300,
            trust_env=False,
        )
    response.raise_for_status()
    result = response.json()
    if args.output and result.get("annotated_image_base64"):
        args.output.write_bytes(base64.b64decode(result["annotated_image_base64"]))
    print(result)


if __name__ == "__main__":
    main()
