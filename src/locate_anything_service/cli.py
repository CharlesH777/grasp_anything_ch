from __future__ import annotations

import argparse
import base64
from pathlib import Path

import uvicorn

from .config import Settings
from .model import LocateAnythingRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grasp-anything")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    predict = subparsers.add_parser("predict", help="run inference on one image")
    predict.add_argument("image", type=Path)
    predict.add_argument("query", nargs="?", default="")
    predict.add_argument(
        "--mode",
        choices=(
            "raw",
            "detect",
            "ground_single",
            "ground_multi",
            "ground_text",
            "detect_text",
            "gui_box",
            "gui_point",
            "point",
            "grasp_contact",
        ),
        default="ground_single",
    )
    predict.add_argument(
        "--generation-mode", choices=("fast", "hybrid", "slow"), default=None
    )
    predict.add_argument("--annotated-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = Settings.from_env()

    if args.command == "serve":
        uvicorn.run(
            "locate_anything_service.api:app",
            host=settings.host if args.host is None else args.host,
            port=settings.port if args.port is None else args.port,
            log_level=settings.log_level,
        )
        return

    runtime = LocateAnythingRuntime(settings)
    result = runtime.predict(
        args.image,
        query=args.query,
        mode=args.mode,
        generation_mode=args.generation_mode,
        annotate=args.annotated_output is not None,
    )
    if args.annotated_output and result.annotated_image_base64:
        args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
        args.annotated_output.write_bytes(
            base64.b64decode(result.annotated_image_base64)
        )
    print(result.model_dump_json(indent=2, exclude={"annotated_image_base64"}))


if __name__ == "__main__":
    main()
