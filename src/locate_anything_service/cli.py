from __future__ import annotations

import argparse
import base64
from pathlib import Path

import uvicorn

from .config import Settings
from .model import LocateAnythingRuntime
from .preflight import format_results, run_preflight


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
            "grasp_rect",
        ),
        default="ground_single",
    )
    predict.add_argument(
        "--generation-mode", choices=("fast", "hybrid", "slow"), default=None
    )
    predict.add_argument("--annotated-output", type=Path)

    doctor = subparsers.add_parser("doctor", help="validate a deployment")
    doctor.add_argument(
        "--require-grasp",
        action="store_true",
        help="require a local checkpoint with grasp task token IDs",
    )
    doctor.add_argument(
        "--require-grasp-rect",
        action="store_true",
        help="require a local checkpoint with grasp rect task token IDs",
    )
    doctor.add_argument(
        "--skip-cuda",
        action="store_true",
        help="check files and dependencies without probing CUDA",
    )
    doctor.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = Settings.from_env()

    if args.command == "doctor":
        results = run_preflight(
            settings,
            require_grasp=(args.require_grasp or settings.require_grasp_checkpoint),
            require_grasp_rect=(
                args.require_grasp_rect
                or settings.require_grasp_rect_checkpoint
            ),
            check_cuda=not args.skip_cuda,
        )
        print(format_results(results, json_output=args.json_output))
        if not all(item.ok for item in results):
            raise SystemExit(1)
        return

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
