from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .middleware import RequestBodyLimitMiddleware
from .model import LocateAnythingRuntime
from .prompts import PromptMode
from .schemas import HealthResponse, LocateResponse

LOGGER = logging.getLogger(__name__)
SETTINGS = Settings.from_env()
RUNTIME = LocateAnythingRuntime(SETTINGS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if SETTINGS.load_model_on_startup:
        try:
            RUNTIME.load()
        except Exception:
            LOGGER.exception(
                "Model startup failed; health endpoint will report degraded"
            )
    yield


app = FastAPI(
    title="grasp_anything API",
    version="0.1.0",
    description=(
        "Language-guided 2D contact grasping built on NVIDIA LocateAnything-3B."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=(SETTINGS.max_upload_mb + 1) * 1024 * 1024,
)


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if RUNTIME.loaded else "degraded",
        model_loaded=RUNTIME.loaded,
        model=SETTINGS.model_id,
        device=RUNTIME.device,
        detail=RUNTIME.load_error,
    )


@app.post("/v1/locate", response_model=LocateResponse)
async def locate(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
    query: Annotated[str, Form()] = "",
    mode: Annotated[PromptMode, Form()] = "ground_single",
    generation_mode: Annotated[
        Literal["fast", "hybrid", "slow"] | None, Form()
    ] = None,
    annotate: Annotated[bool, Form()] = False,
) -> LocateResponse:
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="unsupported image type")

    payload = await image.read(SETTINGS.max_upload_mb * 1024 * 1024 + 1)
    if len(payload) > SETTINGS.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image exceeds upload limit")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            return await run_in_threadpool(
                RUNTIME.predict,
                Path(temporary_file.name),
                query=query,
                mode=mode,
                generation_mode=generation_mode,
                annotate=annotate,
            )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("grasp_anything inference failed")
        raise HTTPException(status_code=503, detail=str(error)) from error
