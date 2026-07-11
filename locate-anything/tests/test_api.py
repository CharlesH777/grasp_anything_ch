import asyncio
import io
import time

import httpx
from PIL import Image

import locate_anything_service.api as api
from locate_anything_service.schemas import LocateResponse


class FakeRuntime:
    loaded = True
    device = "cpu"
    load_error = None

    def __init__(self) -> None:
        self.received_generation_mode = "unset"

    def predict(self, _path, query, mode, generation_mode, annotate):
        self.received_generation_mode = generation_mode
        time.sleep(0.2)
        return LocateResponse(
            model="fake",
            mode=mode,
            generation_mode="slow",
            prompt=query,
            raw_output="<box>None</box>",
            image_width=8,
            image_height=8,
            generation_stats={"generation_seconds": 0.2, "box_count": 1},
        )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_inference_does_not_block_event_loop(monkeypatch) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(api, "RUNTIME", runtime)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            locate_task = asyncio.create_task(
                client.post(
                    "/v1/locate",
                    files={"image": ("image.png", _png_bytes(), "image/png")},
                    data={"query": "object"},
                )
            )
            await asyncio.sleep(0.02)
            assert not locate_task.done()

            health_response = await client.get("/healthz")
            assert health_response.status_code == 200

            locate_response = await locate_task
            assert locate_response.status_code == 200

    asyncio.run(scenario())
    assert runtime.received_generation_mode is None
