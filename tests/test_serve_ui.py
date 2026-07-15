from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_ui.py"
SPEC = importlib.util.spec_from_file_location("serve_ui", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
serve_ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve_ui)


def test_background_task_is_retained_until_completion() -> None:
    async def scenario() -> None:
        serve_ui.app.state.background_tasks.clear()
        release = asyncio.Event()

        async def wait_for_release() -> None:
            await release.wait()

        task = serve_ui._track_background_task(wait_for_release())
        assert task in serve_ui.app.state.background_tasks

        release.set()
        await task
        await asyncio.sleep(0)
        assert task not in serve_ui.app.state.background_tasks

    asyncio.run(scenario())
