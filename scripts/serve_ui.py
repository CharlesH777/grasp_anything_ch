#!/usr/bin/env python3
"""grasp_anything UI 代理 + 模型管理服务器

管理推理 API 子进程，支持运行时切换模型权重：
  - GET  /            → Web UI
  - GET  /v1/models   → 可用模型列表
  - POST /v1/switch   → 切换模型 (杀旧起新)
  - GET  /healthz     → 代理健康检查
  - POST /v1/locate   → 代理推理请求

用法:
    python scripts/serve_ui.py
    python scripts/serve_ui.py --port 8001 --api-port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

HTML_PATH = Path(__file__).resolve().parent / "index.html"
MODELS_PATH = Path(__file__).resolve().parent / "models.json"
PROXY_TIMEOUT = 300

app = FastAPI(title="grasp_anything UI Proxy", docs_url=None, redoc_url=None)
app.state.api_base = "http://127.0.0.1:8000"
app.state.models = {}
app.state.current_model = "official"
app.state.switching = False
app.state.background_tasks = set()


# ── API 子进程管理 ────────────────────────────────────────
class APIManager:
    def __init__(self, project_dir: str, env_file: str, api_port: int):
        self.project_dir = Path(project_dir)
        self.env_file = Path(env_file)
        self.api_port = api_port
        self.process: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self._log_fh = None

    def _parse_env(self) -> dict:
        env = os.environ.copy()
        if self.env_file.exists():
            for line in self.env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
        return env

    def start(self, model_config: dict) -> None:
        if self.process and self.process.poll() is None:
            raise RuntimeError("API already running, call stop() first")

        env = self._parse_env()
        env.update(model_config)
        env["LOCATE_PORT"] = str(self.api_port)
        env["LOCATE_HOST"] = "0.0.0.0"

        log_dir = self.project_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / f"api-{time.strftime('%Y%m%d-%H%M%S')}.log"

        venv_python = str(self.project_dir / ".venv" / "bin" / "python")
        cmd = [venv_python, "-m", "locate_anything_service.cli", "serve"]

        self._log_fh = open(self.log_file, "w")  # noqa: SIM115
        self.process = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(self.project_dir),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.process = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def is_healthy(self) -> bool:
        try:
            r = urllib.request.urlopen(
                f"http://127.0.0.1:{self.api_port}/healthz", timeout=5
            )
            data = json.loads(r.read())
            return data.get("status") == "ok" and data.get("model_loaded", False)
        except Exception:
            return False

    def wait_for_health(self, timeout: int = 300) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                err = ""
                if self.log_file and self.log_file.exists():
                    err = "\n".join(self.log_file.read_text().splitlines()[-20:])
                return False, f"API process exited\n{err}"
            if self.is_healthy():
                return True, ""
            time.sleep(2)
        return False, f"timeout after {timeout}s"

    def tail_log(self, n: int = 20) -> str:
        if self.log_file and self.log_file.exists():
            return "\n".join(self.log_file.read_text().splitlines()[-n:])
        return ""


api_manager: APIManager | None = None


# ── 静态 UI ──────────────────────────────────────────────
@app.get("/")
@app.get("/ui")
async def ui() -> HTMLResponse:
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


# ── 模型管理 ──────────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    return {
        "current": app.state.current_model,
        "switching": app.state.switching,
        "models": {
            k: {"name": v["name"], "desc": v["desc"]}
            for k, v in app.state.models.items()
        },
    }


@app.post("/v1/switch")
async def switch_model(request: Request):
    body = await request.json()
    model_key = body.get("model", "")
    if model_key not in app.state.models:
        return JSONResponse({"detail": f"unknown model: {model_key}"}, status_code=400)
    if api_manager is None:
        return JSONResponse({"detail": "API manager not initialized"}, status_code=503)
    if app.state.switching:
        return JSONResponse({"detail": "already switching"}, status_code=400)
    if model_key == app.state.current_model and api_manager.is_healthy():
        return {"status": "ok", "model": model_key, "message": "already active"}

    app.state.switching = True
    prev_model = app.state.current_model
    try:
        cfg = app.state.models[model_key]
        env_cfg = {k: v for k, v in cfg.items() if k not in ("name", "desc")}
        await asyncio.to_thread(api_manager.stop)
        await asyncio.to_thread(api_manager.start, env_cfg)
        ok, err = await asyncio.to_thread(api_manager.wait_for_health, 300)
        if not ok:
            # rollback: try to restart previous model
            rollback_msg = ""
            if prev_model != model_key and prev_model in app.state.models:
                try:
                    prev_cfg = app.state.models[prev_model]
                    prev_env = {
                        k: v for k, v in prev_cfg.items() if k not in ("name", "desc")
                    }
                    await asyncio.to_thread(api_manager.stop)
                    await asyncio.to_thread(api_manager.start, prev_env)
                    rb_ok, _ = await asyncio.to_thread(
                        api_manager.wait_for_health, 300
                    )
                    rollback_msg = (
                        f"; rolled back to '{prev_model}'"
                        if rb_ok
                        else "; rollback also failed, no API running"
                    )
                except Exception:
                    rollback_msg = "; rollback also failed, no API running"
            return JSONResponse(
                {"detail": f"model load failed: {err}{rollback_msg}",
                 "log": api_manager.tail_log()},
                status_code=503,
            )
        app.state.current_model = model_key
        return {"status": "ok", "model": model_key}
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
    finally:
        app.state.switching = False


# ── 代理 ──────────────────────────────────────────────────
@app.api_route("/healthz", methods=["GET"])
@app.api_route("/v1/locate", methods=["POST"])
async def proxy(request: Request) -> Response:
    if app.state.switching:
        return JSONResponse(
            {"detail": "model is switching, please wait"}, status_code=503
        )
    target = f"{app.state.api_base}{request.url.path}?{request.url.query}"
    body = await request.body()
    skip = {"host", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT, trust_env=False) as client:
            if request.method == "GET":
                resp = await client.get(target, headers=headers)
            else:
                resp = await client.post(target, content=body, headers=headers)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return JSONResponse(
            {"status": "error", "detail": "API service unavailable",
             "model_loaded": False},
            status_code=503,
        )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ── 生命周期 ──────────────────────────────────────────────
async def _monitor_api_startup():
    """后台等待 API 加载完成，不阻塞 UI 启动。"""
    ok, err = await asyncio.to_thread(api_manager.wait_for_health, 300)
    if not ok:
        print(f"[serve_ui] API 启动失败: {err}")
        print(f"[serve_ui] 日志末尾:\n{api_manager.tail_log()}")


def _track_background_task(coroutine) -> asyncio.Task:
    task = asyncio.create_task(coroutine)
    app.state.background_tasks.add(task)
    task.add_done_callback(app.state.background_tasks.discard)
    return task


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    if api_manager and app.state.models:
        if app.state.current_model not in app.state.models:
            app.state.current_model = next(iter(app.state.models))
        cfg = app.state.models[app.state.current_model]
        env_cfg = {k: v for k, v in cfg.items() if k not in ("name", "desc")}
        await asyncio.to_thread(api_manager.start, env_cfg)
        _track_background_task(_monitor_api_startup())
    yield
    # shutdown
    if api_manager:
        await asyncio.to_thread(api_manager.stop)


app.router.lifespan_context = lifespan


# ── 入口 ──────────────────────────────────────────────────
def main() -> None:
    global api_manager

    parser = argparse.ArgumentParser(description="grasp_anything UI 代理服务器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001, help="UI 监听端口")
    parser.add_argument("--api-port", type=int, default=8000, help="API 监听端口")
    project_dir = str(Path(__file__).resolve().parent.parent)
    parser.add_argument("--project-dir", default=project_dir)
    parser.add_argument("--env-file", default=f"{project_dir}/.env")
    parser.add_argument("--models-file", default=str(MODELS_PATH))
    parser.add_argument(
        "--default-model",
        default=None,
        help="启动时加载的模型 key (默认 official)",
    )
    args = parser.parse_args()

    if MODELS_PATH.exists():
        app.state.models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    if args.models_file != str(MODELS_PATH) and Path(args.models_file).exists():
        app.state.models = json.loads(
            Path(args.models_file).read_text(encoding="utf-8")
        )

    app.state.api_base = f"http://127.0.0.1:{args.api_port}"
    if args.default_model and args.default_model in app.state.models:
        app.state.current_model = args.default_model
    api_manager = APIManager(
        project_dir=args.project_dir,
        env_file=args.env_file,
        api_port=args.api_port,
    )

    default_name = app.state.models.get(app.state.current_model, {}).get("name", "?")
    print("\n  grasp_anything UI")
    print("  ─────────────────────────────────")
    print(f"  界面:     http://127.0.0.1:{args.port}")
    print(f"  API:      {app.state.api_base}")
    print(f"  默认模型: {default_name}")
    print(f"  可选模型: {', '.join(app.state.models.keys())}")
    print("  ─────────────────────────────────\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
