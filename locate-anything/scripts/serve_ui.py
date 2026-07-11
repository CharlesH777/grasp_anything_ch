#!/usr/bin/env python3
"""LocateAnything UI 独立代理服务器

不修改原有 API 代码。本脚本启动一个轻量 HTTP 服务：
  - 在 / 和 /ui 返回 Web 界面
  - 将 /healthz 和 /v1/locate 透明转发到主推理服务

用法:
    python scripts/serve_ui.py
    python scripts/serve_ui.py --port 8001 --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response

HTML_PATH = Path(__file__).resolve().parent / "index.html"
PROXY_TIMEOUT = 300

app = FastAPI(title="LocateAnything UI Proxy", docs_url=None, redoc_url=None)
app.state.api_base = "http://127.0.0.1:8000"


# ── 静态 UI ──────────────────────────────────────────────
@app.get("/")
@app.get("/ui")
async def ui() -> HTMLResponse:
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


# ── 透明代理 ──────────────────────────────────────────────
@app.api_route("/healthz", methods=["GET"])
@app.api_route("/v1/locate", methods=["POST"])
async def proxy(request: Request) -> Response:
    target = f"{app.state.api_base}{request.url.path}?{request.url.query}"
    # 读取原始 body，保留 multipart boundary 不动
    body = await request.body()
    # 只保留必要的 header，丢弃 hop-by-hop
    skip = {"host", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}

    async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
        if request.method == "GET":
            resp = await client.get(target, headers=headers)
        else:
            resp = await client.post(target, content=body, headers=headers)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


# ── 入口 ──────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LocateAnything 独立 UI 代理服务器"
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000",
        help="主推理服务地址（默认 http://127.0.0.1:8000）",
    )
    args = parser.parse_args()

    app.state.api_base = args.api.rstrip("/")

    print(f"\n  LocateAnything UI")
    print(f"  ─────────────────────────────────")
    print(f"  界面:  http://127.0.0.1:{args.port}")
    print(f"  代理:  {app.state.api_base}")
    print(f"  ─────────────────────────────────\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
