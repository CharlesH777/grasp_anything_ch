# LocateAnything 标准化部署

这是 NVIDIA `LocateAnything-3B` 的工程化部署封装，提供 CLI、FastAPI、Docker Compose、结构化坐标解析和可视化输出。模型代码与权重仍从官方 Hugging Face 仓库加载，本项目不修改模型算法。

## 目录结构

```text
locate-anything/
├── src/locate_anything_service/  # 业务包：配置、模型、API、解析、可视化
├── tests/                        # 不依赖 GPU 的单元测试
├── scripts/                      # 模型预下载等运维脚本
├── examples/                     # API 客户端示例
├── Dockerfile                    # 单容器生产镜像
├── compose.yaml                  # GPU、端口、缓存卷编排
├── pyproject.toml                # Python 包与锁定依赖
├── Makefile                      # 常用开发/部署命令
└── .env.example                  # 环境变量模板
```

模型缓存不进入代码仓库，通过 `HF_HOME` 或 Docker named volume 独立管理。

## 环境要求

- Linux x86_64
- NVIDIA GPU；BF16 推理建议至少 16 GB 显存
- 可用的 NVIDIA 驱动和 `nvidia-container-toolkit`
- Docker 24+ 与 Docker Compose v2，或 Python 3.10–3.12
- 首次启动需要下载约 7.4 GB 模型文件，并为 Python/CUDA 依赖预留额外磁盘空间

先验证容器能够访问 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Docker 部署

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f locate-anything
```

首次启动会把模型下载到 `huggingface-cache` volume。服务地址：

- 健康检查：`GET http://127.0.0.1:8000/healthz`
- Swagger：`http://127.0.0.1:8000/docs`
- 推理：`POST http://127.0.0.1:8000/v1/locate`

```bash
curl -X POST http://127.0.0.1:8000/v1/locate \
  -F 'image=@assets/test.jpg' \
  -F 'query=red cup' \
  -F 'mode=ground_single' \
  -F 'generation_mode=hybrid' \
  -F 'annotate=true'
```

## 本地部署

不要使用本机 Python 3.13；官方依赖组合应使用 Python 3.10–3.12。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
cp .env.example .env
set -a && source .env && set +a
make download-model
make serve
```

安装为当前用户的 systemd 常驻服务：

```bash
make service-install
make service-status
journalctl --user -u locate-anything.service -f
```

服务单元由 `deploy/systemd/locate-anything.service.in` 生成，自动写入当前项目绝对路径，无需手工修改。

单图 CLI：

```bash
locate-anything predict assets/test.jpg 'red cup' \
  --mode ground_single \
  --generation-mode hybrid \
  --annotated-output outputs/result.png
```

## 任务模式

| `mode` | 用途 | `query` 示例 |
|---|---|---|
| `detect` | 指定类别检测 | `person, car, bicycle` |
| `ground_single` | 单目标自然语言定位 | `the leftmost red cup` |
| `ground_multi` | 多目标自然语言定位 | `all workers wearing helmets` |
| `ground_text` | 指定文字定位 | `TOTAL` |
| `detect_text` | 全图文字检测 | 留空 |
| `gui_box` | GUI 控件框定位 | `submit button` |
| `gui_point` | GUI 点击点定位 | `settings icon` |
| `point` | 通用点定位 | `center of the mug handle` |
| `raw` | 直接发送官方 prompt | 完整 prompt |

`generation_mode` 支持：

- `hybrid`：默认推荐，在 PBD 与逐 token 回退之间自动切换。
- `fast`：优先吞吐量。
- `slow`：完全自回归，优先稳定性。

## API 返回

返回中保留模型原始文本，并同时给出三套坐标：

- `coordinates_1000`：模型的 `[0,1000]` 离散坐标。
- `normalized`：归一化到 `[0,1]`。
- `pixels`：按原图宽高换算后的像素坐标。
- `annotated_image_base64`：仅在 `annotate=true` 时返回 PNG。

## 运维说明

- `/healthz` 只有在模型完成加载后才返回 `status=ok`。
- 单进程内通过锁串行访问模型，避免共享 KV/cache 状态产生并发错误。
- 扩容时建议“一张 GPU 一个服务副本”，不要在同一 GPU 上启动多个完整模型副本。
- 私有镜像或受限网络可先执行 `make download-model`，再将 `LOCATE_MODEL_ID` 指向本地模型目录。
- 模型采用 Hugging Face remote code，生产环境应把 `LOCATE_MODEL_REVISION` 固定到经过验证的模型仓库 commit，而不是长期使用 `main`。

## 验证

```bash
make lint
make test
docker compose config
```
