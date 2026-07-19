# grasp_anything

`grasp_anything` 是一个基于 NVIDIA `LocateAnything-3B` 的语言引导二维抓取项目。输入 RGB 图像和自然语言目标描述，模型通过一次 PBD 联合生成两个平行夹爪接触点：

```text
<ref>grasp</ref><grasp><x1><y1><x2><y2></grasp>
```

模型不直接预测存在周期歧义的角度和宽度。二维中心、接触点闭合轴方向（模 `pi`）和像素开口全部由两个接触点确定性计算；二维实例 mask 仅用于投影碰撞率、越界率和 clearance 代理评估。

上游模型、remote-code 类名和 `LOCATE_*` 环境变量保留 LocateAnything 命名。本项目的发行名、CLI、API/UI、Docker 和 systemd 服务统一使用 `grasp_anything` / `grasp-anything`。

跨机器安装、checkpoint 挂载、Docker、systemd 与训练环境的完整步骤见
[`DEPLOYMENT.md`](DEPLOYMENT.md)。最短安装路径：

```bash
bash scripts/bootstrap.sh --service
set -a; source .env; set +a
.venv/bin/grasp-anything doctor
```

## 当前状态

代码侧已经完成：

- `grasp_contact` 服务模式，一次输出一个四坐标接触点块或 `none`。
- contact-aware Fast/Hybrid PBD 联合解码，不再调用两次 `point`。
- 端点交换不变 pair CE、多 GT hard-min、中心/角度/宽度辅助损失。
- 带 90 度确定性非零次梯度的 `1-|cos|` 模 `pi` 角度损失和几何置信度门控。
- fixed-K stream-packing 数据桥、rank/worker 互斥分片、全局 denominator 和梯度累积对齐。
- grasp 输出 adapter 在训练和推理的所有位置使用同一 logits 修正；错误槽位也有负梯度。
- 独立记录坐标 CE 与 top-1 accuracy，不再用总 token loss 代替坐标收敛判断。
- RealVLG-R1 `contact_points` 转换、候选压缩、碰撞与越界候选过滤。
- 严格二维评测：非法正样本计零、修正角度指标、碰撞 unknown 单列。
- 每个 checkpoint 自动带推理 remote code 和 `auto_map`。
- 四卡 BF16 + SDPA + ZeRO-2 分阶段训练，LLM 使用 LoRA。
- MoonViT 最后若干 attention 层可选 rank-N Vision LoRA。
- 全新 Eagle checkout 启动训练时自动校验并应用 contact patch。
- Phase 2 起强制检查 grounding replay，negative/multigt 阶段额外检查可信负样本。

代码仓库不包含数据集、checkpoint、optimizer state 或生成结果；部署抓取服务时必须单独提供通过 `grasp-anything doctor` 校验的完整 checkpoint。

## 核心表示

### 正样本

```text
<ref>grasp</ref><grasp><100><420><760><510></grasp>
```

四个坐标都是 `[0,1000]` 离散 token，槽位顺序固定为 `(x1,y1,x2,y2)`。端点在数据转换时使用字典序规范化，训练损失和评测误差对端点交换保持不变。

### 负样本

```text
<ref>grasp</ref><grasp>none</grasp>
```

只有数据明确证明目标不存在或不可抓取时才能生成 `none`。缺少接触点、mask 不完整或非穷尽候选全部碰撞不能自动当作负样本。

### 二维几何

像素接触点记为 `p1=(x1,y1)`、`p2=(x2,y2)`：

```text
center = (p1 + p2) / 2
opening_width = ||p2 - p1||
closing_axis_direction = atan2(y2-y1, x2-x1) mod pi
```

角度和宽度必须在原图像素空间计算，不能直接在归一化的正方形 token 网格中计算。这里的宽度是像素距离，不是毫米/米制夹爪开口；项目不输出 6-DoF 姿态。

## 数据集与划分

主数据集是 [RealVLG-11B](https://modelscope.cn/datasets/cslinfeili/RealVLG-11B) 中的 `GraspNet_VLG`，使用对象级自然语言描述和 `contact_points` 标注。

| 用途 | scene 范围 | 相机 | 说明 |
|---|---|---|---|
| 训练 | `0000-0099` | kinect | 接触点正样本，多帧可用 |
| seen | `0100-0129` | kinect | 官方评测只取 `0000.json` |
| similar | `0130-0159` | kinect | 官方评测只取 `0000.json` |
| novel | `0160-0189` | kinect | 官方评测只取 `0000.json` |

Cornell、VMRD、OCID 和 Jacquard 当前不混入 GraspNet 主指标。未来加入时必须分别建立原生 test protocol，不能共享 GraspNet 的 80 像素抓取厚度和评测分母。

### 下载

只下载 GraspNet 子集：

```bash
modelscope download cslinfeili/RealVLG-11B GraspNet_VLG.zip \
  --repo-type dataset --revision master --local-dir /data/RealVLG-11B
unzip /data/RealVLG-11B/GraspNet_VLG.zip -d /data
```

完整 zip 下载后应先校验官方 SHA-256：

```text
d2d3f3f3ee81f97d10810af1bc29cfbbf13bbf515c8d1abbc4868807cbde5c02
```

### 转换训练集

```bash
python training/scripts/convert_realvlg_contact.py \
  --data-root /data/GraspNet_VLG \
  --output /data/grasp_anything/contact_train.jsonl \
  --stats /data/grasp_anything/contact_train_stats.json \
  --split train --camera kinect --max-candidates 8
```

只有确认 metadata 覆盖同图全部实例 mask 后才能加入 `--collision-masks-exhaustive`。否则碰撞状态保留为 `unknown`，不能把未标注区域视为空闲。只有进一步确认接触点标注穷尽了当前夹爪约束下的可行候选，才可同时加入 `--grasp-candidates-exhaustive`；此时全不安全候选才会转换成 `ungraspable`，否则转换器跳过该对象。

### 转换官方评测集

```bash
for split in seen similar novel; do
  python training/scripts/convert_realvlg_contact.py \
    --data-root /data/GraspNet_VLG \
    --output "/data/grasp_anything/contact_${split}.jsonl" \
    --split "${split}" --official-graspnet-eval --max-candidates 8
done
```

官方转换会在 `evaluation_contact_candidates_pixels` 中保留全部有效原始 GT；`--max-candidates` 只控制样本内的代表候选，不影响 evaluator 分母或最大 IoU 搜索。

训练和评测必须按 scene 划分，禁止按 JSON 行或相邻帧随机切分。

## 训练方案

当前主线冻结视觉与 LLM 基座，在 LLM attention/MLP 上训练 LoRA，
同时全参数训练视觉到语言的 MLP connector；Vision LoRA 默认关闭，但可按阶段仅适配 MoonViT 最后若干 attention 层：

| 模块 | 状态 |
|---|---|
| vision backbone | 冻结 |
| language model base | 冻结 |
| LLM attention/MLP | LoRA rank 32，alpha 64，dropout 0.05 |
| token embedding / LM head | 基础权重冻结；仅训练 `<grasp>`、`</grasp>` 的两个 `2 x hidden_size` 输入/输出 delta |
| multimodal MLP connector | 全参数训练 |
| vision LoRA | 默认关闭；`VISION_LORA_RANK>0` 时适配最后 N 层 `wqkv/wo` |

默认使用 4 张 RTX 3090/4090、BF16、SDPA、gradient checkpointing 和 ZeRO-2。LoRA target 为 LLM 的 `q/k/v/o_proj` 与 `gate/up/down_proj`，各阶段保持相同 rank 以保证 checkpoint 兼容。

### 训练 meta

参考 [`training/data/realvlg_contact_meta.example.json`](training/data/realvlg_contact_meta.example.json)。稳定阶段建议抽样比例：

```text
contact positive  70%
contact negative  10%  # 从 5% 逐步增加
grounding replay  20%
```

第一轮没有可靠负样本时使用 `80% contact positive + 20% grounding replay`，不要伪造 `none`。
这些比例只由 meta 中各数据集的显式 `sampling_weight` 控制，不存在同名环境变量。SFT 默认门禁要求至少 1000 个 contact 样本且 contact/replay 权重至少 70%/15%；negative/multigt 再要求可信负样本至少 1%。小规模 smoke 必须显式降低 `CONTACT_MIN_FULL_SAMPLES`，不能靠改文件名伪装全量数据。

### 分阶段训练

```text
Phase 1  64 样本格式过拟合
Phase 2  单候选普通 SFT
Phase 3  交换不变 pair loss
Phase 4  center/angle/width 几何辅助损失
Phase 5a  单候选 geometry + 可靠负样本
Phase 5b  保持负样本配比，开启 multi-GT
Phase 6   经审计的二维碰撞约束
```

每个阶段先通过固定验证门槛，再进入下一阶段。跨阶段使用上一阶段最佳 checkpoint 作为新的 `MODEL_PATH`，但使用全新的 optimizer/scheduler；`RESUME_FROM_CHECKPOINT` 只用于 phase、数据指纹和 worker topology 都一致的同阶段中断恢复。启动脚本会检查真实 LoRA/task-adapter 权重、trainer state 和 `phase_acceptance.json`，拒绝只修改 config 的伪 checkpoint 或 overfit64 meta 进入 `sft/pair/geometry/negative/multigt`。

### 启动

编辑 [`training/configs/grasp_anything_realvlg_contact.env`](training/configs/grasp_anything_realvlg_contact.env)，然后执行：

```bash
CONTACT_PHASE=overfit \
CONFIG_FILE=training/configs/grasp_anything_realvlg_contact.env \
bash training/scripts/train_realvlg_contact.sh
```

启动脚本会对 `training/patches/locateanything-grasp-contact.patch` 执行双向 `git apply --check`：

- 已应用：直接继续。
- 干净且兼容的 Eagle：自动应用。
- revision 或工作区不兼容：立即退出，不启动训练。

`DRY_RUN=1` 可检查最终四卡命令。正式训练前必须完成 64 样本过拟合，确认格式有效率接近 100%、坐标损失下降且没有 `none` 塌缩。

## 二维评测

```bash
python training/scripts/evaluate_realvlg_contact.py \
  --annotations /data/grasp_anything/contact_seen_grasp_v2.jsonl \
  --data-root /data/GraspNet_VLG \
  --model-path /models/grasp-anything-checkpoint \
  --output /tmp/grasp_predictions.jsonl \
  --metrics /tmp/grasp_metrics.json
```

模型选择使用：

- `gacc_corrected_strict`：IoU 大于 0.25、模 `pi` 角度误差小于 30 度，非法输出按 0。
- `miou_strict`：非法正样本按 0。
- `format_valid_rate` 和 `positive_grasp_output_rate`。
- `collision_aware_gacc_strict`、`collision_rate_2d` 和 `mean_outside_ratio_2d`。
- 负样本 `none precision/recall/F1`。

`gacc_official_buggy_valid` 只用于复现 RealVLG 旧日志中的角度单位错误，不是合法基线，也不能用于选择 checkpoint。

## 本地推理与评估

本地推理使用 Python 3.10-3.12。不要使用 Python 3.13：当前 Eagle/Transformers 依赖链仍可能导入已从标准库移除的 `cgi`，会在模型加载前失败。

### 1. 创建本地推理环境

bootstrap 默认安装 CUDA 12.1 的固定 PyTorch 版本；其他 CUDA 版本可通过 `TORCH_INDEX_URL` 覆盖。

```bash
bash scripts/bootstrap.sh --dev --venv .eval-venv
source .eval-venv/bin/activate
```

检查依赖、checkpoint 和 CUDA：

```bash
set -a; source .env; set +a
grasp-anything doctor
```

所有检查必须显示 `[OK]`。CPU 可以执行格式解析和已有预测文件评估，但不适合加载 3B 模型进行逐图生成。

如果出现 `ModuleNotFoundError: No module named 'cgi'`，说明命令没有使用 `.eval-venv` 或仍在使用 Python 3.13：

```bash
source .eval-venv/bin/activate
which python
python --version
```

### 2. 准备 checkpoint

`--model-path` 必须指向完整模型目录，而不是单个 `.safetensors` 文件。目录至少应包含：

```text
config.json
model.safetensors.index.json 或 model.safetensors
tokenizer_config.json
preprocessor_config.json
```

抓取 checkpoint 还必须在 `config.json` 中保存两个不同的任务 token ID：

```bash
python - <<'PY' /path/to/checkpoint
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads((path / "config.json").read_text())
ids = config.get("grasp_task_token_ids")
assert isinstance(ids, list) and len(ids) == 2 and ids[0] != ids[1], ids
print("grasp token ids:", ids)
PY
```

不要只把 `box` token ID 手工改成 `grasp`。如果 checkpoint 没有 `grasp_task_token_ids`，应重新从已修复的训练 checkpoint 导出；否则 Fast 解码可能回退为 `<box>...</box>`，即使坐标看起来合理，严格抓取格式率也会被记为 0%。

### 3. 单张图片推理

```bash
source .eval-venv/bin/activate
export GRASP_ANYTHING_MODEL=/absolute/path/to/checkpoint

LOCATE_MODEL_ID="$GRASP_ANYTHING_MODEL" \
LOCATE_MODEL_REVISION="" \
LOCATE_DEVICE=cuda \
grasp-anything predict \
  /absolute/path/to/image.jpg \
  "抓取图中左侧的红色杯子" \
  --mode grasp_contact \
  --generation-mode fast \
  --annotated-output /tmp/grasp_result.png
```

CLI 没有 `--model-path` 参数；模型路径通过 `LOCATE_MODEL_ID` 传入。也可以写入本地 `.env`：

```bash
LOCATE_MODEL_ID=/absolute/path/to/checkpoint
LOCATE_MODEL_REVISION=
LOCATE_DEVICE=cuda
LOCATE_REQUIRE_GRASP_CHECKPOINT=1
```

CLI 不会自动读取任意目录下的 `.env`；执行前请 `set -a; source .env; set +a`，或直接使用上一条命令中的环境变量。然后执行：

```bash
grasp-anything predict image.jpg "抓取红色杯子" \
  --mode grasp_contact --generation-mode fast \
  --annotated-output /tmp/grasp_result.png
```

推荐先用 `fast`。它使用 PBD 块内四坐标联合解码；`slow` 用于排查 Fast 解码或格式问题；`hybrid` 在 Fast 失败时回退 Slow。抓取模式服务端固定使用 greedy 解码，不应打开随机采样。

### 4. 本地批量评估

评估脚本要求 annotation JSONL 和原始图片根目录。先只跑少量样本验证环境：

```bash
source .eval-venv/bin/activate
python training/scripts/evaluate_realvlg_contact.py \
  --annotations /absolute/path/to/contact_seen.jsonl \
  --data-root /absolute/path/to/GraspNet_VLG \
  --model-path /absolute/path/to/checkpoint \
  --output /tmp/seen.predictions.jsonl \
  --metrics /tmp/seen.metrics.json \
  --generation-mode fast \
  --limit 20
```

确认 20 条样本能完成后再跑完整 split：

```bash
for split in seen similar novel; do
  python training/scripts/evaluate_realvlg_contact.py \
    --annotations "/absolute/path/to/contact_${split}.jsonl" \
    --data-root /absolute/path/to/GraspNet_VLG \
    --model-path /absolute/path/to/checkpoint \
    --output "/tmp/${split}.predictions.jsonl" \
    --metrics "/tmp/${split}.metrics.json" \
    --generation-mode fast
done
```

主要查看：

```bash
python - <<'PY'
import json
from pathlib import Path
for path in map(Path, ("/tmp/seen.metrics.json", "/tmp/similar.metrics.json", "/tmp/novel.metrics.json")):
    if path.exists():
        m = json.loads(path.read_text())
        print(path.name, {
            "format": m.get("format_valid_rate"),
            "gacc": m.get("gacc_corrected_strict"),
            "miou": m.get("miou_strict"),
            "endpoint_px": m.get("swap_invariant_endpoint_error_pixels"),
            "angle_deg": m.get("swap_invariant_angle_error_degrees"),
            "collision_valid": m.get("collision_valid_samples"),
        })
PY
```

`gacc_corrected_strict` 是当前主指标：IoU 大于 0.25 且模 `pi` 角度误差小于 30 度；非法输出按失败计入分母。`collision_valid_samples=0` 时只能解释为二维几何结果，不能声称碰撞安全。

### 4.1 Pair 到 6000 步自动评估

如果 Pair 训练在远端后台运行，可以提前启动等待脚本。它会等待 `pair/checkpoint-6000` 完整落盘，然后向训练的 `torchrun` PID 发送一次 `SIGINT`，等待进程退出后再评估，不会与训练抢显存：

```bash
TRAIN_PID="${TRAIN_PID}" \
PAIR_OUTPUT_DIR=/srv/outputs/contact-run \
EVAL_DATA_ROOT=/srv/data/GraspNet_VLG \
EVAL_ANNOTATIONS_SEEN=/srv/data/contact_seen.jsonl \
EVAL_ANNOTATIONS_SIMILAR=/srv/data/contact_similar.jsonl \
EVAL_ANNOTATIONS_NOVEL=/srv/data/contact_novel.jsonl \
PYTHON_BIN="$PWD/.venv/bin/python" \
bash training/scripts/evaluate_pair_at_step.sh
```

脚本完成后会在 checkpoint 中写入 `phase_acceptance.json`，该文件是进入 Geometry 阶段的必要条件。不要在四张训练 GPU 上并发加载 3B 模型；脚本先停止训练再评估。

### 5. 使用已有预测文件离线评估

如果本地没有 GPU，只要已有远端生成的预测 JSONL，仍可计算指标，不加载模型：

```bash
python training/scripts/evaluate_realvlg_contact.py \
  --annotations /absolute/path/to/contact_seen.jsonl \
  --data-root /absolute/path/to/GraspNet_VLG \
  --predictions /absolute/path/to/seen.predictions.jsonl \
  --output /tmp/seen.rechecked.predictions.jsonl \
  --metrics /tmp/seen.rechecked.metrics.json
```

这条路径适合本地复核远端结果；它不会重新生成坐标，也不会改变原预测文件。

### 6. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `cgi` 导入失败 | 使用 Python 3.13 或错误环境 | 激活 `.eval-venv`，确认 Python 3.10/3.11 |
| `CUDA out of memory` | 同时运行多个模型或显存不足 | 关闭其他进程，单卡运行；不要在评估脚本中并行启动多个模型 |
| 输出 `<box>...</box>` | checkpoint 缺少抓取 token 配置 | 换用包含 `grasp_task_token_ids` 的新 checkpoint |
| `ImageNotFound` 或路径错误 | annotation 中的相对图片路径与 `--data-root` 不匹配 | 检查 JSONL 的 `image` 字段和根目录 |
| `format_valid_rate=0` | 使用普通 grounding prompt 或旧服务配置 | 指定 `--mode grasp_contact`，并使用抓取 checkpoint |
| 指标与远端不一致 | 数据 split、厚度或 checkpoint 不同 | 固定同一 annotation、`--rectangle-thickness 80` 和同一 checkpoint |

本地推理与远端推理应使用同一 commit、同一 checkpoint、同一 tokenizer 和同一 `generation_mode`。先用 `--limit 20` 做 smoke，再进行完整评估，避免把环境问题误判成模型效果变化。

## 推理服务

### 安装

```bash
bash scripts/bootstrap.sh --service
set -a; source .env; set +a
grasp-anything doctor
```

### 启动 API

```bash
grasp-anything serve
# http://127.0.0.1:8000/docs
```

单图抓取：

```bash
grasp-anything predict image.jpg "the leftmost red cup" \
  --mode grasp_contact --generation-mode fast \
  --annotated-output grasp.png
```

HTTP API：

```bash
curl -X POST http://127.0.0.1:8000/v1/locate \
  -F 'image=@image.jpg' \
  -F 'query=the leftmost red cup' \
  -F 'mode=grasp_contact' \
  -F 'generation_mode=fast' \
  -F 'annotate=true'
```

`grasp_contact` 在服务端强制 greedy：`do_sample=False`、`temperature=0`、`top_p=None`，同图同指令不会因默认采样产生漂移。

### Web UI

```bash
./launch/grasp-anything-ui.sh
# UI:  http://127.0.0.1:8001
# API: http://127.0.0.1:8000
```

### Docker

```bash
docker compose build
docker compose up -d
docker compose logs -f grasp-anything
```

### systemd

```bash
make service-install
make service-status
journalctl --user -u grasp-anything.service -f
```

## 目录结构

```text
grasp_anything/
├── src/locate_anything_service/   # 上游兼容的 API/CLI/解析/二维几何
├── training/
│   ├── Eagle/                     # 上游 Eagle checkout，gitignored
│   ├── configs/                   # 四卡训练配置
│   ├── data/                      # meta 示例
│   ├── patches/                   # contact 与兼容补丁
│   └── scripts/                   # 转换、训练和严格评测
├── scripts/                       # Web UI、模型管理、下载工具
├── tests/                         # CPU 回归测试
├── DEPLOYMENT.md                  # 跨机器安装、模型挂载与发布检查
├── GRASP_CONTACT_README.md        # 完整设计、收敛和风险文档
├── Dockerfile
├── compose.yaml
└── pyproject.toml
```

Python 内部模块暂时保留 `locate_anything_service`，因为 checkpoint 的 remote code、已有部署和导入路径依赖它。发行包和主 CLI 已改为 `grasp-anything`；旧 `locate-anything` CLI 仅作为兼容别名。

## 验证

```bash
make lint
make test
for script in scripts/bootstrap.sh training/scripts/*.sh; do bash -n "$script"; done
bash training/scripts/bootstrap_eagle.sh --no-clone
```

详细损失、DDP reduction、碰撞协议、风险和消融计划见 [`GRASP_CONTACT_README.md`](GRASP_CONTACT_README.md)。`training/README.md` 和 `report/` 中的 VOC 数据与旧超参数属于历史实验，不是当前抓取训练主线。
