# grasp_anything

`grasp_anything` 是一个基于 NVIDIA `LocateAnything-3B` 的语言引导二维抓取项目。输入 RGB 图像和自然语言目标描述，模型通过一次 PBD 联合生成两个平行夹爪接触点：

```text
<ref>grasp</ref><grasp><x1><y1><x2><y2></grasp>
```

模型不直接预测存在周期歧义的角度和宽度。二维中心、接触点闭合轴方向（模 `pi`）和像素开口全部由两个接触点确定性计算；二维实例 mask 仅用于投影碰撞率、越界率和 clearance 代理评估。

上游模型、remote-code 类名和 `LOCATE_*` 环境变量保留 LocateAnything 命名。本项目的发行名、CLI、API/UI、Docker 和 systemd 服务统一使用 `grasp_anything` / `grasp-anything`。

## 当前状态（这是一个未完成的项目）

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
- 全新 Eagle checkout 启动训练时自动校验并应用 contact patch。
- Phase 2 起强制检查 grounding replay，negative/multigt 阶段额外检查可信负样本。

仍需在训练服务器上完成的数据验收包括：RealVLG 全量审计、200 张可视化抽检、64 样本过拟合、四卡保存/恢复和 seen/similar/novel 正式评测。

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
同时全参数训练视觉到语言的 MLP connector：

| 模块 | 状态 |
|---|---|
| vision backbone | 冻结 |
| language model base | 冻结 |
| LLM attention/MLP | LoRA rank 32，alpha 64，dropout 0.05 |
| token embedding / LM head | 基础权重冻结；仅训练 `<grasp>`、`</grasp>` 的两个 `2 x hidden_size` 输入/输出 delta |
| multimodal MLP connector | 全参数训练 |
| vision LoRA | 关闭 |

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

## 推理服务

### 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
cp .env.example .env
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
bash -n training/scripts/train_realvlg_contact.sh
git -C training/Eagle apply --reverse --check \
  ../patches/locateanything-grasp-contact.patch
```

详细损失、DDP reduction、碰撞协议、风险和消融计划见 [`GRASP_CONTACT_README.md`](GRASP_CONTACT_README.md)。`training/README.md` 和 `report/` 中的 VOC 数据与旧超参数属于历史实验，不是当前抓取训练主线。
