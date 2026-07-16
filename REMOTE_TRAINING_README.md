# Grasp Anything 2D 远端训练交接说明

更新日期：2026-07-16

本文档记录远端服务器的当前状态、数据与代码路径，以及训练准备和分阶段训练步骤。当前没有启动训练。

## 1. 当前状态

| 项目 | 状态 |
| --- | --- |
| 远端工程 | `/home/zhenghengcao/grasp_anything` |
| 工程提交 | `4f53073ac6285eaf1662d2fa7869c13f5ddbb24f` |
| RealVLG 数据压缩包 | `/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG.zip` |
| RealVLG 解压目录 | `/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG` |
| RealVLG 官方 metadata | `/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG/metadata` |
| LocateAnything-3B 模型 | `/home/zhenghengcao/LocateAnything-3B` |
| Python | `/home/zhenghengcao/la_env/bin/python` |
| Eagle 源码 | `/home/zhenghengcao/grasp_anything/training/Eagle` |
| Eagle 提交 | `783f656` |
| 接触点训练补丁 | 已应用到工程内 Eagle |
| GPU | 4 x RTX 4090，支持 BF16 |
| TensorBoard | 已安装于 `la_env` |
| 正式训练 JSONL/meta | `contact_train_grasp_v2.jsonl` / `full_contact_meta.json` |
| 官方评测 JSONL | `contact_{seen,similar,novel}_grasp_v2.jsonl` |
| grounding replay JSONL | 尚未找到，正式稳定 SFT 前必须补齐 |
| 训练任务 | 尚未启动 |

数据压缩包已经解压，正式 contact v2 数据、统计文件、contact-only full meta 和三个官方评测 split 已生成并通过格式/安全性校验。grounding replay 仍未找到，因此长期稳定 SFT 前必须补齐 replay。

最近一次检查时，4 张 GPU 正由用户 `yihanzhu` 的 Qwen3.5 bbox 评测占用，对应 PID 为 `4190905`、`4190906`、`4190907`、`4190908`，每卡约占用 5570 MiB。PID 可能已经变化，开跑前必须重新查看；不得终止其他用户的任务。

## 2. 本项目实际训练形式

当前脚本使用 LLM LoRA 微调：

- vision backbone 冻结；
- LLM 基座冻结；
- LLM attention/MLP 使用 rank 32 LoRA；
- multimodal MLP 全参数训练；
- backbone LoRA 关闭；
- 基础 embedding/LM head 冻结，只训练 `<grasp>`、`</grasp>` 的两个 `2 x hidden_size` 输入/输出 delta；
- 4 卡 BF16、SDPA、gradient checkpointing、DeepSpeed ZeRO-2。

训练目标是一次生成一个 PBD 四坐标块：

```text
(x1, y1, x2, y2)
```

它表示两个夹爪接触点。抓取中心、二维夹爪方向和开口宽度均由这两个点确定性计算，不再分别调用两次 point 模式，也不直接生成角度和宽度。

## 3. 不要做的事情

1. 不要下载、转换或训练 VOC；二维接触抓取训练不依赖 VOC。
2. 不要启用 vision/backbone LoRA；当前只在 LLM attention/MLP 上启用 LoRA。
3. 不要跳过 64 样本过拟合阶段直接跑全量数据。
4. 不要伪造 `none`、不可抓取或无目标负样本。没有可信负样本时，先保持负样本权重为 0。
5. 不要在 mask 完整性未审计前给转换器加 `--collision-masks-exhaustive`。该选项会把“没有找到 mask”解释成“mask 完整且无碰撞”，可能制造错误监督。
6. 不要删除碰撞相关代码和字段。二维碰撞率、越界率和后续碰撞约束仍需要保留，只是必须在监督基线稳定后再逐步启用。
7. 不要终止其他用户的 GPU 进程。
8. 不要在缺少 grounding replay 时把 Phase 2 及以后称为正式稳定训练。contact-only 全量训练可能破坏 LocateAnything 原有定位能力。
9. 不要把前一阶段 optimizer/scheduler 状态带到新阶段；跨阶段加载模型 checkpoint，但重新创建优化器。只有同一阶段意外中断时才使用 `RESUME_FROM_CHECKPOINT`。

## 4. 固定路径

后续命令统一使用以下路径：

```bash
PROJECT=/home/zhenghengcao/grasp_anything
PY=/home/zhenghengcao/la_env/bin/python
MODEL=/home/zhenghengcao/LocateAnything-3B
DATA_ROOT=/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG
PREP=/data2/zhenghengcao/grasp_anything_2d/prepared
OUTPUT=/data2/zhenghengcao/grasp_anything_2d/outputs
LOG_ROOT=/data2/zhenghengcao/grasp_anything_2d/logs
```

## 5. 上层 agent 的准备任务

以下步骤尚未执行，交由下一位 agent 完成。

### 5.1 检查 GPU 和环境

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

必须等到可用显卡足够且不会影响他人任务后再启动 4 卡训练。

安装训练脚本需要的 TensorBoard：

```bash
/home/zhenghengcao/la_env/bin/python -m pip install tensorboard
```

创建数据输出目录：

```bash
mkdir -p \
  /data2/zhenghengcao/grasp_anything_2d/prepared \
  /data2/zhenghengcao/grasp_anything_2d/outputs \
  /data2/zhenghengcao/grasp_anything_2d/logs
```

### 5.2 生成正式 contact 训练 JSONL

使用官方 `metadata`，不要使用 `metadata_filter`：

```bash
cd /home/zhenghengcao/grasp_anything

/home/zhenghengcao/la_env/bin/python \
  training/scripts/convert_realvlg_contact.py \
  --data-root /data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG \
  --metadata-dir /data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG/metadata \
  --output /data2/zhenghengcao/grasp_anything_2d/prepared/contact_train_grasp_v2.jsonl \
  --stats /data2/zhenghengcao/grasp_anything_2d/prepared/contact_train_grasp_v2_stats.json \
  --split train \
  --camera kinect \
  --max-candidates 8
```

这里故意不传 `--collision-masks-exhaustive`。转换完成后由上层 agent 负责检查统计、路径可读性、坐标范围、候选数量和样本可视化。

只有同时完成人工审计并能证明同图实例 mask 与抓取候选集合都穷尽时，才允许同时传入 `--collision-masks-exhaustive --grasp-candidates-exhaustive`。仅有前一个标志时，全部已标注候选都不安全的对象会被跳过，不会被错误转换成 `none`。

评测数据应分别生成 `seen`、`similar`、`novel` 三个 split，并保留官方 GraspNet 评测标记。具体参数以脚本 `--help` 为准，目标文件建议为：

官方评测 JSONL 的完整 GT 位于 `evaluation_contact_candidates_pixels`。即使设置 `--max-candidates 8`，该字段也不会被压缩；越界比例只用于训练候选安全筛选，不会删除官方评测正样本。

```text
/data2/zhenghengcao/grasp_anything_2d/prepared/contact_seen_grasp_v2.jsonl
/data2/zhenghengcao/grasp_anything_2d/prepared/contact_similar_grasp_v2.jsonl
/data2/zhenghengcao/grasp_anything_2d/prepared/contact_novel_grasp_v2.jsonl
```

### 5.3 建立 64 样本过拟合集

```bash
head -n 64 \
  /data2/zhenghengcao/grasp_anything_2d/prepared/contact_train_grasp_v2.jsonl \
  > /data2/zhenghengcao/grasp_anything_2d/prepared/contact_overfit64_grasp_v2.jsonl
```

创建 `/data2/zhenghengcao/grasp_anything_2d/prepared/overfit64_grasp_v2_meta.json`。字段名必须以仓库中 `training/examples/realvlg_contact_meta.json` 和 `training/scripts/validate_training_meta.py` 接受的实际格式为准，核心语义如下：

```json
{
  "contact_overfit64": {
    "root": "/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG",
    "annotation": "/data2/zhenghengcao/grasp_anything_2d/prepared/contact_overfit64_grasp_v2.jsonl",
    "task_type": "grasp_contact",
    "sampling_weight": 1.0,
    "max_contact_candidates": 1,
    "data_augment": false
  }
}
```

不要未经脚本校验直接采用上述示意 JSON；上层 agent 应对照示例文件确认 Eagle 当前版本的字段结构。

### 5.4 准备正式 SFT meta

正式稳定 SFT 推荐采样比例：

仓库内 `training/data/full_contact_meta.remote.json` 是可校验的全量 contact-only 基线，可安装为 `prepared/full_contact_meta.json`；它解决全量 contact 数据入口，但不等价于带 replay 的长期 SFT 配置。找到原始 grounding 数据后再按下表加入同一 meta。

| 数据源 | sampling weight |
| --- | ---: |
| RealVLG contact | 0.80 |
| LocateAnything grounding replay | 0.20 |
| 可信负样本 | 当前 0.00 |

当前远端尚未找到 grounding replay JSONL。上层 agent 应先定位或生成与当前 Eagle 数据桥兼容的 replay 数据，再创建正式 meta。没有 replay 时，可以完成 Phase 1 的 64 样本 contact-only 过拟合验证，但不要直接进入长时间的正式全量 SFT。

## 6. 远端训练配置模板

建议新建远端专用配置：

```text
/home/zhenghengcao/grasp_anything/training/configs/grasp_anything_realvlg_contact.remote.env
```

Phase 1 推荐从以下保守配置开始：

```bash
MODEL_PATH=/home/zhenghengcao/LocateAnything-3B
EAGLE_ROOT=/home/zhenghengcao/grasp_anything/training/Eagle
PYTHON_BIN=/home/zhenghengcao/la_env/bin/python
REALVLG_ROOT=/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG
REALVLG_TRAIN_JSONL=/data2/zhenghengcao/grasp_anything_2d/prepared/contact_train_grasp_v2.jsonl
META_PATH=/data2/zhenghengcao/grasp_anything_2d/prepared/overfit64_grasp_v2_meta.json
REALVLG_OUTPUT_DIR=/data2/zhenghengcao/grasp_anything_2d/outputs/realvlg-contact-lora

NPROC_PER_NODE=4
CUDA_VISIBLE_DEVICES=0,1,2,3
CONTACT_PHASE=overfit
MAX_SEQ_LENGTH=2048
GRADIENT_ACCUMULATION_STEPS=4
MAX_STEPS=300
LEARNING_RATE=5e-6
LLM_LORA_RANK=32
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0
LOGGING_STEPS=10
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2

CONTACT_MAX_CANDIDATES=8
CONTACT_RECTANGLE_THICKNESS=80
CONTACT_COLLISION_THRESHOLD=0.0
CONTACT_OUTSIDE_THRESHOLD=0.0
CONTACT_PAIR_WEIGHT=1.0
CONTACT_CENTER_WEIGHT=0.25
CONTACT_ANGLE_WEIGHT=0.10
CONTACT_WIDTH_WEIGHT=0.10
CONTACT_GEOMETRY_START_BLOCKS=0
CONTACT_GEOMETRY_RAMP_BLOCKS=20000
CONTACT_COORD_MASS_THRESHOLD=0.35
CONTACT_COORD_ENTROPY_THRESHOLD=0.85

CONTACT_MIN_CONTACT_SAMPLES=1
CONTACT_MIN_FULL_SAMPLES=1000
GROUNDING_MIN_REPLAY_SAMPLES=1
CONTACT_MIN_NEGATIVE_SAMPLES=1
CONTACT_MIN_POSITIVE_FRACTION=0.70
GROUNDING_MIN_REPLAY_FRACTION=0.15
CONTACT_MIN_NEGATIVE_FRACTION=0.01
```

配置文件中的 `META_PATH` 决定实际读入的数据，混合比例只由 meta 内每个数据集的 `sampling_weight` 决定。上述变量是启动门禁，不会替代权重。Phase 1 使用 `overfit64_grasp_v2_meta.json`；Phase 2 以后必须改成已加入 grounding replay 的正式 meta，multigt 还必须加入可信负样本。

## 7. 分阶段训练

训练入口：

```text
/home/zhenghengcao/grasp_anything/training/scripts/train_realvlg_contact.sh
```

该脚本会检查/应用 contact patch、验证 meta，并按 `CONTACT_PHASE` 组装训练参数。

| 阶段 | `CONTACT_PHASE` | 候选数 | 启用目标 | 目的 |
| --- | --- | ---: | --- | --- |
| Phase 1 | `overfit` | 1 | token CE | 64 样本过拟合，先证明输出格式与数据桥正确 |
| Phase 2 | `sft` | 1 | token CE | contact + grounding replay 的稳定全量 SFT |
| Phase 3 | `pair` | 1 | CE + 交换不变 pair loss | 消除两个接触点顺序影响 |
| Phase 4 | `geometry` | 1 | pair + center/angle/width | 逐步加入二维几何辅助目标 |
| Phase 5a | `negative` | 1 | geometry + 可信负样本 | 先单独学习 `none`，不同时改变候选集合 |
| Phase 5b | `multigt` | 最多 8 | Phase 5a + 多 GT | 只新增多候选 hard-min |
| Phase 6 | 后续配置 | 最多 8 | 碰撞/越界约束 | 仅在碰撞标注和监督基线可靠后启用 |

### 7.1 只打印命令

任何阶段先运行 dry-run：

```bash
cd /home/zhenghengcao/grasp_anything

DRY_RUN=1 \
CONTACT_PHASE=overfit \
CONFIG_FILE=/home/zhenghengcao/grasp_anything/training/configs/grasp_anything_realvlg_contact.remote.env \
bash training/scripts/train_realvlg_contact.sh
```

### 7.2 启动 Phase 1

确认 GPU 空闲、meta 校验通过且 dry-run 命令正确后，才执行：

```bash
cd /home/zhenghengcao/grasp_anything

CONTACT_PHASE=overfit \
CONFIG_FILE=/home/zhenghengcao/grasp_anything/training/configs/grasp_anything_realvlg_contact.remote.env \
bash training/scripts/train_realvlg_contact.sh \
  2>&1 | tee /data2/zhenghengcao/grasp_anything_2d/logs/phase1-overfit64.log
```

训练脚本使用 TensorBoard：

```bash
tensorboard \
  --logdir /data2/zhenghengcao/grasp_anything_2d/outputs/realvlg-contact-lora \
  --host 0.0.0.0
```

### 7.3 阶段切换规则

- 同一阶段意外中断：只有 checkpoint 的 `training_phase`、v6 dataloader 数据指纹、worker topology 和当前配置完全一致时，才使用 `RESUME_FROM_CHECKPOINT=/path/to/checkpoint` 恢复 optimizer、scheduler 和数据流；否则改用 `MODEL_PATH` 启动新阶段。
- 从一个阶段进入下一阶段：把 `MODEL_PATH` 指向上一阶段选出的最佳 checkpoint，不设置 `RESUME_FROM_CHECKPOINT`，使用新的 optimizer 和 scheduler。
- 每次阶段切换都先 dry-run，并保留上一阶段 checkpoint 和评测结果。
- Phase 3/4/5 不应同时首次启用多个新目标。一次只增加一个变化，异常时才能定位原因。

跨阶段前由验收 agent 在选定 checkpoint 写入 `phase_acceptance.json`：

```json
{
  "phase": "overfit",
  "accepted": true,
  "checkpoint_step": 300,
  "metrics": {
    "format_valid_rate": 1.0,
    "coordinate_top1_accuracy": 0.96
  }
}
```

`checkpoint_step` 必须等于 `trainer_state.json`。进入 SFT 时两个指标分别不得低于 0.99 和 0.95；后续阶段的 `phase` 必须依次为 `sft`、`pair`、`geometry`、`negative`。只有 config、没有真实 LoRA/task-adapter 权重或没有验收文件的目录会被启动脚本拒绝。

## 8. 防止不收敛的验收门槛

以下检查由上层 agent 执行。本轮不执行这些校验。

### Phase 1 必须满足

1. 64 样本训练损失能够持续下降，并能明显过拟合。
2. 可解析为恰好一个四坐标原子块的比例至少 99%。
3. 四个坐标 token 的 teacher-forced accuracy 至少 95%。
4. 端点误差应接近 PBD 坐标量化误差，不应持续大幅偏离。
5. 输出不得塌缩到固定中心、固定宽度，或大量集中在 `0/500/1000`。
6. 两个端点交换后，抓取中心、无向角度和宽度必须保持一致。

任何一项失败，都不要扩大数据量，也不要启用 pair/geometry/multi-GT loss。先检查：

- contact mask 是否与 shifted label 对齐；
- 图像 resize 后的坐标是否使用相同变换；
- PBD 四坐标 block 是否连续且只出现一次；
- padding、packing 和候选维度是否错位；
- 学习率是否过高；
- 是否错误混入不可解析的样本。

### Phase 2 及以后必须监控

- 总 loss、token CE、pair loss、center/angle/width loss 分项；
- 有效 contact block 数和被过滤 block 数；
- 坐标概率质量和熵过滤比例；
- parse rate、端点误差、中心误差、无向角误差、宽度误差；
- seen/similar/novel 二维抓取指标；
- grounding replay 指标，防止原能力灾难性遗忘；
- 越界率和二维碰撞率，但不要在碰撞数据未审计前把未知值当负样本。

出现 NaN/Inf、梯度范数持续打满、坐标熵突然塌缩、parse rate 快速下降或 grounding replay 明显恶化时，应立即停止当前阶段并保存日志，不要靠增加 loss 权重硬压。

## 9. 碰撞处理要求

碰撞检测仍然是项目要求的一部分：

- 保留转换阶段的夹爪矩形、图像边界和 outside ratio 计算；
- 有完整物体 mask 时，计算夹爪闭合区域与非目标物体的二维相交比例；
- mask 不完整时，碰撞状态必须保持 unknown，不得自动标成 collision-free；
- 首先用监督接触点把生成任务训稳，再在独立阶段用较小权重逐步加入碰撞约束；
- 二维评测只关注 RGB 平面的接触、中心、无向角、宽度、越界和碰撞指标，不引入三维姿态指标。

## 10. Codex 状态

远端已安装：

```text
/home/zhenghengcao/.local/bin/codex
codex-cli 0.144.4
```

远端配置文件：

```text
/home/zhenghengcao/.codex/config.toml
```

配置已经能访问所设 provider，但当前没有远端认证，探测返回 `401 API_KEY_REQUIRED`。没有复制本机 API key 或任何密钥。上层 agent 如需在远端运行 Codex，必须让用户采用其认可的方式完成登录/密钥配置，不得自行复制或输出现有密钥。

## 11. 上层 agent 交接清单

- [ ] 重新确认 4 张 GPU 是否空闲，不杀其他用户进程。
- [ ] 在 `la_env` 安装 TensorBoard。
- [x] 生成正式 `contact_train_grasp_v2.jsonl` 和统计文件。
- [ ] 检查数据统计、图片路径、坐标范围和候选数量。
- [ ] 按需要抽样可视化 contact 与夹爪矩形。
- [x] 生成 seen/similar/novel 评测 JSONL。
- [x] 生成 `contact_overfit64_grasp_v2.jsonl` 与合法 Eagle meta。
- [x] 创建远端训练 env 文件，并先执行 `DRY_RUN=1`。
- [ ] 只启动 Phase 1，达到验收门槛后再进入下一阶段。
- [ ] 定位或生成 grounding replay；缺失时不要启动正式全量 Phase 2。
- [ ] 每次只增加一个训练目标，并使用上一阶段最佳模型启动新阶段。
- [ ] 在监督基线稳定和 collision mask 完整性确认后，再启用碰撞约束。
- [ ] 如需远端 Codex，先由用户完成认证授权。

## 12. 代码参考

```text
training/GRASP_CONTACT_README.md
training/scripts/convert_realvlg_contact.py
training/scripts/validate_training_meta.py
training/scripts/train_realvlg_contact.sh
training/scripts/evaluate_realvlg_contact.py
training/configs/grasp_anything_realvlg_contact.env
training/examples/realvlg_contact_meta.json
training/Eagle/Embodied/eaglevl/train/grasp_contact.py
src/locate_anything_service/collision_2d.py
```

本交接点的明确状态是：代码、原始数据、contact v2 全量 JSONL、overfit meta、contact-only full meta 和评测 split 已在远端；正式 Phase 2 仍缺 grounding replay，multigt 仍缺可信负样本。训练进程当前已停止。修复前 checkpoint 的 dataloader state 没有 v6 数据指纹，只能通过 `MODEL_PATH` 继承权重并启动新 optimizer/数据流，不能用 `RESUME_FROM_CHECKPOINT` 冒充精确续训，也不能把旧 `global_step` 解释为完整遍历数据集的 epoch 数。
