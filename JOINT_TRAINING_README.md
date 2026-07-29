# Contact、Grasp Rect 与 BBox 联合训练

本文档描述当前实现的联合训练入口。联合模式使用一个共享 checkpoint，同时训练：

- `grounding`：原有 `<box>` PBD 输出，作为 bbox/定位 replay；
- `grasp_contact`：`<grasp><x1><y1><x2><y2></grasp>`；
- `grasp_rect`：`<grasp_rect><cx><cy><theta><width></grasp_rect>`。

这里的“同一个权重”指视觉编码器、MLP connector、LLM LoRA 和坐标词表共享。三种任务仍使用不同结构 token 和不同几何 loss，不能把四个槽位直接混成一个语义。

## 当前实现状态

联合代码已经接入，入口是：

```bash
bash training/scripts/train_realvlg_joint.sh
```

已完成：

- `joint_task_enabled` 参数；
- Contact 与 Grasp Rect 两套 task-token input/output delta 同时解冻；
- 同一 `forward()` 中同时计算 Contact 与 Grasp Rect auxiliary loss；
- grounding/bbox 保持原有 base PBD CE；
- 联合数据 `sampling_weight`；
- `joint_trainer_state.json`，同时记录两个任务的累计 block 数；
- `JOINT_PHASE` 的阶段门禁；
- 联合 phase acceptance，要求 Contact、Grasp Rect 和 grounding replay 都通过门槛；
- 联合 adapter、checkpoint 和 phase-transition CPU 测试。

尚未在本机完成：

- 真实 RealVLG 全量联合数据训练；
- 4 卡 BF16/DeepSpeed 实跑；
- 联合 checkpoint 的三路真实 evaluator 对账。

## 模型结构

```text
image + language + task prompt
              |
           MoonViT
              |
       shared MLP connector
              |
        shared Qwen LoRA
       /          |          \
   <box>       <grasp>    <grasp_rect>
   bbox       contact      rect pose
```

基础 embedding 和完整 `lm_head` 仍然冻结。每个抓取任务只拥有两个小型 delta：

```text
grasp_task_embedding_delta
grasp_task_output_delta
grasp_rect_task_embedding_delta
grasp_rect_task_output_delta
```

Grounding 不新增任务 adapter，直接使用共享模型的 `<box>` 结构。

## 数据 Meta

复制并修改：

```text
training/data/realvlg_joint_meta.example.json
```

建议第一版比例：

```text
contact positive       0.40
grasp rect positive    0.40
grounding replay       0.20
```

每个数据集必须声明正确的 `task_type` 和 `sampling_weight`：

```json
{
  "contact": {
    "root": "/data/GraspNet_VLG",
    "annotation": "/data/contact_train.jsonl",
    "task_type": "grasp_contact",
    "sampling_weight": 0.4,
    "max_contact_candidates": 1
  },
  "grasp_rect": {
    "root": "/data/GraspNet_VLG",
    "annotation": "/data/grasp_rect_train.jsonl",
    "task_type": "grasp_rect",
    "sampling_weight": 0.4,
    "max_grasp_rect_candidates": 1
  },
  "grounding": {
    "root": "/data/locateanything-data",
    "annotation": "/data/grounding_replay.jsonl",
    "task_type": "grounding",
    "sampling_weight": 0.2
  }
}
```

`grasp_contact` 和 `grasp_rect` 可以来自同一图像，但第一版不要求三种标注逐行配对。联合训练是 task-conditioned multi-task training；只有在构造同一对象的三联标注后，才能加入跨任务几何一致性 loss。

## 联合 Loss

总 loss 为：

```text
L = L_base_PBD
  + lambda_contact * L_pair
  + lambda_rect * L_pose
  + s_contact * L_contact_geometry
  + s_rect * L_rect_geometry
```

其中：

- `L_base_PBD` 包含文本、`<ref>`、`<box>` 和结构 token CE；
- Contact 四个坐标从 base CE 中剔除后使用交换不变 pair CE；
- Grasp Rect 四个参数从 base CE 中剔除后使用 pose CE；
- `L_contact_geometry` 是中心、`pi` 周期角度和宽度 loss；
- `L_rect_geometry` 是中心、double-angle 周期角度和宽度 loss；
- grounding 与两个抓取任务的样本比例由 `sampling_weight` 控制。

## 阶段流程

| 阶段 | Contact loss | Rect loss | 候选数 | 目的 |
|---|---:|---:|---:|---|
| `overfit` | off | off | 1 | 三任务格式和坐标 CE 过拟合 |
| `sft` | off | off | 1 | 全量单候选 SFT + grounding replay |
| `structured_r0` | pair | pose | 1 | 开启 auxiliary loss，Rect 角度 radius=0 |
| `structured` | pair | pose | 1 | Rect 开启 wrapped angle CE |
| `geometry` | on | on | 1 | ramp 中心/角度/宽度 loss |
| `negative` | on | on | 1 | 加入可靠 Contact/Rect negative |
| `multigt` | on | on | K | 开启多候选 hard-min |
| `collision` | on | on | K | 使用可靠二维 collision/outside 过滤 |

跨阶段使用新的 optimizer/scheduler：

```bash
JOINT_PHASE=sft \
MODEL_PATH=/models/joint/overfit/checkpoint-xxx \
META_PATH=/data/joint_meta.json \
REALVLG_OUTPUT_DIR=/outputs/joint \
bash training/scripts/train_realvlg_joint.sh
```

只有同一阶段因中断恢复时才设置 `RESUME_FROM_CHECKPOINT`。不要用它代替跨阶段的 `MODEL_PATH`。

## Phase Acceptance

联合 checkpoint 必须包含：

```text
grasp_task_embedding_delta
grasp_task_output_delta
grasp_rect_task_embedding_delta
grasp_rect_task_output_delta
joint_trainer_state.json
phase_acceptance.json
```

记录联合 acceptance：

```bash
python training/scripts/record_phase_acceptance.py \
  --checkpoint /outputs/joint/overfit/checkpoint-xxx \
  --phase overfit \
  --task joint \
  --contact-metrics overfit=/tmp/contact_metrics.json \
  --grasp-rect-metrics overfit=/tmp/grasp_rect_metrics.json \
  --grounding-metrics /tmp/grounding_retention.json
```

grounding 指标可以直接给 retention ratio：

```json
{"retention_ratio": 0.99}
```

或者给 base/current 分数：

```json
{"baseline_score": 0.72, "score": 0.715}
```

默认联合门槛：

- Contact/Rect 格式合法率至少 `0.99`；
- Contact/Rect 坐标 top-1 至少 `0.95`；
- Rect width 合法率 `1.0`；
- Rect 完整六槽率至少 `0.99`；
- Rect 表示天花板兑现率至少 `0.95`；
- grounding retention ratio 至少 `0.98`。

## 评测要求

同一个联合 checkpoint 必须分别评测：

1. grounding/bbox replay，检查原有定位能力；
2. RealVLG Contact，检查 `miou_strict`、`gacc_corrected_strict`、碰撞 unknown/valid；
3. RealVLG Grasp Rect，检查 `mIoU`、corrected `gAcc`、角度 seam 和 width 合法率。

不能只用联合训练 loss 选择 checkpoint，也不能用 Contact 指标代替 Grasp Rect 或 bbox 指标。

## 重要限制

- `<box>`、`<grasp>`、`<grasp_rect>` 必须保持独立；
- 不能按数组下标把 RealVLG `grasps` 和 `contact_points` 直接配对；
- 没有完整覆盖的数据不能自动生成 `none` 或 `ungraspable`；
- VOC bbox 可以作为 grounding replay，但它不是 RealVLG 目标对象 bbox 的等价替代；
- 没有 GPU/真实数据时只能验证 parser、meta、loss、checkpoint 和 dry-run，不能宣称模型联合训练收敛。

