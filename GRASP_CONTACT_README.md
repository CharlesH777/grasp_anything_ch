# grasp_anything 语言引导二维抓取方案

本文档描述 `grasp_anything` 如何在 LocateAnything-3B 上实现语言引导的二维平行夹爪抓取任务。输入为 RGB 图像和自然语言抓取指令，模型通过一次 Parallel Box Decoding（PBD）生成两个夹爪接触点：

```text
(x1, y1, x2, y2)
```

抓取中心、闭合方向和像素开口宽度由两个接触点确定性计算，不直接生成存在周期歧义的角度。训练数据来自 RealVLG-R1 的 `contact_points` 标注。

本方案面向单机 4 x RTX 3090/4090 服务器。当前目标是在 RealVLG-R1 等数据集上取得可靠的二维接触点和二维碰撞指标，不讨论深度恢复、三维姿态、三维碰撞或真机执行。本文档不采用本工作区以前的训练实验配置，训练入口显式禁用 LoRA。

## 0. 当前实施状态

截至当前版本，代码侧已经完成：

- `grasp_contact` 服务模式、严格单块 parser、接触点几何和可视化。
- 基于可靠 obstacle mask 的二维矩形碰撞、越界率和 clearance；mask 缺失时返回 `unknown`。
- RealVLG 正样本转换、原始像素 GT 保留、IoU medoid + FPS、多候选和可选实例排除碰撞标签。
- fixed-K packed 数据桥接、严格四坐标槽 mask、交换不变完整词表 pair CE、多 GT hard-min、中心/角度/宽度门控损失。
- 四卡/梯度累积窗口的全局 shifted-token 与 contact denominator、一次 world-size 补偿、ramp checkpoint 恢复。
- contact-aware PBD top-k 联合解码，首槽和坐标质量严格门控；contact 只接受一次四坐标块或 `none`，Hybrid 遇到旧 point 结构会补齐四坐标后再闭合。
- 严格二维评测：无效正样本计零、修正角度指标并仅以 `official_buggy` 字段保留官方错误实现用于旧日志对账、碰撞有效/unknown 分开统计。
- 每个 `checkpoint-*` 都包含推理代码和 `auto_map`，可直接交给严格评估器。
- 无 LoRA 的四卡 SDPA + BF16 + ZeRO-2 分阶段启动脚本和 CPU 合成回归测试。

仍需在远端数据和 GPU 上完成的验收不是代码单测可以替代的：RealVLG 全量审计、200 张人工可视化、64 样本过拟合、四卡保存/恢复实跑，以及 seen/similar/novel 指标。转换器不会从缺标注推断负样本；`no_target`/`ungraspable` 必须由有完整覆盖和明确语义的数据源提供。当前训练代码实现可靠候选的二维碰撞过滤，但没有默认启用 Phase 6 的连续 SDF 碰撞损失；在 mask 完整性和监督基线通过前，提前启用该项比暂不启用更容易降低效果。

跨阶段时把 `MODEL_PATH` 指向上一阶段通过验收的最佳 checkpoint，并使用新的输出目录和 optimizer/scheduler；不要把上一阶段路径传给 `RESUME_FROM_CHECKPOINT`。后者只用于同一阶段因机器或进程中断后的原位续训，否则旧 global step 和 scheduler 可能让新阶段少训或直接结束。

## 1. 目标与约束

### 1.1 目标

- 输入 RGB 图像和一条自然语言指令，输出一个稳定的两指接触点对。
- 第一版语言语义用于指代目标对象；只有数据提供按语言筛选的 contact 子集时，才宣称支持“抓杯把/抓杯身/从某侧抓”等方式或部位指令。
- 一次 PBD 前向联合生成四个坐标，不调用两次 `point` 模式。
- 由接触点和二维实例 mask 确定性计算碰撞率、间隙及二维可行性。
- 保留 LocateAnything 原有的 detection、grounding、GUI、OCR 和 point 能力。
- 支持目标不存在或目标不可抓取时输出 `none`。
- 兼容 `fast`、`slow`、`hybrid` 三种生成模式。
- 训练过程可分阶段验证，每个新增损失都能单独回滚和消融。

### 1.2 非目标

- 不预测 6-DoF 抓取姿态、深度或物理单位下的夹爪开口。
- 不把二维预测转换为机器人控制指令。
- 不把二维 mask 碰撞代理解释为三维或真实执行安全保证。
- 本阶段不把所有 RealVLG 抓取候选一次性序列化输出。
- 第一版不新增 `<grasp>`、`</grasp>` 词表 token。
- 第一轮训练不使用 GRPO/GSPO；先建立稳定的监督学习基线。

## 2. 关键设计决策

### 2.1 第一版复用 `<box>` 原子块

使用模型已经学会的六 token 结构：

```text
<ref>grasp</ref><box><x1><y1><x2><y2></box>
```

负样本使用：

```text
<ref>grasp</ref><box>none</box>
```

当前 `handle_pattern()` 只检查四个位置是否为坐标 token，并不要求 `x1 < x2` 或 `y1 < y2`，所以四坐标块可以承载两个任意接触点。API 根据 `mode=grasp_contact` 解释其语义，不将其当作左上角/右下角边界框。

第一版不新增结构 token，是为了避免同时改变任务语义和词表。直接增加 `<grasp>` token 需要处理 embedding/lm-head 初始化、词表兼容和 checkpoint 加载，会把接触点学习与词表扩展两个变量混在一起。

### 2.2 NTP 使用规范顺序，交换不变性只作用于 MTP

PBD 训练同时包含：

```text
原始答案 NTP 流 + box-aligned MTP 流
```

在 NTP 流中，交换接触点会改变教师强制上下文，不能在同一次前向中简单计算两种排列的等价损失。因此：

- NTP 标签按端点字典序规范化。
- MTP 四坐标在同一原子块中并行预测，可以使用交换不变损失。
- 第一阶段先只使用规范顺序和原始 CE。
- 模型已经学会任务后，再从 checkpoint 开启 MTP 交换不变损失。

规范顺序定义：

```python
if (x1, y1) > (x2, y2):
    x1, y1, x2, y2 = x2, y2, x1, y1
```

### 2.3 一个输出，多合法标注

RealVLG-R1 的一个目标通常对应多个 `contact_points`。模型任务是输出一个抓取，因此不能把所有候选串成任意顺序，也不应把数十亿抓取全部展开为重复图像样本。

实施分两步：

1. 稳定基线只使用一个确定性主候选。
2. 基线收敛后，最多保留 `K=8` 个去重候选，在 MTP 坐标损失中对候选集合做 hard-min。

hard-min 允许模型选择任意一个合法抓取，适合单输出任务。它可能偏向容易预测的抓取，但不会像对所有候选平均 CE 那样把坐标分布拉向无效中点。

## 3. 几何定义

设：

```text
p1 = (x1, y1)
p2 = (x2, y2)
```

模型输出的 token 坐标位于 `[0,1000]`，但它不是可直接计算角度和欧氏距离的等距坐标系。设原图宽高为 `W,H`，必须先恢复到原图像素坐标：

```text
q1 = (x1 / 1000 * W, y1 / 1000 * H)
q2 = (x2 / 1000 * W, y2 / 1000 * H)
center_px = (q1 + q2) / 2
delta_px  = q2 - q1
width_px  = ||delta_px||
theta_px  = atan2(delta_px_y, delta_px_x) mod pi
```

二维评测约定：

- `theta_px` 表示原图像素坐标中的接触点连线方向，并按 `pi` 周期比较。
- `width_px` 是原图像素开口；跨分辨率统计时使用 `width_px / sqrt(W^2+H^2)`。
- 中心报告可分别给出 `x/W,y/H` 分量，但用于几何辅助 loss 的欧氏中心误差也应乘 `[W,H]/diag`；角度、宽度、矩形 IoU 必须在恢复纵横比后的空间计算。
- 禁止直接对 `[0,1000]^2` 中的两点调用 `atan2` 或欧氏范数。对非方形图像，这会系统性改变角度和宽度。

二维碰撞基线使用与 GraspNet_VLG 官方 contact 评测一致的固定厚度抓取区域。对该官方 split 设厚度为 `T=80` 原图像素、`u=delta_px/||delta_px||`、`v=(-u_y,u_x)`，二维扫掠矩形为：

```text
R = polygon(q1 - T/2*v, q1 + T/2*v,
            q2 + T/2*v, q2 - T/2*v)
```

令 `Mobs` 为图中所有非目标实例 mask 的并集，定义：

```text
collision_ratio_2d = area(R intersect Mobs) / area(R)
outside_ratio_2d   = area(R outside image) / area(R)
clearance_px        = min distance(R, Mobs)
collision_free_2d   = (collision_ratio_2d <= tau_collision)
                      and (outside_ratio_2d <= tau_outside)
```

目标自身 mask 必须从 `Mobs` 排除，因为抓取区域本来就应覆盖目标。`T` 和阈值必须写入评测配置；其他分辨率或 RealVLG 子数据不能无条件复用 80 像素，应使用各自官方定义或对角线归一化厚度再转回像素。

仅凭两个接触点不能唯一恢复真实夹爪的指长、指厚、掌部轮廓和接近轨迹，因此这里的 `R` 只能称为二维 grasp-rectangle collision proxy。若项目还要求更完整的二维夹爪碰撞，必须额外固定一份 2D gripper footprint 配置并另报指标，不能把 proxy collision 与 footprint collision 混为同一数值；两者都不代表三维无碰撞。

## 4. 数据准备

### 4.1 新增转换脚本

新增：

```text
training/scripts/convert_realvlg_contact.py
```

输入为 RealVLG-R1 元数据目录，输出为 LocateAnything JSONL、统计文件和训练 meta。当前本地工作区没有 RealVLG-11B 数据，脚本的真实数据路径应在四卡服务器上配置，不能写死成本机路径。

### 4.2 坐标归一化

与当前服务的 `[0,1000] -> pixel` 逆变换保持一致：

```python
nx = round(x * 1000 / image_width)
ny = round(y * 1000 / image_height)
nx = min(1000, max(0, nx))
ny = min(1000, max(0, ny))
```

不要对 contact point 使用 VOC 边界框的 `xmin - 1` 逻辑。接触点是普通图像点，不是 1-based inclusive box boundary。

转换结果必须保存原图 `image_width` 和 `image_height`。训练中的随机等比例缩放不会改变 token 坐标，但几何辅助损失和评测仍需要原图纵横比。

过滤、rectangle IoU medoid 和 candidate collision 先使用 RealVLG 原始像素浮点坐标；完成候选选择/压缩后再量化成 `[0,1000]` token，并在量化后再次去重和检查退化。评测 GT 始终保留原始像素标注，不能用已经量化的训练副本替代。

### 4.3 正样本过滤

每个候选必须通过以下校验：

- 四个值均为有限数。
- 两个端点均在图像范围内。
- 归一化后两个点不重合。
- 像素宽度或对角线归一化宽度大于数据统计得到的最小值。
- 像素宽度或对角线归一化宽度不超过数据统计得到的最大值。

mask 边界距离和“中心必须在 mask 内”第一版只做审计，不作为硬过滤条件。标注投影误差、细长物体和凹物体都可能违反这些启发式条件；没有可视化统计前强行过滤会丢掉大量困难正样本。

缺少标注不等于不可抓取。不能把 `contact_points=[]` 自动作为负样本，因为它也可能表示标注缺失。

### 4.4 主候选选择

第一阶段每个对象只选择一个主候选：

1. 删除重复或近重复候选。
2. 按该数据集的评测厚度在原图像素空间为每个候选构造 grasp rectangle。
3. 选择与其他合法候选平均 polygon IoU 最高的 IoU-medoid。
4. 对端点做字典序规范化。

这比随机选择稳定，也能避免人为设置中心/角度/宽度距离权重，并直接对齐官方 mIoU。完全并列时用规范化后的四坐标字典序确定性打破平局。其他子数据使用各自评测矩形定义，不能统一套 GraspNet 的 80 像素厚度。

### 4.5 多候选压缩

第二阶段最多保留 `K=8` 个候选：

- 先按中心、`theta mod pi`、宽度或 rectangle IoU 去重。
- 再使用 `1 - polygon_iou` 做 farthest-point sampling，保留矩形几何多样性；如果为速度改用中心/角度/宽度特征距离，必须显式配置各分量权重并做消融。
- 不按原始候选数量复制样本，避免拥有大量抓取标注的对象支配训练。
- `contact_candidates` 中保存的是 `[0,1000]` 坐标值，不是 token ID。
- `contact_candidates[0]` 固定为规范化主候选，conversation 中的 NTP 四坐标必须与它逐值相等；其余候选只供 MTP multi-GT 使用。

### 4.6 二维碰撞标注

训练和评测需要为每个目标构建：

```text
target_mask:      当前语言目标的实例 mask
obstacle_mask:    同一图像中所有其他已标注实例 mask 的并集
collision_valid:  该图像的实例标注是否足以计算碰撞
```

RealVLG 元数据按同一图像列出多个对象时，可用各对象的 `mask_path` 合并 `obstacle_mask`。必须按实例 ID 排除当前目标，不能简单使用前景并集。只有目标 mask 而没有其他实例的完整覆盖时，`collision_valid=False`；未知区域不能默认视为空闲空间。

同图像分组键使用规范化后的 `(dataset, scene, camera, image_path/frame_id)`，不能只用 `image_name` 或 `0000` 帧号；不同 scene 会重复使用相同帧名。合并前断言 RGB 与所有 mask 的原始尺寸一致，并用路径/hash 抽查没有跨场景合并。

Phase 0 对每个 GT candidate 用固定 `T=80` 像素计算碰撞率并输出分布。第一版不因碰撞率硬删 GT；只有确认 mask 覆盖完整、阈值合理且每个目标仍有合法候选后，才把高碰撞 candidate 从 multi-GT 集合排除。否则碰撞标签噪声会误删困难正样本。

在线训练不必传原始高分辨率 mask。转换脚本可预计算固定分辨率的二值障碍图或 signed distance field，并保存其缩放参数。SDF 距离值统一除以原图对角线，不能把低分辨率网格距离直接当原图像素；细小障碍下采样应使用保守的 occupancy/max-pool 规则，避免障碍消失。插值和坐标变换必须有单元测试，精确验证仍使用原分辨率 mask。

### 4.7 负样本

负样本需要区分来源：

- `no_target`：语言描述在图像中不存在。
- `ungraspable`：目标存在，但所有候选均违反当前夹爪约束。
- `ambiguous`：描述无法唯一指定目标，第一版从训练集中排除并单独评测，不自动输出 `none`。

生成 `no_target` 时，需要在对象标注覆盖完整的数据源中验证目标确实不存在，不能只随机交换描述，也不能把“不在 VLG 对象子集”当成“不在图中”。`ungraspable` 只用于数据集明确标注“没有合法二维接触点”的样本。即使 `collision_valid=True`，有限的已标注 candidates 全部碰撞也不证明不存在未标注的可行抓取；只有数据集同时保证候选覆盖完整时才可生成 collision-derived `ungraspable`。不能把缺标注、mask 不完整、候选非穷尽或碰撞状态未知的样本当作负样本。

### 4.8 JSONL 格式

正样本：

```json
{
  "task_type": "grasp_contact",
  "image_width": 1280,
  "image_height": 720,
  "target_mask": "masks/example_target.png",
  "obstacle_mask": "masks/example_obstacles.png",
  "collision_valid": true,
  "contact_candidates": [[136, 456, 540, 880]],
  "candidate_collision_2d": [0.0],
  "candidate_outside_2d": [0.0],
  "conversations": [
    {
      "from": "human",
      "value": "Predict one stable two-finger grasp contact pair for the target described as: the red cup."
    },
    {
      "from": "gpt",
      "value": "<ref>grasp</ref><box><136><456><540><880></box>"
    }
  ],
  "image": "images/example.jpg"
}
```

负样本：

```json
{
  "task_type": "grasp_contact_negative",
  "image_width": 1280,
  "image_height": 720,
  "negative_reason": "no_target",
  "contact_candidates": [],
  "conversations": [
    {
      "from": "human",
      "value": "Predict one stable two-finger grasp contact pair for the target described as: the red cup."
    },
    {
      "from": "gpt",
      "value": "<ref>grasp</ref><box>none</box>"
    }
  ],
  "image": "images/no_red_cup.jpg"
}
```

### 4.9 数据混合

稳定后的目标比例：

| 数据 | 比例 |
|---|---:|
| Contact positive | 70% |
| Contact negative | 10% |
| LocateAnything grounding/detection replay | 20% |

负样本从 5% 起步，确认正样本召回没有下降后再升到 10%。

当前数据集权重由 `repeat_time * len(dataset)` 推导。需要在 meta 中增加显式 `sampling_weight`，否则 RealVLG 的规模会吞掉 replay 和负样本：

```json
{
  "realvlg_contact_positive": {
    "annotation": ".../contact_positive.jsonl",
    "root": ".../RealVLG-11B",
    "task_type": "grasp_contact",
    "sampling_weight": 0.70
  },
  "realvlg_contact_negative": {
    "annotation": ".../contact_negative.jsonl",
    "root": ".../RealVLG-11B",
    "task_type": "grasp_contact_negative",
    "sampling_weight": 0.10
  },
  "grounding_replay": {
    "annotation": ".../grounding_replay.jsonl",
    "root": "...",
    "task_type": "grounding",
    "sampling_weight": 0.20
  }
}
```

`build_stream_packed_dataset_mtp()` 必须实际读取 `sampling_weight` 并直接传给 `StreamPackedDatasetMTP`；只把字段写进 JSON 不会生效。启动后打印的归一化权重和前 1000 个实际采样样本比例都必须接近配置值，否则停止训练。

上述 70/10/20 是样本抽样比例，不是监督 token 或梯度比例。一个 detection replay 样本可能包含多个 box，20% replay 仍可能贡献多数 CE token。每个 optimizer step 还必须按任务记录 `valid_shifted_tokens` 和 loss numerator 占比，再根据验证结果调整 sampler；不能只看样本条数。

不要依赖当前 forward 的 `loss_weight` 做任务调权：本地代码虽然生成了 `shift_loss_weight`，调用 `LigerFusedLinearCrossEntropyLoss` 时并未把它传入，权重会静默失效。第一版优先只通过显式 sampler 控制数据；若以后启用 token loss weight，必须修复调用、纳入全局 denominator，并用两组不同权重的解析测试证明梯度确实改变。

### 4.10 图像增强一致性

当前 `augmentation.py` 只做保持纵横比的 resize，归一化接触点无需改变，这是安全的。以后若增加水平翻转、裁剪、旋转、透视或非等比缩放，必须对以下内容执行同一个几何变换：

```text
全部 contact candidates
target/obstacle masks 或 SDF
image_width/image_height 与坐标逆变换参数
候选碰撞率
```

变换后重新做端点规范化、越界检查和退化检查。第一版只允许等比 resize 与不改变几何的颜色增强；没有成对变换单元测试前禁止启用 flip/crop/rotate。图像变了而 contact 标签未变会造成看似正常下降、实际二维指标很差的隐蔽数据错误。

`contact_image_size` 始终是标注对应的原图 `W,H`，不能用 processor 输出尺寸或 `image_grid_hws * patch_size` 覆盖。当前 `image_processing_locateanything.py` 会分别把宽高对齐到 patch 网格，可能产生轻微非等比变化；这是模型视觉预处理的一部分，不改变输出坐标仍以原图归一化的约定。几何 loss、碰撞和评测都必须使用原图尺寸。

## 5. 训练数据管线修改

主要文件：

```text
training/Eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py
```

### 5.1 Dataset 字段

在 `LazySupervisedDatasetMTP.__init__()` 中增加：

```python
self.task_type = meta.get("task_type", "generic")
self.max_contact_candidates = meta.get("max_contact_candidates", 8)
```

每个样本统一返回：

```text
contact_mtp_coord_mask: [sequence_length] bool
contact_candidates:     [1, K, 4] long
contact_candidate_mask: [1, K] bool
contact_positive_mask:  [1] bool
contact_image_size:      [1, 2] float  # [W, H]
candidate_collision_2d: [1, K] float
candidate_outside_2d:   [1, K] float
collision_valid:        [1] bool
target_sdf:             [1, Hc, Wc] float    # Phase 6 可选
obstacle_sdf:           [1, Hc, Wc] float    # Phase 6 可选
```

普通 grounding 样本和负样本也必须返回这些 key，但内容为空。`_merge_samples()` 会遍历当前 batch 的 key；如果不同数据集返回的 key 不一致，stream packing 会在拼接时失败。

读取每行 JSONL 时断言样本 `task_type` 与 meta 的 `task_type` 一致。当前数据管线主要从 meta 决定任务分支，若正负行混入错误文件而不检查，contact mask 和 `none` 监督会被错误解释。

### 5.2 标记 MTP 坐标

在 `get_targets_flag_with_mtp()` 构建 `target_block` 时，只有满足以下条件才标记 contact 坐标：

```text
task_type == grasp_contact
target_block[0] == <box>
target_block[1:5] 都是坐标 token
target_block[5] == </box>
```

对应 mask 为：

```text
[False, True, True, True, True, False]
```

六槽位唯一约定为：

```text
[<box>, x1, y1, x2, y2, </box>]
```

本地 `generate_utils.py` 的部分 bbox docstring/comment 写成了 `[x1,x2,y1,y2]`，但实际训练 JSON、service parser 和透传代码使用 `[x1,y1,x2,y2]`。实现 contact 几何时以数据与上述断言为准，并修正误导注释；生成器不得重排四个槽位。

只标记附加的 MTP 流。原始 NTP 流 mask 全为 False，使其继续接受规范顺序的普通 CE。

每个正样本第一版只能包含一个 contact box。增加强断言：

```python
assert contact_mtp_coord_mask.sum() == 4
```

负样本的 mask 和 candidates 均为空。

### 5.3 Stream packing

`_merge_samples()` 已经能拼接 tensor，但必须保证所有样本的候选张量具有相同 `K`，碰撞图也具有相同 `Hc,Wc`。在 `packed_collate_fn_mtp()` 中把启用的新增字段加入返回字典。Phase 6 未启用时不要加载 SDF，但同一次 run 内所有 dataset 必须返回一致的 key 集合。

增加以下一致性检查：

- `input_ids`、`labels`、`position_ids` 和 `contact_mtp_coord_mask` 长度一致。
- contact mask 中 True 的数量能被 4 整除。
- contact 正样本数等于 `mask.sum() / 4`。
- 所有有效 candidate 坐标位于 `[0,1000]`。
- 从 conversation 解析出的 NTP contact 等于 `contact_candidates[0]`，且已经按同一规则 canonicalize。
- 每个 contact 正样本都有有限且大于零的原图 `W,H`。
- `collision_valid=True` 的样本必须有有限的 candidate 碰撞率和有效 obstacle mask/SDF。

`contact_mtp_coord_mask` 标记的是 `labels` 的位置，不是 MTP 输入 mask token 的位置。当前数据管线会在原序列和 `final_mask_targets` 之间插入一个 `bridge_ignore`，新 mask 必须做完全相同的拼接和偏移；偏一位会把监督从 `(x1,y1,x2,y2)` 错移到 `<box>` 或 `</box>`，这是 64 样本也无法正常过拟合的硬错误。

## 6. 损失函数实现

主要文件：

```text
training/Eagle/Embodied/eaglevl/model/locany/modeling_locateanything.py
```

当前前向使用 `LigerFusedLinearCrossEntropyLoss`，并故意令 `logits=None` 以节省显存。不能改成对全部序列生成 `[L, 152681]` logits。

### 6.1 基础损失

在 forward 中增加输入：

```python
contact_mtp_coord_mask=None
contact_candidates=None
contact_candidate_mask=None
contact_positive_mask=None
contact_image_size=None
candidate_collision_2d=None
collision_valid=None
target_sdf=None
obstacle_sdf=None
global_ce_tokens_in_window=None
global_contact_count_in_window=None
global_collision_valid_count_in_window=None
num_items_in_batch=None
geometry_loss_scale=0.0
```

模型类还需显式设置 `accepts_loss_kwargs=True`。当前 Transformers 版本只通过该属性或 `forward(**kwargs)` 判断是否传入 `num_items_in_batch`；仅在 forward 中声明一个同名普通参数不会自动开启 Trainer 的 accumulation-window 路径。

复制 labels 并移除 contact MTP 坐标：

```python
base_labels = labels.clone()
base_labels[contact_mtp_coord_mask] = IGNORE_INDEX
```

原有融合 CE 使用 `base_labels`。以下监督仍保留：

- NTP 中的四个规范顺序坐标。
- `<ref>`、`</ref>`、`<box>`、`</box>`。
- `none` 和 `<|im_end|>`。
- 所有普通 LocateAnything 任务。

### 6.2 仅在接触点位置计算完整词表 logits

contact label 位于位置 `t`，由 hidden state `t-1` 预测：

```python
shift_mask = contact_mtp_coord_mask[..., 1:].reshape(-1)
contact_hidden = shift_hidden_states[shift_mask]
```

只选择四个接触点 hidden positions，但这些位置仍必须与完整 lm-head 计算 logits：

```python
contact_logits = F.linear(contact_hidden, lm_head_weight)
contact_logits = contact_logits.float().view(-1, 4, vocab_size)
```

不能只在 1001 个坐标 token 内归一化交换 CE。否则模型只会学习“坐标集合中选哪个”，不会因把更高概率分给文字或错误结构 token 而受罚，可能出现总 loss 下降但 Fast 输出不是坐标的假收敛。

这里只对每个接触点样本的四个位置生成完整词表 logits，而不是对整条 packed sequence 生成，仍然保留大部分融合 CE 的显存优势。lm-head 矩阵乘保持模型的 BF16 dtype，乘法结果再转 FP32 做 `log_softmax`、期望和几何计算。禁止对整个 `lm_head_weight` 调用 `.float()`；3B 模型的大词表权重会产生数 GB 的临时 FP32 副本，四张 24GB 卡也可能因此 OOM。

### 6.3 交换和多候选 hard-min

对每个候选 `g=[x1,y1,x2,y2]` 构造：

```text
identity = [x1,y1,x2,y2]
swapped  = [x2,y2,x1,y1]
```

`contact_candidates` 保存的是数值 `0...1000`，full-vocab CE 的 target 必须是对应坐标 token 的真实词表 ID，不能直接用数值作为 ID：

```python
coord_token_lut = torch.tensor([
    tokenizer.convert_tokens_to_ids(f"<{i}>") for i in range(1001)
], device=device)

# 当前 LocateAnything 词表应当连续；启动时检查一次，不满足就直接报错。
expected = torch.arange(
    coord_start_token_id, coord_end_token_id + 1, device=device
)
assert torch.equal(coord_token_lut, expected)
candidate_token_ids = coord_token_lut[contact_candidates.long()]

none_ids = tokenizer.encode("none", add_special_tokens=False)
assert none_ids == [none_token_id]
```

若把坐标值 `136` 直接用于 `gather(log_probs, target)`，监督的是词表第 136 个普通 token，而不是 `<136>`；这会让 pair loss 完全训练错目标。
负样本的 `none` 也必须是单 token 并与生成配置中的 `none_token_id` 一致，不能使用当前代码的静默 fallback ID。
上述 tokenizer 检查在启动阶段执行一次；`coord_token_lut` 作为 non-persistent buffer 或已迁移到设备的常量复用，不能在每次 forward 中重新调用 tokenizer。

使用 `contact_logits` 的完整词表 log-softmax，计算每种候选、两种排列的四位置平均 NLL：

```python
losses = stack(candidate_identity_losses + candidate_swapped_losses)
losses[invalid_candidates] = inf
pair_loss, best_index = losses.min(dim=-1)
loss_pair = pair_loss.mean()
```

使用 hard-min，不使用所有排列平均，也不在第一版使用 softmin。原因是平均或温度过高的 softmin 会同时推动两个排列，使每个槽位形成双峰分布；四位置独立采样时可能组合出不属于同一抓取的端点。

Phase 3 首次启用时只提供一个主候选。模型已经通过规范顺序训练，所以 identity 分支会自然胜出。多候选到 Phase 5 再打开。

### 6.4 可微坐标

交换 CE 使用完整词表；几何期望只使用其中的坐标切片，并在坐标集合内重新归一化：

```python
coord_logits = contact_logits[
    ..., coord_start_token_id : coord_end_token_id + 1
]
coord_values = torch.arange(1001, device=contact_logits.device).float()
coord_probs = coord_logits.softmax(dim=-1)
pred = (coord_probs * coord_values).sum(dim=-1) / 1000.0
```

若早期分布过平，可在启用辅助损失后使用温度 `T=0.7`，但不要在 Phase 1 使用。辅助损失必须晚于坐标 CE 收敛，否则四个均匀分布的期望都是 0.5，角度和宽度会产生错误梯度。

### 6.5 几何辅助损失

根据 hard-min 选中的 GT 排列计算：

```python
p1 = pred[:, 0:2]
p2 = pred[:, 2:4]
g1 = target[:, 0:2] / 1000.0
g2 = target[:, 2:4] / 1000.0

image_wh = contact_image_size.float()       # [N, 2], [W, H]
image_diag = image_wh.square().sum(-1, keepdim=True).sqrt()
axis_scale = image_wh / image_diag           # 恢复纵横比并按对角线定标

pred_center = (p1 + p2) * 0.5
gt_center = (g1 + g2) * 0.5
pred_center_metric = pred_center * axis_scale
gt_center_metric = gt_center * axis_scale

pred_delta = (p2 - p1) * axis_scale
gt_delta = (g2 - g1) * axis_scale

pred_norm2 = pred_delta.square().sum(dim=-1)
gt_norm2 = gt_delta.square().sum(dim=-1)

pred_width_diag = torch.linalg.vector_norm(pred_delta, dim=-1)
gt_width_diag = torch.linalg.vector_norm(gt_delta, dim=-1)

center_per_contact = F.smooth_l1_loss(
    pred_center_metric, gt_center_metric, reduction="none"
).mean(dim=-1)
valid_angle = (
    (pred_width_diag.detach() > 0.01)
    & (gt_width_diag > 0.01)
)
cosine = F.cosine_similarity(
    pred_delta,
    gt_delta,
    dim=-1,
    eps=1e-8,
)
angle_per_contact = torch.where(
    valid_angle,
    1.0 - cosine.square(),
    torch.zeros_like(cosine),
)
width_per_contact = F.smooth_l1_loss(
    pred_width_diag, gt_width_diag, reduction="none"
)
```

方向损失天然具有端点交换和 `pi` 周期不变性，不使用 `atan2`。`axis_scale` 是必要项：例如 1280x720 图像中的角度不能在正方形 token 网格上直接计算。宽度尚未稳定时不使用 log-width；角度损失只在预测和 GT 宽度都超过最小门槛后计算。不要在平方范数分母中加入量级过大的 epsilon：它会使完美平行的短抓取仍有非零损失，并通过角度项错误地推动开口变大。若当前 batch 没有通过门槛的预测，角度损失返回与计算图相连的 0，而不是对空 tensor 求均值。

几何 loss 还需要门控坐标可信度。由完整词表概率计算每个位置的 `P(coord)`，只有四个位置的最小坐标概率超过阈值且坐标条件分布熵不过高时才启用辅助项；门控值使用 `detach()`，防止模型通过降低置信度逃避几何监督。否则即使 Fast 输出仍是文字 token，坐标子词表内部归一化的 soft-argmax 也会产生一个看似有效的伪坐标。

三个辅助项必须保留为 `[N_contact]` 的逐 contact tensor，不在这里调用 batch `.mean()`。置信度门控失败的 contact 将三项置零；统一 reduction 只在下一节执行，这样才能在四卡和梯度累积下使用正确的全局 denominator。

### 6.6 总损失

不能直接使用 `Lbase + Lpair`。`Lbase` 是所有剩余监督 token 的均值，`Lpair` 是四个接触坐标的均值，直接相加会让四个坐标从原来占少量 token 突然变成与整段文本等权，开启交换损失时梯度尺度会跳变。先恢复当前 rank 的 CE numerator：

```python
n_base = (base_labels[..., 1:] != IGNORE_INDEX).sum()
n_pair = 4 * num_contact_positive
local_ce_sum = n_base * loss_base + pair_weight * n_pair * loss_pair
local_geom_sum = (
    0.25 * center_per_contact
    + 0.10 * angle_per_contact
    + 0.10 * width_per_contact
).sum()
```

初始 `pair_weight=1.0`，它大致保持替换前的 CE 尺度。只有基线稳定并有消融结果后才提高该权重。

四卡和梯度累积下不能再除以当前 microbatch 的 `n_base+n_pair`。stream packing 使不同 rank、不同 microbatch 的有效 token 数不同，DDP 默认对 rank loss 等权平均，Trainer 还可能对累积 microbatch 再平均。正确做法是以整个 gradient-accumulation window、所有 data-parallel rank 的总数作为分母：

```python
# denominator 由 Trainer 在取出完整 accumulation window 后统计并跨 rank 求和；
# 同一 window 内的每个 microbatch 都收到相同的 detached global denominator。
loss_ce_local = local_ce_sum / global_ce_tokens_in_window.clamp_min(1)
loss_geom_local = local_geom_sum / global_contact_count_in_window.clamp_min(1)
loss_collision_local = (
    local_collision_sum
    / global_collision_valid_count_in_window.clamp_min(1)
)
```

随后 DDP 的梯度平均需要恰好补偿一次 `world_size`。使用 Hugging Face Trainer 时，推荐设置 `average_tokens_across_devices=True`，让 Trainer 对最终 loss 乘一次 data-parallel world size；模型内部不要再次乘。若自定义训练循环，则返回 `world_size * local_sum / global_denominator`。两处同时乘会把梯度放大四倍。

当前 Transformers 的默认 `_get_num_items_in_batch()` 只统计原始 `labels.ne(-100)`，没有 contact/collision denominator，也没有严格使用 shifted labels。`StreamPackingMTPTrainer` 应覆盖 accumulation-window 统计逻辑，计算：

```text
global_ce_tokens_in_window
global_contact_count_in_window
global_collision_valid_count_in_window
```

所有 rank 即使本地没有 contact 或 collision 样本也必须参与相同的 collective。增加确定性测试：同一组样本分别用单卡一个 batch、四卡不均匀切分和梯度累积切分运行，比较最终 loss 与关键参数梯度，误差应仅来自数值精度。

碰撞项也先按逐 contact 形式构造带权 `local_collision_sum`。单个 microbatch 返回的总损失为：

```text
Llocal = Lce_local
       + ramp(contact_blocks) * Lgeom_local
       + collision_ramp(contact_blocks) * Lcollision_local
```

辅助损失按累计见过的 contact 正样本原子块数平滑启用，而不是按 optimizer step：

```python
ramp = clamp(
    (seen_contact_blocks - geometry_start_blocks) / geometry_ramp_blocks,
    0,
    1,
)
```

stream packing 使每个 step 的 contact 数量波动很大，按 `global_step` ramp 会让不同分布式配置得到不同的实际升权速度。`seen_contact_blocks` 必须随 checkpoint 保存和恢复。

如果辅助项的梯度范数超过基础损失的 20%，优先降低权重，不要提高 gradient clipping 来掩盖问题。

### 6.7 二维表面与碰撞损失

二维碰撞需要保留，但不进入第一轮坐标基线。它依赖完整实例 mask，并需要把 mask 或距离场加入 batch；在坐标 CE 尚未稳定时启用，会让噪声较大的 mask 梯度压过离散坐标监督。

Phase 6 使用 target/obstacle signed distance field（SDF），根据 soft coordinate 构造接触线和固定厚度二维扫掠区域，在该区域均匀采样。建议定义：

```text
Lsurface   = 两端点到目标边界的距离
Lcenter    = 抓取中心位于目标之外的惩罚
Lcollision = 扫掠区域内 obstacle occupancy
           + relu(clearance_margin - obstacle_distance)
Loutside   = 扫掠区域越出图像边界的比例
```

所有采样先在原图像素几何中定义，再映射到 `Hc x Wc` 的 SDF，不能直接在 1000x1000 token 网格中画固定厚度矩形。`Lcollision` 只对 `collision_valid=True` 样本计算；状态未知的样本返回图连接的零，不当作无碰撞。

构造扫掠区域前复用坐标概率/熵门控，并要求预测开口超过最小门槛。低置信度或退化预测不计算单位方向 `u` 和 collision sampling，先由离散 pair CE（以及置信度足够时的 width loss）拉回有效区域；否则多峰 soft-argmax 或零向量归一化会制造错误碰撞梯度甚至 NaN。

初始 `Lsurface/Lcollision/Loutside` 总梯度贡献不得超过坐标 CE 的 10%，并按累计 contact block 数 ramp。训练损失只用于软约束；验证指标必须用非可微 polygon-mask 相交做精确计算。

若可靠 GT 候选中已有多个无碰撞候选，优先在 multi-GT hard-min 前同时通过 `candidate_collision_2d` 和始终可确定计算的 `candidate_outside_2d` 屏蔽碰撞/越界候选。obstacle 项只在 `collision_valid=True` 时生效，outside 项不依赖 mask；如果全部候选都会被屏蔽则回退原集合，不制造空监督。

### 6.8 数值安全

- 接触点 lm-head 矩阵乘使用模型 dtype；其输出 logits、坐标 softmax、norm 和辅助损失使用 FP32。
- width 使用 FP32 `vector_norm`，不在第一阶段对 width 做 log 或硬截断；angle 只在非退化样本上计算。
- 每个 loss 分量分别检查 `torch.isfinite()`。
- 候选全无效时直接抛出数据错误，不返回零损失。
- 没有 contact 正样本的 packed batch 跳过 contact `F.linear/log_softmax`，返回与图相连的零值，不执行空 `view/mean`。
- 图像 token 与视觉 feature 数量不匹配时直接抛错并在进入 packing 前修复样本；禁止沿用当前 `ignore_flag -> loss * 0` 的静默整包丢弃路径。
- 保存 detached loss breakdown，不能把带计算图的 tensor 挂到模型属性上。

## 7. Trainer 修改

在 `StreamPackingMTPTrainer` 中增加：

- 覆盖 accumulation-window 计数，向每个 microbatch 传入相同的全局 CE/contact/collision denominator。
- 设置模型 `accepts_loss_kwargs=True`，覆盖 `_get_num_items_in_batch()` 使用 shifted CE token 数，并把 contact/collision 全局计数附加到 window 内各 batch。
- 启用并验证 `average_tokens_across_devices=True`，保证 Trainer 不再逐 microbatch 除 gradient accumulation steps，且 DDP world-size 补偿只执行一次。
- 根据累计 `seen_contact_blocks` 计算 `geometry_loss_scale`。
- 将 scale 随 batch 传入模型。
- 每个 logging step 记录各损失分量。
- 对 packed batch 中正、负、replay 样本数做跨 rank 求和后再记录；不能只记录 rank 0 的本地数量。
- 记录实际 `samples/optimizer_step`，因为 stream packing 使全局 batch 不等于简单的 `4 x grad_accum`。

建议增加参数：

```text
--contact_loss_enabled
--contact_max_candidates 8
--geometry_start_blocks
--geometry_ramp_blocks
--contact_center_weight 0.25
--contact_angle_weight 0.10
--contact_width_weight 0.10
--collision_loss_enabled
--collision_weight
--collision_threshold
--gripper_thickness_px 80
```

训练恢复时，loss 配置必须写入 checkpoint 配置或 run metadata。不能只依赖环境变量，否则 resume 后可能在错误 step 重新执行 ramp。
`seen_contact_blocks` 保存全局累计值，并在 optimizer step 边界更新；不能让四个 rank 各自保存不同的本地计数。

## 8. 推理服务修改

### 8.1 Prompt

修改 `src/locate_anything_service/prompts.py`，增加：

```python
PromptMode = Literal[..., "grasp_contact"]
```

模板：

```text
Predict one stable two-finger grasp contact pair for the target described as: {query}.
```

不要要求输出 CoT。长推理文本会增加延迟、打乱 block-aligned 输出，并与 LocateAnything 的结构化解码目标不一致。

### 8.2 Schema

修改 `src/locate_anything_service/schemas.py`，增加：

```python
class GraspContact(BaseModel):
    label: str | None
    contacts_1000: tuple[float, float, float, float]
    contacts_normalized: tuple[float, float, float, float]
    contacts_pixels: tuple[int, int, int, int]
    center_1000: tuple[float, float]
    center_pixels: tuple[int, int]
    angle_radians_image: float
    opening_width_pixels: float
    opening_width_diagonal_normalized: float
    collision_2d_status: Literal["free", "collision", "unknown"]
    collision_proxy_thickness_pixels: float | None
    collision_ratio_2d: float | None
    outside_ratio_2d: float | None
    clearance_pixels_2d: float | None
```

在 `LocateResponse` 中增加：

```python
grasps: list[GraspContact] = Field(default_factory=list)
grasp_status: Literal["ok", "none", "invalid"] | None = None
grasp_parse_error: str | None = None
```

虽然沿用 list 便于兼容现有响应结构，`grasp_contact` 第一版必须断言 `len(grasps) in {0,1}`；它不是多抓取输出接口。`grasps=[]` 时必须通过 `grasp_status` 区分模型明确输出 `none` 和格式解析失败，不能把两者都当成拒绝预测。

### 8.3 Parser

修改 `src/locate_anything_service/parser.py`。不要改变普通 box 的解析语义，新增 `parse_grasp_output()`：

- 接受 `<box><x1><y1><x2><y2></box>`。
- 接受 `<box>none</box>` 并返回空列表。
- 整个响应只能出现一个 contact block；多个 box、`none` 与坐标 box 并存或 box 后继续生成几何内容均判为 strict invalid，不能取第一个或挑最优。
- 不排序端点。
- 不把坐标裁剪后再判断合法性；先检查原始坐标范围。
- 拒绝重合点和超过配置范围的开口。
- 计算中心、角度和像素宽度。

`LocateAnythingRuntime.predict()` 根据 `mode` 选择普通 parser 或 grasp parser。

碰撞不是字符串 parser 的职责。新增 `collision_2d.py`，在得到 `GraspContact` 后使用实例 mask 计算扫掠矩形与障碍物的相交。对外仍可保持 RGB+语言输入：运行时内部调用已选定的实例分割模块生成 target/obstacle masks；若该模块未配置或 mask 置信度不足，必须返回 `collision_2d_status="unknown"`，不能返回 `free`。

离线 benchmark 主要报告 GT mask 下的 oracle collision 指标，部署实验另报 predicted-mask collision 指标，两者不能混在一起。

### 8.4 Visualization

修改 `src/locate_anything_service/visualization.py`，绘制：

- 两个接触点。
- 接触点连线。
- 中心点。
- 与连线垂直的短指身方向。
- 固定厚度二维扫掠区域；无碰撞、碰撞、未知使用不同样式。
- 不绘制普通 bbox 矩形。

### 8.5 API 和测试

修改或新增：

```text
tests/test_prompts.py
tests/test_parser.py
tests/test_model.py
tests/test_api.py
tests/test_grasp_geometry.py
tests/test_collision_2d.py
tests/test_contact_loss_reduction.py
tests/test_grasp_transforms.py
```

覆盖水平、垂直、斜向、接近水平的 1°/2° 角度误差、端点交换、`none`、多个 box、`none` 与坐标并存、越界、重合、格式损坏、目标 mask 排除、障碍物相交、跨 scene 同名帧不合并、边界越界、mask 缺失返回 unknown，以及 grasp 模式不污染普通 box 解析。

## 9. Hybrid 解码修改

相关文件：

```text
training/Eagle/Embodied/eaglevl/utils/locany/generate_utils.py
training/Eagle/Embodied/eaglevl/utils/locany/modeling_locateanything.py
```

当前 Hybrid 把 top-5 坐标跨度较大视为 bbox 空间歧义。接触点任务本身是多模态的：同一物体可能存在多个方向都正确的抓取。直接沿用 bbox 阈值会频繁回退 AR，甚至把合法多峰分布当错误。

给 `generate()` 增加：

```python
geometry_type="bbox"  # bbox | contact
```

contact 模式策略：

- 保留结构 token 错误触发 AR。
- 保留某位置 top-k 中没有任何坐标 token 的错误触发。
- 不把 bbox 的 top-5 coordinate span 直接判错，但将大跨度/高熵视为接触点多模态信号。
- 增加两个端点重合检查。
- 可增加宽度范围检查，但阈值必须来自数据统计。
- 不检查 x/y 的单调顺序。

仅关闭 span fallback 仍不够，因为四个位置最终是分别 argmax。增加可选的单前向 structured rerank：每个坐标位置保留 top-k 坐标 token（第一版 `k<=3`，最多 81 个四元组合），用四位置 log-prob 之和加上非退化宽度、目标表面、图像边界和可用时的二维碰撞约束打分。它不能使用 GT mask；部署只允许 predicted mask，oracle mask 仅用于离线诊断。

若没有任何合法组合、坐标总概率低或最佳/次佳组合 margin 过小，Hybrid 回退 AR。禁止对 top-k 坐标做加权平均，因为平均值可能不属于任何合法抓取。

第一轮分别评测以下互不混淆的模式：

```text
Slow/AR
Fast/raw-argmax
Fast/structured-rerank
Hybrid/rerank-then-AR
```

只有 rerank 和 fallback 阈值在 validation 固定后，才将 Hybrid 作为部署默认；各模式单独报告精度、fallback 和耗时。

## 10. 四卡训练边界（与参数更新方式解耦）

### 10.1 启动方式

新增独立的 `training/scripts/train_locateanything_grasp.sh`，使用服务器最终选定的分布式训练栈启动四个进程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=29520 \
  ...
```

显存检查必须遍历四张卡，不能只检查 `CUDA_VISIBLE_DEVICES` 的第一个编号。

旧实验脚本、冻结配置、学习率和实验报告不属于本方案，不能从中复制超参数。DDP、FSDP 或 ZeRO 的选择由最终主训练配置决定；接触点数据格式和损失实现不得依赖其中任何一种。

### 10.2 任务侧固定项

```text
GPU                         4 x RTX 3090/4090 24GB
attn_implementation         sdpa
block_size                  6
bf16                        True
tf32                        True
seed/data seed              固定
```

`block_size=6` 是 PBD 结构要求，不能为了显存临时改成 4。序列长度、micro batch、梯度累积、优化器、学习率、参数更新范围和切分策略必须由最终训练栈单独做显存探测和基线验证，本 README 不使用旧实验数值。

### 10.3 有效 batch 的统计口径

stream packing 后，一个 optimizer step 内的原始样本数和 contact 原子块数都不是简单的 `world_size * batch_size * grad_accum`。每步必须记录：

```text
contact_positive_blocks
contact_negative_samples
replay_samples
valid_base_tokens
valid_contact_tokens
global_ce_tokens_in_window
global_contact_count_in_window
global_collision_valid_count_in_window
```

学习率或参数更新方式的对比只有在上述任务侧有效 batch 接近时才可解释。改变训练栈时，先重新通过 64 样本过拟合和固定验证集，不得把旧实验 checkpoint 的 loss 数值当作基线。

## 11. 分阶段收敛流程

不能跨过任何阶段的验收门槛。

### Phase 0：数据审计

输出图像数、对象数、正负样本数、坐标/中心/角度/宽度分布、每对象候选数量、target/obstacle mask 覆盖率、GT candidate 二维碰撞率、unknown 比例、过滤原因和 split 泄漏检查。

门槛：

- 0 个越界坐标。
- 0 个退化接触点。
- 0 个图像路径缺失。
- 0 个 train/eval scene 泄漏。
- `collision_valid=True` 样本的 target/obstacle mask 路径、尺寸和实例排除关系全部有效。
- 人工可视化至少 200 个随机样本。

### Phase 1：64 样本过拟合

仅正样本、一个规范主候选、无增强、原始 NTP+MTP CE、无交换/几何/多候选损失。训练到达到以下门槛或确认实现错误，不用旧实验的固定 step 数判断是否收敛。

门槛：

- 训练格式合法率 >= 99%。
- 四坐标 token accuracy >= 95%。
- Fast 模式输出完整六 token block。
- 训练集 endpoint error 接近量化误差。

达不到门槛时不能进入全量训练，优先检查归一化、MTP mask、label shift 和图像/描述对应关系。

### Phase 2：标准 PBD SFT

使用 Contact positive + 20% grounding replay、单主候选和原始 CE。参数更新方式沿用最终主训练配置，以验证集而不是固定 step 数或训练 loss 决定停止。

门槛：

- Slow 和 Fast 均有稳定非零 gAcc。
- Fast 格式合法率 >= 98%。
- grounding replay 指标下降不超过容忍值。
- 坐标没有集中到 0、500、1000 或图像中心。

### Phase 3：交换不变 MTP CE

从 Phase 2 最优 checkpoint 恢复，只开启 `Lpair=min(L12,L21)`，仍使用单主候选。运行长度由固定验证集决定，不沿用旧实验 step 数。

门槛：

- pair loss 持续下降。
- Fast gAcc 不低于 Phase 2。
- identity/swapped 分支选择比例稳定。

### Phase 4：几何辅助损失

从 Phase 3 最优 checkpoint 恢复，center/angle/width 权重按累计 contact 正样本块数从 0 ramp 到目标值。不加入多候选、surface loss 或负样本。

门槛：

- center、angle、width 验证误差至少两项改善。
- gAcc 不下降。
- 辅助损失梯度贡献不超过基础损失约 20%。

### Phase 5：负样本和多候选

按以下顺序单独加入：

1. 5% `no_target`。
2. 验证后升到 10%。
3. 开启最多 K=8 的 multi-GT hard-min。
4. 最后加入数据集明确标注为没有合法二维接触点的 `ungraspable`。

基于“所有候选均二维碰撞”生成的 `ungraspable` 必须同时满足候选覆盖完整，并推迟到 Phase 6 完成 oracle-mask collision 审计和候选过滤消融之后；RealVLG 若不能证明候选穷尽，就只排除该样本，不生成 `none` 标签。

门槛：

- `none` precision 和 recall 同时改善。
- 正样本 recall 下降不超过容忍值。
- 输出 `none` 的比例与负样本比例同量级。
- multi-GT 后 gAcc 提升或保持，端点不出现无效平均。

### Phase 6：二维表面与碰撞约束

仅在监督基线稳定、实例 mask 完整性审计通过后加入 target surface loss 和 obstacle collision loss。先做候选过滤消融，再做低权重连续 loss；一次只启用一种。二维强化学习如需尝试，放在其后，并使用矩形 IoU、角度误差、碰撞率和格式正确率组成奖励。RL 不能用于修复数据格式、归一化、mask 缺失或基础 PBD mask 错误。

## 12. 监控与防塌缩

每个 logging step 至少记录：

```text
loss_total / loss_base / loss_pair
loss_center / loss_angle / loss_width
loss_surface / loss_collision / loss_outside
gradient_norm / learning_rate
positive_samples / negative_samples / replay_samples
collision_valid_samples / collision_unknown_samples
coord_mass_min / coord_entropy / geometry_active_rate
identity_branch_rate / swapped_branch_rate / candidate_assignment_churn
samples_per_optimizer_step
```

每 250 steps 在固定验证集记录：

```text
format_valid_rate
coordinate_accuracy_x1/y1/x2/y2
swap_invariant_endpoint_error
center_error / angle_error_mod_pi / width_relative_error
grasp_rectangle_mIoU / gAcc
collision_rate_2d / mean_collision_ratio_2d / clearance_px_2d
collision_aware_gAcc_valid / collision_aware_gAcc_strict
collision_unknown_rate
none_precision / none_recall / none_F1
fast/slow/hybrid latency and throughput
predicted-mask segmentation latency / collision-check latency
```

立即停止并回滚的信号：

- 任意 loss、logit 或 gradient 出现 NaN/Inf。
- 坐标大量集中到 `<0>`、`<500>` 或 `<1000>`。
- 预测宽度持续趋近 0。
- `none` 比例快速接近 100%。
- 格式合法率连续两个验证点下降。
- grounding replay 指标持续下降。
- 辅助 loss 下降但 gAcc 明显下降。
- collision loss 下降但 oracle collision rate 或 strict gAcc 变差。

不能只监控总 loss，语义和结构 token 数量会掩盖四个坐标位置的失败。

## 13. 评测协议

新增：

```text
training/scripts/evaluate_realvlg_contact.py
```

沿用 RealVLG-R1 的 seen/similar/novel split，并复现其基础指标：

- 以本地实际 evaluator 代码为准：默认 `camera_mode=kinect`，seen=`scene_0100...0129`、similar=`0130...0159`、novel=`0160...0189`，且每个 scene 只取 `0000.json`。`evaluation/dataset.py` 顶部注释的区间与实际 `range()` 不一致，不能照抄注释；其他 camera 单独成表。
- 生成官方评测 JSONL 时必须使用 `convert_realvlg_contact.py --official-graspnet-eval --split seen|similar|novel`；普通训练转换允许多帧，不能直接拿来作为官方测试分母。
- 上述官方表只评 GraspNet_VLG。Cornell、VMRD、OCID 或 Jacquard 若加入训练，必须分别建立其原生 test protocol，不能混入 seen/similar/novel 分母。

- 对所有 GT 接触点候选取最大 IoU 的候选，并使用同一个候选的 angle error 计算官方 gAcc。不能分别取最大 IoU 和最小角度，也不能改成“任一候选同时过阈值”，否则不再复现本地 evaluator。
- 先将 `[0,1000]` 预测按每张图的原始 `W,H` 还原为像素，再由接触点生成抓取矩形。
- 兼容 RealVLG-R1 官方 `eval_contact.py` 时使用其固定 `gripper_width=80` 像素；禁止在归一化 token 坐标上直接套用 80。
- 报告 polygon mIoU。
- 报告 `IoU > 0.25 && angle error < 30 degrees` 的 gAcc。
- 单独报告语法格式有效率和正样本四坐标输出率；合法 `none` 属于格式正确，但在正样本上不属于有效抓取。

本地 `eval_contact.py::angular_diff()` 还有单位启发式错误：调用方传入 degree，但函数在预测和 GT 都位于 `[-pi,pi]` 度时把它们误当 radian。新评测器内部始终保存 radian，并使用明确的 `pi` 周期公式：

```python
delta = torch.remainder(pred_theta_rad - gt_theta_rad + math.pi / 2, math.pi)
angle_error_rad = (delta - math.pi / 2).abs()
angle_error_deg = torch.rad2deg(angle_error_rad)
```

为了和旧结果对账，可额外输出 `official_buggy` gAcc，但它不是合法基线，不能用其选择 checkpoint。

本地 RealVLG-R1 的 `eval_contact.py` 会从 `iou_list` 和 `acc_list` 跳过解析失败样本，因此它打印的 mIoU/gAcc 是“仅合法输出条件下”的指标，会虚高。新评测脚本必须同时报告：

```text
mIoU_valid / gAcc_official_buggy_valid  # 仅复现原脚本错误角度实现
mIoU_strict / gAcc_strict       # corrected angle；非法输出按 0
format_valid_rate
```

checkpoint 选择使用 corrected strict 指标。还要注意本地 `evaluation/dataset.py` 当前按 `grasps` 非空过滤，而 contact 评测应额外要求 `contact_points` 非空，否则无 GT 样本会被记成必错。

`mIoU/gAcc_valid/strict` 的分母都只包含有 GT contact 的正样本；strict 表示其中格式非法的正样本按 0。`no_target/ungraspable` 不存在 polygon GT，只进入 `none precision/recall/F1` 和正样本拒绝率，不能混入 gAcc 分母。正负比例变化时仍应保持正样本几何指标可比。

再增加两点集合交换不变误差、中心像素/对角线归一化误差、像素空间角度模 pi 误差、宽度误差、负样本拒绝指标和各解码变体的吞吐。

二维碰撞指标单独报告：

```text
collision_rate_2d       = 碰撞样本数 / collision_valid 样本数
mean_collision_ratio_2d = 平均扫掠区域障碍占比
mean_outside_ratio_2d   = 平均扫掠区域越界占比
clearance_px_2d         = 无碰撞预测到最近障碍物的像素距离
collision_aware_gAcc_valid  = valid 子集中 gAcc 且 collision_free_2d
collision_aware_gAcc_strict = 全部正样本，invalid/unknown 均按失败
collision_unknown_rate  = 无法可靠构建障碍 mask 的比例
```

主表使用 GT instance masks 的 oracle collision 指标，并按 seen/similar/novel 分开报告。若使用预测 mask，再增加独立表格，同时报告分割质量；不能把预测 mask 漏检造成的“低碰撞率”当成抓取改进。

`tau_collision`、`tau_outside`、mask 置信度、clearance margin 和非官方数据的厚度必须只在 train/validation 上确定并冻结，不能分别在 seen/similar/novel test 上调参。主结果旁报告至少三个固定碰撞阈值的敏感性，避免单一阈值制造结论。

必须比较：

| 基线 | 目的 |
|---|---|
| 两次独立 point 调用 | 验证联合块预测的必要性 |
| 四坐标 Slow/NTP | 验证 PBD 的吞吐收益 |
| 四坐标 Fast/raw PBD | 主要方法的无后处理结果 |
| 四坐标 Fast/structured-rerank | 验证单前向组合约束收益 |
| 四坐标 Hybrid | rerank 后必要时回退 AR 的部署方法 |
| 直接 `(cx,cy,theta,width)` | 验证接触点表示收益 |
| RealVLG-R1 contact model | 对齐数据集原始方法 |

必须消融交换损失、几何损失、单/多 GT、负样本比例、grounding replay 和 Hybrid 判据。参数更新方式属于独立的训练系统变量，不与任务损失消融混在同一张表里。

## 14. 可能导致二维效果不好的因素

本节只讨论会影响二维接触点、矩形 IoU、角度误差、gAcc、二维碰撞指标、负样本指标或推理吞吐的问题。深度、标定、三维姿态、三维碰撞和真机执行不属于当前验收范围。

优先级结论：

| 级别 | 问题 | 后果 |
|---|---|---|
| P0 | 在 `[0,1000]^2` 直接计算角度/宽度 | 非方形图像产生系统性错误几何 |
| P0 | 把坐标值直接当 full-vocab token ID | pair loss 训练到完全错误的词表目标 |
| P0 | 按生成代码的旧注释使用 `[x1,x2,y1,y2]` | 两个接触点被错误组合，全部二维几何失真 |
| P0 | MTP label mask 与 `bridge_ignore` 偏移一位 | 四个接触位置监督错位，无法正常过拟合 |
| P0 | 评测前未恢复原图像素，或非法输出被跳过 | 指标错误且 checkpoint 选择失真 |
| P0 | 沿用 RealVLG 角度单位启发式 | 接近水平的正确抓取可能被算成约 57 倍角误差 |
| P0 | 四卡/梯度累积仍按本地 mean 归一化 | 各 rank 和 microbatch 权重随 packing 随机漂移 |
| P0 | 做 flip/crop/rotate 但未同步变换接触点和 mask | 图像与二维监督直接错位 |
| P1 | 直接使用 `Lbase + Lpair` | 开启交换损失时坐标梯度突然放大 |
| P1 | `sampling_weight` 只写配置但代码不读取 | 数据比例失控，replay 或正样本被吞没 |
| P1 | 把样本比例当成 token/梯度比例，或依赖失效的 `loss_weight` | replay 仍可能压过 contact 监督 |
| P1 | 图像 token/feature 不匹配时静默把 packed loss 置零 | 多条正常样本随坏样本一起被丢弃 |
| P1 | 多候选过早启用 | Fast 输出拼接两个不同候选 |
| P1 | 不完整实例 mask 被当作无障碍 | 碰撞率虚低并产生错误负样本/损失 |
| P1 | 用对象级 RealVLG contact 监督部位/方式指令 | 相互冲突的标签使语言条件退化成目标定位 |

任何 P0 未通过单元测试前都不应开始正式训练。

### 14.1 语言只参与目标定位，没有影响抓取方式

RealVLG 的描述主要用于指出“抓哪个对象”，而同一对象的接触点候选未必随语言中的抓取部位或方式变化。若把同一候选集合同时监督“抓杯把”和“抓杯身”，标签本身互相冲突。模型很可能学成：

```text
语言定位对象 -> 输出该类别的常见二维抓取
```

这种模型在普通 gAcc 上可能不错，但语言条件性较弱。

检测方法：

- 固定图像，只替换目标描述，检查接触点是否切换到对应对象。
- 固定目标，使用等价改写，检查预测是否保持稳定。
- 对 language-shuffled 验证集评测；如果打乱语言后 gAcc 几乎不变，说明模型主要依赖视觉先验。

缓解：

- 保证同一图像中有多个可被不同描述指定的对象。
- 加入 hard positive：类别相同但位置或属性不同的多个实例。
- 第一版 prompt 明确限制为目标指代，不宣称支持抓取方式控制。
- 若要支持部位/方式语言，为每条指令标注其合法 contact candidate 子集，并加入同物体不同指令应输出不同抓取的成对样本。
- 报告 language sensitivity，不只报告总体 gAcc。

### 14.2 多合法抓取导致四个位置组合错配

同一目标可能有方向差异很大的合法候选。PBD 有块内双向注意力，但四个位置最终仍分别产生 token 分布，Fast 模式可能组合出：

```text
p1 来自候选 A，p2 来自候选 B
```

触发信号：

- 格式合法率很高，但矩形 IoU 和 gAcc 明显低于 Slow。
- 单坐标 accuracy 不低，但两点集合误差高。
- 预测中心合理，角度和宽度异常。

缓解：

- Phase 1 只使用一个 medoid 主候选。
- 断言 NTP answer 与 `contact_candidates[0]` 完全一致，避免 Slow/Fast 接受不同主监督。
- Phase 3 使用交换 hard-min，不对排列求平均。
- multi-GT 到 Phase 5 再启用，并限制 `K<=8`。
- 同时报告 Fast/Slow gap；gap 过大时优先检查多峰组合，而不是继续加数据。

### 14.3 hard-min 偏向容易预测的候选

multi-GT hard-min 会选择当前模型 loss 最低的合法候选。它适合单输出任务，但可能长期只学习中心、水平或训练集中最常见的抓取方向。

触发信号：

- 角度和宽度分布明显窄于 GT 分布。
- seen 指标正常，similar/novel 明显下降。
- 不同输入反复输出相近方向或宽度。

缓解：

- 候选压缩时按中心、角度和宽度做多样性采样。
- 按角度/宽度桶统计 gAcc，避免总体均值掩盖长尾失败。
- 可在后期对稀有方向候选增加采样权重，但不能在基础模型尚未收敛时启用。

### 14.4 复用 `<box>` 产生任务语义冲突

同一个结构在原任务表示 bbox 左上/右下角，在抓取任务表示无序接触点。持续微调可能让抓取输出继承 bbox 的坐标单调先验，也可能损害原始 bbox 能力。

触发信号：

- 接触点很少出现负斜率方向。
- 模型倾向满足 `x1<x2` 且 `y1<y2`，即使规范化只要求端点字典序。
- grounding replay 的高 IoU 指标持续下降。

缓解：

- 使用固定 `<ref>grasp</ref>` 和明确的 `grasp_contact` prompt。
- 保留至少 20% grounding/detection replay。
- 分别统计四种坐标象限关系。
- 若确认存在结构冲突，再做独立 `<grasp>` token 消融，不在第一轮同时引入。

### 14.5 几何 soft-argmax 对多峰分布取无效平均

坐标 token 分布可能在多个合法值上有峰值。soft-argmax 的期望可能落在两个候选之间，导致辅助损失推动一个没有 GT 支持的中间抓取。

触发信号：

- 几何辅助 loss 下降，但 argmax gAcc 下降。
- soft-argmax 派生几何与 argmax 派生几何差距持续增大。
- 坐标分布熵在辅助损失启用后上升。

缓解：

- 只有坐标 CE 已收敛后才 ramp 辅助损失。
- 同时记录 soft prediction 和 argmax prediction 的二维指标。
- 同时检查完整词表中的坐标总概率；低坐标概率或高熵位置跳过辅助损失，或使用 top-k 局部归一化期望。
- 辅助损失梯度贡献不超过基础损失约 20%。

### 14.6 接触点位置没有使用完整词表 CE

如果交换损失只在 1001 个坐标 token 内做条件 softmax，模型不会因把更高概率分给文字或结构 token 而受罚，可能出现 pair loss 很低但 Fast 输出非法 token。

防护：

- 只选择四个接触点 hidden positions，但必须计算这些位置的完整词表 logits。
- exchange hard-min 使用完整词表 CE。
- full-vocab target 必须由 `coord_token_lut[value]` 得到，不能直接使用 `0...1000` 数值。
- 启动时断言 `none` 恰好编码成配置中的单个 `none_token_id`，不允许 fallback ID。
- 1001 坐标切片只用于几何 soft-argmax。
- 每次验证必须报告 Fast 格式合法率。

### 14.7 负样本错误或比例过高导致 `none` 塌缩

负样本比精确坐标更容易学习。错误负样本或比例过高会让模型通过输出 `none` 降低平均 loss。

触发信号：

- `none` 输出率远高于训练负样本比例。
- none recall 上升但 precision 和正样本 recall 同时下降。
- 困难正样本首先被预测为 `none`。

缓解：

- 负样本从 5% 开始，稳定后再升到 10%。
- 缺少 contact 标注不能自动视为负样本。
- 正样本 recall、none precision/recall/F1 必须独立报告。
- 正负样本使用显式 `sampling_weight`，不能让数据量隐式决定比例。

### 14.8 把训练系统变化误判为任务设计收益

本工作区以前的参数更新配置和实验报告不作为本方案依据。如果在加入接触点损失的同时更换优化器、参数更新范围、有效 batch 或学习率，就无法判断效果变化来自任务设计还是训练系统。

防护：

- 先用最终训练栈复现未加接触点任务时的固定基线。
- 每次任务消融保持训练系统和实际 contact/replay 样本数一致。
- 64 样本无法过拟合时，先检查 token ID、label shift、坐标纵横比和图文对应，不引用旧实验 loss 判断容量。
- 参数更新策略单独实验，不写入接触点方法的主消融表。

### 14.9 bbox Hybrid 判据误伤合法多峰接触点

当前 bbox Hybrid 将 top-5 坐标跨度视为歧义。接触点任务中的多峰可能都对应合法抓取，直接复用会导致不必要的 AR 回退或错误截断。

触发信号：

- Hybrid 大量切换到 AR，吞吐接近 Slow。
- Fast gAcc 正常但 Hybrid 没有精度收益。
- 多候选样本的 fallback 次数显著高于单候选样本。

缓解：

- contact 模式关闭“coordinate span 大就直接判错”的 bbox 规则。
- 大跨度/高熵改为触发小规模 structured rerank，组合仍不可靠时再回退 AR。
- 保留格式错误、无坐标候选、退化宽度、越界和可用时的二维碰撞等 contact-specific 检查。
- 报告平均 fallback 次数和 fallback 样本比例。

### 14.10 评测随机性掩盖真实差异

当前工程部分评测使用 `do_sample=True`。接触点任务的候选本来就是多模态的，采样会显著增加 checkpoint 间方差。

缓解：

- checkpoint 选择统一使用 greedy/temperature=0。
- 服务 API 的 `grasp_contact` 同样强制 `do_sample=False`、`temperature=0`、`top_p=None`，同图同指令不得因默认采样配置漂移。
- 固定图像列表、prompt、输入分辨率和数据顺序。
- sampling 结果作为附加实验，至少运行三个固定 seed。
- 只有差异超过重复运行方差时，才判断一个 checkpoint 更好。

### 14.11 数据泄漏和重复图像造成虚高

RealVLG 同一 scene、对象和相邻帧可能高度相似。按 JSON 行或图片随机切分会使验证指标虚高；将每个 contact candidate 展开为独立样本也会让高候选对象被过度采样。

防护：

- 严格使用 scene/object 级 train、seen、similar、novel 划分。
- 检查 image hash、scene ID 和 object ID 交叉集合。
- 每个对象保留固定上限候选，不按候选总数复制权重。
- 同时报告 seen、similar、novel，不能只看总体平均。

### 14.12 单一 gAcc 不能定位二维误差来源

RealVLG gAcc 同时使用矩形 IoU 和角度阈值。模型可能因为中心、方向、宽度或格式中任一项失败，仅看 gAcc 无法知道改动是否有效。

必须同时报告：

```text
format validity
exchange-invariant endpoint error
center error
angle error mod pi
width absolute/relative error
rectangle mIoU
gAcc
```

模型选择以 gAcc/mIoU 为主，但任何损失设计都必须结合分解指标解释。

### 14.13 归一化 token 网格扭曲像素几何

LocateAnything 分别用 `W,H` 把横纵坐标归一化到 1000。该表示适合输出坐标，但不是保角、保距变换。对 1280x720 等非方形图像，直接使用 token 差值计算角度、宽度、medoid 或几何 loss 会持续优化错误目标。

防护：

- 数据中保存每张图的原始 `W,H`。
- 欧氏中心、角度和宽度损失先乘 `axis_scale=[W,H]/diag`。
- 主候选使用原图像素 rectangle IoU medoid；候选去重和 FPS 使用相同 rectangle 几何或恢复纵横比后的空间。
- 评测先恢复原图像素，再构造固定 80 像素厚度的矩形。
- 单元测试至少包含同一像素角度在 `1:1`、`16:9` 和 `9:16` 图像中的恢复结果。

### 14.14 CE reduction 和四卡归一化错误

原始 fused CE 对所有有效 token 求均值。移除四个 MTP 坐标后若直接相加一个按四坐标求均值的 `Lpair`，相当于显著提高接触点 token 权重，Phase 3 开启瞬间可能破坏已经学会的格式和定位。

即使单卡公式正确，四卡 stream packing 仍会让每个 rank 和累积 microbatch 拥有不同数量的有效 token/contact。如果各自先求 mean 再由 DDP 等权平均，短 batch 与长 batch 权重相同，实际 contact/replay 比例会逐步漂移。

防护：

- 用 `n_base` 和 `n_pair=4*N` 加权恢复统一 token mean。
- denominator 按完整 gradient-accumulation window 统计，并跨所有 data-parallel rank 求和。
- geometry 和 collision 分别使用全局 contact/collision-valid denominator。
- 显式设置 `accepts_loss_kwargs=True`；仅增加普通 forward 参数不会让当前 Trainer 自动走正确的 accumulation 路径。
- DDP world-size 只补偿一次；Trainer 与模型不能重复乘。
- 初始 `pair_weight=1`，记录切换前后总梯度范数和 contact 坐标梯度范数。
- Phase 3 第一批数据上要求新旧 CE 在仅 identity、单 GT 条件下数值近似一致。
- 增加单卡、四卡不均匀切分和梯度累积三种方式的 loss/gradient 等价性测试。

### 14.15 官方评测会跳过非法输出并误判角度单位

本地 RealVLG-R1 `eval_contact.py` 只把成功解析的结果加入 mIoU/gAcc 列表。格式越差，条件指标反而可能看起来越高；同时它用原图像素中的固定 `gripper_width=80`，不能直接接收 `[0,1000]` 坐标。其 `angular_diff()` 还会把绝对值小于 `pi` 的 degree 误判成 radian，使接近水平的角度误差严重失真。

防护：

- 同时报告 valid-only 和 invalid-as-zero 两套指标，模型选择只使用 strict 指标。
- 角度内部固定使用 radian 和显式 `pi` 周期，不做单位猜测；`official_buggy` gAcc 只用于和旧日志对账。
- 记录分母和非法样本数，禁止只抄脚本打印的 mIoU/gAcc。
- contact 评测过滤条件使用 `contact_points` 非空，而不是只检查 `grasps`。

### 14.16 二维碰撞 mask 不完整或定义不一致

RealVLG 的目标 mask 不自动等于完整场景障碍标注。若只看到当前目标 mask 就把其余区域当作空闲，未标注或被漏检的物体会使碰撞率虚低；若没有排除目标自身，几乎所有正确抓取又会被判为碰撞。

防护：

- 按同图像全部实例构建障碍并明确排除当前 target instance ID。
- 用 `(dataset,scene,camera,image_path/frame_id)` 分组，禁止只按会重复的帧名合并 mask。
- 用 `collision_valid/unknown` 区分完整标注和未知状态，unknown 不参与有效子集均值。
- 另报把 unknown 视为失败的 strict collision-aware gAcc，防止通过降低 mask coverage 提高分数。
- 固定并记录扫掠区域厚度、障碍阈值、mask 分辨率及插值方式。
- GT mask 与预测 mask 指标分表报告；预测 mask 的漏检不能算抓取模型收益。

### 14.17 几何增强没有同步变换监督

当前代码的 resize 保持纵横比，因此归一化坐标仍有效。但若以后直接加入通用视觉增强，水平翻转、裁剪、旋转或透视会改变接触点、角度、mask 和碰撞关系。只增强 RGB 会产生完全错误但格式仍合法的标签。

防护：

- 第一版只使用等比 resize 和颜色增强。
- `contact_image_size` 保持原图尺寸，禁止替换成 processor 的 patch-aligned 输出尺寸。
- 所有几何增强通过同一个 transform 同时处理图像、全部候选和 mask/SDF。
- 变换后重新 canonicalize 端点并重算 candidate collision。
- 对水平翻转、裁剪和旋转分别做像素级可视化单测，未实现前配置层禁止开启。

### 14.18 数据集与碰撞协议混用

本地 RealVLG contact evaluator 实际只针对 GraspNet_VLG 的固定 scene 和首帧，并使用 80 像素厚度。把 Cornell、VMRD、OCID、Jacquard 或不同分辨率样本混入同一个分母，会让 gAcc 和碰撞率失去可比性；两个接触点本身也不能恢复真实夹爪完整轮廓。

防护：

- GraspNet 官方表严格使用代码中的 `100:130/130:160/160:190` 和 `0000.json`。
- 其他子数据各自使用原生 split、厚度和坐标约定，最后只做宏平均附表。
- `T=80` 指标命名为 grasp-rectangle collision proxy。
- 如需指身/掌部二维碰撞，额外固定 gripper footprint，并与 rectangle proxy 分表报告。

### 14.19 样本混合比例不等于梯度比例

stream sampler 的 70/10/20 控制抽样次数，但不同任务的答案长度、box 数量和有效 token 数差异很大。当前模型代码还没有把计算出的 `shift_loss_weight` 传给 fused CE，因此配置 token 权重也不会自动生效。

防护：

- 同时记录每个任务的样本数、有效 shifted token 数、CE numerator 和梯度范数。
- 第一版用 sampler 调整数据，不叠加未验证的 token weight。
- 若必须使用 `loss_weight`，先修复 fused loss 调用，并把加权 denominator 纳入跨卡/累积窗口归一化。
- 固定验证集上同时看 contact strict gAcc、collision-aware gAcc 和 replay 指标，不能按训练 loss 占比直接推断混合是否合理。

### 14.20 图像对齐错误静默丢弃整个 packed batch

当前 `modeling_locateanything.py` 在 image token 数和视觉 feature 数不一致时设置 `ignore_flag=True`，最终执行 `loss = loss * 0.0`。stream packing 下，一个坏样本会让同包其他正常样本也没有梯度，而全局 token denominator 仍可能把它们计入。

防护：

- 保留 dataset 的 `_validate_image_token_alignment()`，并在 packing 前 fail-fast。
- 模型训练路径遇到 mismatch 直接抛出带 dataset/sample ID 的异常，不返回零 loss。
- 记录每个数据源的预处理失败率；任何数据源持续失败都停止训练，而不是通过 retry 隐藏。
- 分布式下保证所有 rank 协调退出，避免一个 rank 抛错、其他 rank 卡在 collective。

### 14.21 四坐标槽位注释与真实数据顺序不一致

本地 `generate_utils.py` 的部分注释把 bbox 写成 `[x1,x2,y1,y2]`，实际 VOC JSON、service parser 和模型输出都是 `[x1,y1,x2,y2]`。函数当前没有主动重排，所以照旧注释实现抓取会把两个点错误解释成 `(x1,x2)` 与 `(y1,y2)`。

防护：

- 在 converter、MTP mask、loss、parser 和 evaluator 中共享一个命名常量 `CONTACT_SLOTS=(X1,Y1,X2,Y2)`。
- 用一组非对称坐标（例如 `100,250,700,900`）做端到端 round-trip，避免对称样例掩盖顺序错误。
- 修正生成代码的旧 docstring，但不改变普通 bbox 的运行时槽位顺序。

## 15. 文件修改清单

新增：

```text
training/scripts/convert_realvlg_contact.py
training/scripts/evaluate_realvlg_contact.py
training/scripts/train_realvlg_contact.sh
training/configs/grasp_anything_realvlg_contact.env
training/data/realvlg_contact_meta.example.json
tests/test_grasp_geometry.py
tests/test_collision_2d.py
tests/test_realvlg_contact_pipeline.py
tests/test_training_meta_contact.py
tests/test_contact_data_bridge.py
tests/test_contact_loss.py
tests/test_contact_decode.py
```

修改服务层：

```text
src/locate_anything_service/prompts.py
src/locate_anything_service/schemas.py
src/locate_anything_service/parser.py
src/locate_anything_service/collision_2d.py
src/locate_anything_service/model.py
src/locate_anything_service/visualization.py
tests/test_prompts.py
tests/test_parser.py
tests/test_model.py
tests/test_api.py
```

修改训练层：

```text
training/Eagle/Embodied/eaglevl/dist_utils.py
training/Eagle/Embodied/eaglevl/train/arguments.py
training/Eagle/Embodied/eaglevl/train/grasp_contact.py
training/Eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py
training/Eagle/Embodied/eaglevl/model/locany/modeling_locateanything.py
training/Eagle/Embodied/eaglevl/utils/locany/generate_utils.py
training/Eagle/Embodied/eaglevl/utils/locany/modeling_locateanything.py
training/scripts/train_realvlg_contact.sh
```

训练代码位于 gitignored 的 Eagle 子仓库。所有修改必须同步维护为项目 patch，避免服务器重新 clone Eagle 后丢失：

```text
training/patches/locateanything-grasp-contact.patch
```

`training/scripts/train_realvlg_contact.sh` 启动时会先执行双向
`git apply --check`：补丁已应用则继续，干净基线则自动应用，Eagle revision
或工作区不兼容时立即退出。禁止绕过该检查直接调用训练 Python 入口。

## 16. 完成定义

- 64 样本过拟合测试通过。
- 最终选定的四卡分布式训练栈可以从零启动、保存和恢复。
- 单卡、四卡不均匀 packing 和梯度累积下的 loss/gradient 等价性测试通过。
- Resume 后 loss ramp 和数据采样状态一致。
- Fast 模式一次生成完整接触点对。
- 交换端点不改变派生中心、方向模 pi 和宽度。
- 正样本 gAcc 达到预设目标。
- oracle-mask 与 predicted-mask 二维碰撞指标分开完成，unknown 不被计作无碰撞。
- collision-aware strict gAcc 达到预设目标，且碰撞约束没有显著降低普通 strict gAcc。
- 负样本不存在全局拒绝塌缩。
- 原始 grounding 能力下降处于容忍范围。
- Fast 相比两次 point 调用有明确吞吐收益。
- seen、similar、novel 三个二维 split 均完成评测。
- 所有启用的几何增强都有 image/contact/mask 同步变换测试。
- 所有训练修改有 patch、测试和配置记录。

## 17. 推荐实施顺序

```text
1. 数据审计和转换器
2. token ID / MTP shift / aspect-ratio / transform 单元测试
3. 单卡-四卡-梯度累积 loss/gradient 等价性测试
4. grasp_contact API/parser/visualization
5. 原始 CE 的 64 样本过拟合
6. 四卡标准 PBD SFT
7. MTP exchange hard-min
8. center/angle/width ramp
9. negative samples + grounding replay
10. multi-GT hard-min
11. contact-specific Hybrid
12. oracle-mask 2D collision evaluation
13. collision-aware candidate filtering
14. low-weight 2D surface/collision loss
15. optional 2D reward RL
```

任何阶段出现不收敛，都回到上一阶段 checkpoint，只保留一个新增变量进行排查。不要通过同时调大学习率、放宽梯度裁剪或继续堆训练步数来掩盖数据或 mask 错误。
