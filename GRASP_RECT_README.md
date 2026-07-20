# grasp_anything 语言引导二维矩形抓取方案

本文档描述如何在现有 `grasp_contact` 能力旁新增 RealVLG-R1 的二维矩形抓取任务。模型直接输出：

```text
(center_x, center_y, theta, width)
```

输出被严格解析和联合解码后，再确定性展开为四个角点，即 8 个标量：

```text
(x0, y0, x1, y1, x2, y2, x3, y3)
```

本文档既是冻结的设计基线，也是实施和验收依据。代码实现已于 2026-07-20
完成本地闭环；真实数据审计、GPU 训练和 checkpoint 阶段验收仍必须按本文门槛
逐项执行，不能因代码测试通过而视为模型阶段通过。现有 Contact 任务仍以
[`GRASP_CONTACT_README.md`](GRASP_CONTACT_README.md) 为准，两条任务必须独立评测、独立报告，不能把矩形抓取结果混入 Contact 指标。

## 0. 当前结论和协议来源

### 0.1 已确认的 RealVLG 协议

RealVLG-R1 的 Grasp 和 Contact 是两个不同任务：

- Contact 输出两个二维接触点 `(x1, y1), (x2, y2)`。
- Grasp 输出二维矩形参数 `(x, y, theta, width)`。
- Grasp 官方代码使用固定 `gripper_depth=40.0` 像素，将四个参数确定性展开成 4 个角点、共 8 个坐标标量。
- 官方 Grasp 评测将预测矩形与所有 GT `grasps` 比较，按最大 polygon IoU 选择 GT，再用同一个 GT 的角度误差计算 gAcc。
- gAcc 判定为 `IoU > 0.25` 且角度误差 `< 30 degrees`。
- RealVLG 没有定义真实夹爪的毫米制最大张开行程，也没有在 parser、reward 或 evaluator 中检查 `width` 是否超过硬件最大开口。
- RealVLG 官方为 Grasp 和 Contact 提供不同的训练脚本、prompt、reward function 和发布仓库；公开的 `RealVLG-R1_GRPO_Grasp_3B` 与 `RealVLG-R1_GRPO_Contact_3B` 是两个独立任务 checkpoint，不是一个公开的联合权重。

因此本方案明确采用以下边界：

- `width` 必须严格大于最小非退化宽度。
- 不新增 `maximum_gripper_opening` 或类似物理上限。
- `[0,1000]` token 编码域带来的图像对角线表示范围只是编码约束，不宣称为真实夹爪能力。
- 如果未来进入真机执行阶段，物理夹爪行程必须由相机标定、深度、物体尺度和硬件配置共同定义，并作为独立的部署后处理，不能回写成 RealVLG 官方二维指标的一部分。

### 0.2 已确认的实施决策

第一版按以下方案实施，不再作为待选项：

- Grasp 使用独立 checkpoint 作为主线，Contact 当前 checkpoint 保持不变。
- Grasp 使用独立 `<grasp_rect>` start/end token，不复用 Contact `<grasp>`。
- Contact 的端点交换 Pair Loss 不直接用于 Grasp。
- Grasp 角度使用 1001-bin 圆周编码。
- 结构基线通过后，角度分类使用半径 `r=1` 的 wrapped marginal CE，将 target 及其圆周相邻 bin 作为同一量化邻域。
- 几何阶段叠加 double-angle 周期损失，显式表达 `theta` 与 `theta+180 degrees` 等价。
- Grasp 与 Contact 的联合权重、跨任务一致性和多任务 replay 只作为独立 Grasp 主线完成后的可选消融，不进入第一版官方对齐基线。
- 如果未来尝试跨任务一致性，必须先证明 `grasps` 与 `contact_points` 可以逐候选可靠对应；不能直接按列表索引 `zip()`。

### 0.3 固定参考版本

协议核对基于 RealVLG-R1 官方仓库 commit：

```text
040562e0cf8f64a8c6e922d8f7e5e098bb3633c3
```

关键来源：

- [RealVLG-R1 README](https://github.com/lif314/RealVLG-R1/blob/040562e0cf8f64a8c6e922d8f7e5e098bb3633c3/README.md)
- [官方 Grasp evaluator](https://github.com/lif314/RealVLG-R1/blob/040562e0cf8f64a8c6e922d8f7e5e098bb3633c3/evaluation/eval_grasp.py)
- [官方 Grasp reward](https://github.com/lif314/RealVLG-R1/blob/040562e0cf8f64a8c6e922d8f7e5e098bb3633c3/examples/reward_function/rect_grasp.py)
- [RealVLG-R1 论文](https://arxiv.org/abs/2603.14880)

实现时应把上述 commit 写进评测产物，避免官方代码后续变化后无法复现实验。

### 0.4 当前实施状态（2026-07-20）

代码侧已经完成：

- 独立 `<grasp_rect>` token、input/output delta adapter、checkpoint 配置和
  preflight 校验；Grasp Rect 训练不更新 Contact task adapter。
- 共享四参数编码、`points8_to_rect()`、`rect_to_points8()`、严格 parser、
  API schema、可视化、越界和二维碰撞诊断。
- RealVLG Grasp converter、medoid/FPS、多 GT 保留、官方 split、样本 ID hash、
  固定官方 commit 和 Phase 0 审计/可视化入口。
- corrected strict evaluator；valid、strict、official buggy、negative、outside、
  latency 和 fallback 指标分开记录。
- 独立 fixed-K data bridge、全局 pose denominator、`r=0/1` wrapped CE、
  double-angle loss、geometry ramp、multi-GT hard-min 和可靠负样本字段。
- Fast 联合 top-k、严格宽度合法性、独立 AR 槽约束和 Hybrid 单次回退。
- 专用阶段脚本、配置模板、meta 示例、phase transition/acceptance 门禁，以及
  可从固定 Eagle revision 应用的 `locateanything-grasp-rect.patch`。

本地代码验收：全量 pytest 通过，新增 Python 文件定向 Ruff 通过，Shell
脚本通过 `bash -n`，Eagle patch 在干净 `783f656d127ee498137b5ff52603ce36c292d317`
checkout 上通过正向、反向 `git apply --check`。

尚未宣告通过的模型阶段：

- 官方 90 个评测帧已用于冻结 seen/similar/novel count/hash；完整训练集尚未
  执行 200+50 张人工审计，因此仍没有 Phase 0 完成标记。
- Phase 1 的 64 样本过拟合尚未启动，没有 Grasp Rect checkpoint。
- Phase 2 到 Phase 6 均未训练或验收。必须从上一阶段已接受 checkpoint 启动
  新优化器，不能跳阶段。

Phase 0 入口：

```bash
PYTHONPATH=src .venv/bin/python training/scripts/audit_realvlg_grasp.py \
  --data-root /path/to/GraspNet_VLG \
  --output-dir /path/to/grasp_rect_phase0
```

脚本内已冻结官方 commit 对应的三个 split count/hash。首次运行只生成统计与
200+50 张审计图，manifest 保持 `accepted: false`。人工检查两组图后，以相同参数
重跑并增加 `--confirm-visual-review`，此时才会写 `accepted: true` 和
`.phase0_complete`。任何失败重跑都会先撤销旧完成标记。sample ID 只使用
`data_root` 相对图片路径，因此数据挂载点变化不会改变 hash。

训练预览和启动入口：

```bash
GRASP_RECT_PHASE=overfit \
MODEL_PATH=/path/to/LocateAnything-or-approved-source \
META_PATH=/path/to/grasp_rect_overfit64_meta.json \
PHASE0_AUDIT_PATH=/path/to/grasp_rect_phase0 \
REALVLG_OUTPUT_DIR=/path/to/grasp_rect_outputs \
DRY_RUN=1 \
bash training/scripts/train_realvlg_grasp.sh
```

## 1. 目标与非目标

### 1.1 目标

- 输入 RGB 图像和自然语言目标描述，输出一个二维矩形抓取姿态。
- 使用一次六槽 PBD 原子块联合生成中心、角度和宽度，而不是四次独立调用。
- 在联合 top-k 排名阶段检查中心、角度和宽度是否合法。
- 选出最终四参数后，使用一个共享纯函数确定性生成官方顺序的 4 个角点、共 8 个坐标标量。
- 兼容 LocateAnything 的 `fast`、`slow` 和 `hybrid` 三种生成模式。
- 支持 RealVLG 同一目标的多个合法 `grasps`，最终只输出一个抓取。
- 保留现有 Contact checkpoint 作为不可变基线；Grasp 训练输出到独立 checkpoint，不覆盖 Contact 权重。
- 通过 grounding replay 控制 Grasp checkpoint 对 LocateAnything 基础定位能力的遗忘。
- 严格复现 RealVLG seen、similar、novel Grasp 评测，同时修复官方 evaluator 的条件分母和角度单位问题。
- 输出中心误差、角度误差、宽度误差、IoU、gAcc、格式合法率、越界率和二维碰撞诊断。

### 1.2 非目标

- 不预测 6-DoF 抓取姿态、深度、接近方向或夹爪闭合力。
- 不把二维 `width` 解释为毫米或米制夹爪开口。
- 不增加 RealVLG 未定义的最大夹爪张开幅度。
- 不从二维矩形单独推断真机可执行性。
- 不把固定 `gripper_depth=40 px` 解释为真实夹指长度或厚度。
- 第一轮不使用 GRPO/GSPO。先建立可复现的监督学习和结构化 PBD 基线。
- 不用后处理裁剪、强行缩短宽度或把矩形拉回图像内来掩盖模型错误。

## 2. 与 Contact 方案的关系

两条任务可以复用基础设施，但不能复用错误的几何语义。

| 项目 | Contact | Grasp Rect |
|---|---|---|
| 模型输出 | `(x1,y1,x2,y2)` | `(cx,cy,theta,width)` |
| 原子块长度 | 6 | 6 |
| 四个内容槽 | 四个点坐标 | 中心 x、中心 y、角度 bin、宽度 bin |
| 对称性 | 两端点可交换 | `theta` 具有 180 度周期 |
| 任务对称损失 | identity/swapped Pair CE | wrapped angle CE + double-angle loss |
| 角度来源 | 两点确定性派生 | 模型直接预测 |
| 宽度来源 | 两点距离 | 模型直接预测 |
| 矩形短边 | Contact 官方固定 80 px | Grasp 官方固定 40 px |
| 最终 polygon | 接触线扫掠矩形 | 四参数确定性展开矩形 |
| 官方 GT | `contact_points` | `grasps` 的 8 标量角点 |

可以直接复用：

- RealVLG 元数据遍历、scene split 和图像尺寸读取。
- fixed-K packed candidate 张量、候选 mask 和全局 denominator 框架。
- PBD 六槽 mask、Fast/Hybrid 结构强制、坐标概率质量门控。
- IoU medoid + FPS 的单主候选和 multi-GT 课程。
- grounding replay、四卡训练、checkpoint phase acceptance 和严格评测框架。
- target/obstacle mask 的可靠性约定及 `collision_valid/unknown` 语义。

不能直接复用：

- Contact 的端点字典序规范化。
- Contact 的 identity/swapped hard-min。
- 从两点派生中心、角度、宽度的几何函数。
- Contact 固定 80 px 的矩形厚度。
- Contact parser、schema 和 `<grasp>` 结构 token 语义。
- Contact 的 endpoint error 作为主要诊断指标。

Contact 的端点交换 Pair Loss 只能解决：

```text
(x1,y1,x2,y2) == (x2,y2,x1,y1)
```

它不能解决 angle token 的圆周接缝。Grasp Rect 使用独立的 wrapped angle CE，不能把 Contact 的 identity/swapped 分支改名后直接复用。

## 3. 输出协议和结构 token

### 3.1 使用独立结构块

第一版新增独立结构 token：

```text
<ref>grasp pose</ref><grasp_rect><cx><cy><theta_bin><width_bin></grasp_rect>
```

负样本形式：

```text
<ref>grasp pose</ref><grasp_rect>none</grasp_rect>
```

不要复用 Contact 的 `<grasp>...</grasp>`。虽然两种输出都是四个数，但槽位语义完全不同。共用结构 token 会产生以下问题：

- 同一 token 的第三、第四槽在 Contact 中是第二个接触点，在 Grasp 中却是角度和宽度。
- MTP mask 无法仅根据 start/end token 判断应使用交换损失还是周期角度损失。
- Fast 解码无法可靠决定使用 `decode_contact_pair()` 还是 `decode_grasp_rect()`。
- task-token adapter 会把两种结构梯度混在一起，增加旧 Contact checkpoint 被覆盖的风险。

模型配置新增：

```text
grasp_rect_task_token_ids = [grasp_rect_start_token_id, grasp_rect_end_token_id]
```

新 token 仍采用小型 input/output delta adapter，不解冻完整 embedding 或 lm_head。初始化可使用 box 与 Contact 结构 token 向量的确定性均值，但必须记录初始化来源和 token ID。

### 3.2 六槽定义

```text
slot 0: <grasp_rect>
slot 1: center_x token
slot 2: center_y token
slot 3: theta token
slot 4: width token
slot 5: </grasp_rect>
```

PBD shift、MTP mask 和结构验证必须以完整六槽为单位。任何缺槽、多块、旧 `<box>`、Contact `<grasp>` 或坐标 token 泄漏到块外，都判为结构非法。

### 3.3 API 输出不暴露编码细节

原始模型文本可以保留四个 token 值用于复现，但服务 API 应返回有明确单位的字段：

```json
{
  "center_1000": [500, 420],
  "center_pixels": [640, 302],
  "angle_degrees_image": 27.5,
  "angle_radians_image": 0.479966,
  "opening_width_pixels": 184.2,
  "opening_width_diagonal_normalized": 0.1255,
  "gripper_depth_pixels": 40.0,
  "rectangle_points_pixels": [
    566.9, 242.0,
    730.3, 327.0,
    713.1, 362.0,
    549.7, 277.0
  ]
}
```

`rectangle_points_pixels` 必须来自共享几何函数，不能由模型另外生成。

## 4. 参数编码

模型沿用 LocateAnything 的 1001 个连续坐标 token。设：

```text
B = 1001
coord token value in {0, 1, ..., 1000}
image size = (W, H)
image diagonal D = sqrt(W^2 + H^2)
```

### 4.1 中心坐标

训练编码：

```text
cx_token = round(cx_px / W * 1000)
cy_token = round(cy_px / H * 1000)
```

推理解码：

```text
cx_px = cx_token / 1000 * W
cy_px = cy_token / 1000 * H
```

编码前先校验中心在图像范围内。禁止先 clip 再伪装成合法 GT；被 clip 的标注会系统性堆积在 0 或 1000。

### 4.2 角度

RealVLG 使用图像坐标系中的 degree，图像 y 轴向下。二维平行夹爪具有 180 度周期，因此先规范化：

```text
theta_180 = theta_degrees mod 180, range [0, 180)
```

使用 1001 个均匀圆周 bin：

```text
theta_token = floor(theta_180 / 180 * B + 0.5) mod B
theta_degrees = theta_token / B * 180
```

该定义有以下性质：

- `0 degrees` 可以精确编码。
- `theta` 与 `theta + 180 degrees` 编码相同。
- 0 与 1000 是圆周相邻 bin，不是物理角度的两端。
- 最大量化误差约为 `90 / 1001 = 0.0899 degrees`。

禁止使用 `theta / 180 * 1000` 后把 180 保留为独立端点，因为 0 和 180 是同一个抓取方向。训练和损失需要显式处理 0/1000 seam。

### 4.3 宽度

宽度按原图对角线归一化：

```text
width_normalized = width_px / D
width_token = round(width_normalized * 1000)
width_px = width_token / 1000 * D
```

采用对角线而不是 `W` 或 `H`，可以保持不同纵横比图像上的统一尺度，并与 Contact 现有宽度诊断一致。

边界语义：

- `width_normalized` 必须严格大于 `minimum_width_diagonal`。
- 不定义物理 `maximum_gripper_opening`。
- 训练 GT 必须能被当前 token 域无损表达至量化精度，即 `width_token` 位于 `[0,1000]`。
- `width > D` 的 GT 不允许静默 clip。Phase 0 必须先统计；若真实合法样本存在，应调整全局编码尺度并重新生成全部数据，而不是把它解释成物理不可抓。
- 对四个角点均位于图像内的普通矩形，长边不会超过图像对角线，因此预期 RealVLG GraspNet_VLG 不需要额外宽度尺度。

`width_token <= 1000` 是表示域，不是夹爪硬件最大开口。论文和结果中不得混淆两者。

## 5. 确定性 4 角点（8 标量）矩形

### 5.1 官方公式

设中心：

```text
c = (cx, cy)
theta_rad = theta_degrees * pi / 180
u = (cos(theta_rad), sin(theta_rad))
v = (-sin(theta_rad), cos(theta_rad))
half_width = width / 2
half_depth = gripper_depth / 2
```

RealVLG Grasp 官方默认：

```text
gripper_depth = 40.0 pixels
```

四个角点顺序必须固定为：

```text
p0 = c - half_width * u + half_depth * v
p1 = c + half_width * u + half_depth * v
p2 = c + half_width * u - half_depth * v
p3 = c - half_width * u - half_depth * v
```

展平结果：

```text
[p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y]
```

这与官方 `rect_to_points8()` 的顺序一致。所有 converter、evaluator、service、visualization 和 collision 代码必须调用同一个共享实现。

### 5.2 必须满足的几何不变量

- 四点均值等于输入中心。
- `distance(midpoint(p1,p2), midpoint(p0,p3)) == width`。
- `distance(p0,p1) == width`。
- `distance(p1,p2) == gripper_depth`。
- polygon area 等于 `width * gripper_depth`，允许浮点误差。
- `theta` 与 `theta + 180 degrees` 生成同一个 polygon，角点顺序可以循环或反向等价。
- `points8_to_rect(rect_to_points8(params))` 在角度模 180 后应恢复原参数至量化误差。
- 生成函数不得依赖随机数、设备、batch 顺序或 Shapely 的顶点重排。

### 5.3 不做隐式裁剪

生成的角点可以落到图像外。官方 evaluator 没有要求四个角点全部在图像内，因此：

- 不在几何函数中 clip 角点。
- 不因为 polygon 部分越界就把结构输出判为非法。
- 单独计算 `outside_ratio_2d` 并报告。
- 部署策略可以在官方指标之外拒绝高越界预测，但必须单独命名为 deployment filter。

## 6. 合法性定义

### 6.1 结构合法

正样本输出必须满足：

- 恰好一个完整 `<grasp_rect>...</grasp_rect>` 块。
- 块内恰好四个连续坐标 token。
- 块外没有坐标 token。
- 没有同时出现 `none` 和坐标。
- start/end token 与 Grasp Rect 任务一致，不能是 `<box>` 或 Contact `<grasp>`。

负样本输出必须恰好为：

```text
<grasp_rect>none</grasp_rect>
```

### 6.2 几何合法

将 token 解码到原图像素后，正样本必须满足：

- `W > 0` 且 `H > 0`。
- 中心、角度和宽度均为有限数。
- `0 <= cx <= W` 且 `0 <= cy <= H`。
- 角度经过统一映射后位于 `[0,180)`。
- `width / D > minimum_width_diagonal`，使用严格大于。
- `width * gripper_depth` 大于数值 epsilon，polygon 非退化。

明确不检查：

- 真实夹爪最大开口。
- 物理碰撞、深度或可达性。
- 四个角点是否全部在图像内。
- 中心是否位于 target mask 内。该项先作为审计指标。

结构合法但几何不合法的预测记为 `geometry_invalid`，在 strict mIoU/gAcc 中按 0。不要把宽度强制抬到最小值后继续评测。

### 6.3 共享验证函数

训练转换、parser、Fast 解码、Slow 后验证和 evaluator 应共享等价的验证规则。建议提供：

```python
validate_grasp_rect_tokens(values, image_size, minimum_width_diagonal)
validate_grasp_rect_geometry(center, theta, width, image_size, minimum_width_diagonal)
```

测试必须证明所有入口在同一组边界样本上给出相同结论，特别是 `width == minimum` 必须失败。

## 7. 数据准备

### 7.1 新增转换脚本

新增：

```text
training/scripts/convert_realvlg_grasp.py
```

输入使用 RealVLG 元数据中的：

```json
{
  "image_path": "...",
  "description": "...",
  "grasps": [
    [x0, y0, x1, y1, x2, y2, x3, y3]
  ]
}
```

训练和官方评测数据必须分开生成：

- 训练数据允许按配置使用多个 frame。
- 官方 seen/similar/novel 评测严格按官方 `evaluation/dataset.py` 的实际范围和首帧规则。
- 当前官方代码实际使用 seen `scene_0100...0129`、similar `0130...0159`、novel `0160...0189`，每个 scene 只加载 `0000.json`。
- 官方文件顶部注释与实际 `range()` 存在不一致，必须以可执行代码为准并固定样本 ID 清单。

### 7.2 从 8 标量 GT 恢复四参数

按照官方 `points8_to_rect()`：

```text
corners = points8.reshape(4, 2)
center = mean(corners)
left_edge_center  = mean(corners[0,3])
right_edge_center = mean(corners[1,2])
width = norm(right_edge_center - left_edge_center)
theta = atan2(corner1.y - corner0.y, corner1.x - corner0.x) in degrees
theta = theta mod 180
```

同时计算 GT 矩形短边：

```text
depth = norm(corner2 - corner1)
```

Phase 0 必须报告 depth 分布。模型协议只输出四参数，预测阶段固定 depth 为 40 px。如果 GT depth 大量偏离 40 px，这是官方协议自身的信息损失，不能通过修改 `width` 补偿。

### 7.3 候选校验

每个原始 `grasps` 候选必须检查：

- 恰好 8 个可转为 float 的有限值。
- reshape 后 polygon 有效且面积大于 epsilon。
- 按官方角点顺序恢复的 width 严格大于最小非退化阈值。
- center、theta、width 可成功编码为四个 token。
- `points8 -> rect -> points8` 的 polygon IoU 和参数回环误差可记录。
- 原始 polygon 与重建的固定 40 px polygon 分开保存，不允许互相覆盖。

第一轮只审计、不硬过滤：

- center 在 target mask 内的比例。
- 四角越界比例和 polygon outside ratio。
- GT depth 与 40 px 的偏差。
- polygon 自交或角点顺序异常比例。
- width 超出 token 表示域的比例。

审计后才能决定是否过滤明显损坏的标注。所有过滤原因必须进入 stats，不能静默删除。

### 7.4 去重、主候选和 multi-GT

一个对象可能有大量 `grasps`。沿用 Contact 的“一个输出、多合法 GT”原则：

1. 对原始 8 标量 polygon 做数值校验。
2. 用 polygon IoU、中心距离、模 180 角度差和宽度差删除重复或近重复候选。
3. 在可靠 mask 可用时先排除明确不安全候选；mask 不完整时只记录 unknown。
4. 用原始 GT polygon 的平均 IoU 选择 medoid 主候选。
5. 并列时按编码后的 `(cx, cy, theta_bin, width_bin)` 字典序确定性打破平局。
6. multi-GT 阶段使用 `1 - polygon_iou` 的 FPS，最多保留 `K=8`。

主候选用于 conversation 中的 NTP 目标，其余候选只用于 MTP multi-GT hard-min。不要把每个 grasp 展开为重复图像样本，否则候选多的对象会支配训练。

### 7.5 JSONL 格式

正样本建议：

```json
{
  "sample_id": "GraspNet_VLG:scene_0000:...:14",
  "dataset": "GraspNet_VLG",
  "task_type": "grasp_rect",
  "image_width": 1280,
  "image_height": 720,
  "gripper_depth_pixels": 40.0,
  "grasp_rect_candidates": [
    [501, 420, 153, 126]
  ],
  "grasp_rect_candidates_pixels": [
    [641.2, 302.4, 27.5, 184.7]
  ],
  "grasp_rectangles_pixels": [
    [568.0, 242.0, 731.0, 327.0, 714.0, 362.0, 551.0, 277.0]
  ],
  "candidate_collision_2d": [0.0],
  "candidate_outside_2d": [0.0],
  "collision_valid": false,
  "collision_detail": "instance masks not declared exhaustive",
  "conversations": [
    {
      "from": "human",
      "value": "Predict one stable 2D rectangular grasp pose for the described target."
    },
    {
      "from": "gpt",
      "value": "<ref>grasp pose</ref><grasp_rect><501><420><153><126></grasp_rect>"
    }
  ],
  "image": "scenes/.../rgb/0000.png"
}
```

官方评测数据额外保存完整未压缩 GT：

```text
evaluation_grasp_rectangles_pixels
evaluation_protocol = "realvlg_graspnet_official"
evaluation_only = true
```

`--max-candidates` 只能限制训练候选，不能截断 evaluator 的 GT 集合。

### 7.6 负样本

沿用 Contact 的保守定义：

- `no_target`：目标确实不在图中。
- `ungraspable`：数据源明确保证在当前二维协议下没有合法 grasp。
- `ambiguous`：语言无法唯一定位目标，第一版排除并单独审计。

RealVLG 的有限候选集合不能证明候选穷尽，因此不能因为所有已标注 grasp 都碰撞或退化就自动生成 `ungraspable`。第一版可以只训练正样本和可靠 `no_target`。

## 8. 数据桥接和训练 batch

### 8.1 task type

新增：

```text
grasp_rect
grasp_rect_negative
```

不要复用 `grasp_contact` 的 task code。建议扩展成稳定枚举，并在 checkpoint 中保存映射版本，避免后续插入任务导致旧整数 code 失配。

### 8.2 packed 字段

正样本输出 fixed-K：

```text
grasp_rect_candidates:       [1, K, 4] int64
grasp_rect_candidate_mask:   [1, K] bool
grasp_rect_positive_mask:    [1] bool
grasp_rect_image_size:       [1, 2] float32, [W, H]
grasp_rect_gripper_depth:    [1] float32
grasp_rect_mtp_slot_mask:    [num_mtp_tokens] bool
candidate_collision_2d:      [1, K] float32
candidate_outside_2d:        [1, K] float32
collision_valid:             [1] bool
```

四个参数都使用坐标 token，但 slot mask 必须知道其语义。不能仅记录“这是坐标”，否则 loss 无法对 angle 使用圆周统计，也无法对 width 使用最小阈值。

### 8.3 conversation 和候选一致性

loader 必须断言：

- `grasp_rect_candidates[0]` 与 assistant 文本中的四 token 逐值一致。
- 正样本恰好一个完整 `<grasp_rect>` 块。
- 负样本恰好一个 `<grasp_rect>none</grasp_rect>`。
- `gripper_depth_pixels` 为正有限数，官方协议数据固定为 40。
- 每个 token 是 `[0,1000]` 内整数。
- 每个候选经过共享几何验证后 width 严格大于最小阈值。

## 9. 损失函数

### 9.1 基线 CE

Phase 1 和 Phase 2 先使用现有 NTP + MTP token CE，不开启多候选或几何损失。四槽顺序固定，不存在 Contact 的端点交换分支。

该基线必须先证明模型能记住格式和四个 token。如果基础 CE 无法在 64 样本上过拟合，不能用几何 loss、加大学习率或 RL 掩盖数据桥接错误。

### 9.2 pose-aware MTP CE 和 wrapped angle CE

Phase 1 和 Phase 2 使用 hard CE 验证数据桥接与结构收敛。进入 pose-aware 阶段后，中心和宽度继续使用普通完整词表 CE，角度改用圆周 wrapped marginal CE。

设：

```text
B = 1001
r = 1
N_r(t) = {(t+j) mod B | -r <= j <= r}
```

角度损失定义为：

```text
CE_theta_wrapped(t) = -log sum(
    P_full_vocab[coord_start_token_id + b] for b in N_r(t)
)
```

这里使用完整词表 softmax 中对应角度坐标 token 的概率，不能先只在 1001 个坐标 token 内重新归一化。否则模型把概率分给文字或错误结构 token 时不会受到惩罚。

主线固定 `r=1`，只容忍一个量化 bin 的圆周邻域。`r=0` 必须退化为普通 target token CE，用于数值和梯度等价测试。不得为了提高 token accuracy 任意扩大半径。

单 GT 时：

```text
L_pose_ce = mean(CE_cx, CE_cy, CE_theta_wrapped, CE_width)
```

multi-GT 时，对每个合法候选计算完整四槽和，再选最小项：

```text
L_pose_ce = min_k mean(
    CE_cx[k], CE_cy[k], CE_theta_wrapped[k], CE_width[k]
)
```

必须对完整 pose 选同一个候选，不能让中心来自 GT A、角度来自 GT B、宽度来自 GT C。否则会组合出数据集中不存在的矩形。

wrapped CE 不是 Contact Pair Loss。它对所有角度使用相同的圆周邻域规则，因此 0 的相邻 bin 包括 1000，普通中间角度也只容忍左右各一个量化 bin。训练同时报告 exact top-1 和 circular-within-1 accuracy，不能只用后者掩盖角度分布变宽。

### 9.3 soft pose

中心和宽度可使用条件坐标概率的期望：

```python
p = coordinate_logits.softmax(dim=-1)
values = torch.arange(1001, device=p.device).float()
cx_soft = (p_cx * values).sum(-1) / 1000
cy_soft = (p_cy * values).sum(-1) / 1000
width_soft = (p_width * values).sum(-1) / 1000
```

角度不能使用普通线性 soft-argmax。0 和 1000 在角度圆周上相邻，线性平均会错误落到约 90 度。使用 double-angle circular mean：

```python
phi = 2 * math.pi * torch.arange(1001, device=p.device) / 1001
angle_cos = (p_theta * phi.cos()).sum(-1)
angle_sin = (p_theta * phi.sin()).sum(-1)
angle_resultant = torch.sqrt(angle_cos.square() + angle_sin.square())
```

`phi` 已经对应物理角度的 `2 * theta`，天然实现 180 度周期。

### 9.4 几何辅助损失

恢复到原图尺度后计算：

```text
L_center = SmoothL1(pred_center_px, gt_center_px) / image_diagonal
L_width  = SmoothL1(pred_width_px, gt_width_px) / image_diagonal
L_angle  = 1 - dot(pred_double_angle_unit, gt_double_angle_unit)
```

角度圆周分布的 resultant 太小时，预测方向没有明确集中度。几何门控要求：

- 四槽完整词表中的 coordinate mass 均超过阈值。
- 条件坐标熵低于阈值。
- angle resultant 超过阈值。
- GT width 和预测 width 都严格大于最小非退化阈值时才启用 angle loss。

center 和 width loss 不因预测 width 太小而关闭，否则模型可以通过预测退化宽度逃避负责恢复宽度的梯度。

可在后续消融中增加 corner loss：

```text
L_corner = swap/cyclic-order-aware SmoothL1(
    rect_to_points8(pred_pose),
    rect_to_points8(gt_pose)
) / image_diagonal
```

第一版不默认启用 corner loss。它与 center/angle/width 高度相关，且错误的角点对应会在 180 度等价方向上制造不必要梯度。

### 9.5 Contact Pair Loss 和可选跨任务一致性

现有 Contact Pair Loss：

```text
min(
    CE(x1,y1,x2,y2),
    CE(x2,y2,x1,y1)
)
```

不进入 Grasp 主线。Grasp 的四槽没有端点排列，`theta` 的问题是圆周拓扑而不是 tuple 交换。

独立 Grasp checkpoint 验收完成后，可以单独实验 Grasp/Contact 跨任务一致性。只有在数据审计证明两个候选可可靠匹配后，才允许从 Contact 点对派生：

```text
theta_contact = atan2(y2-y1, x2-x1) mod 180
width_contact = norm(p2-p1)
```

并增加：

```text
L_cross_angle = 1 - cos(2 * (theta_grasp - theta_contact))
L_cross_width = SmoothL1(width_grasp, width_contact) / image_diagonal
```

如果 `grasps` 与 `contact_points` 不能逐候选对应，只能先做集合级 polygon 匹配或 hard-min，不能直接 `zip()`。该实验产生独立联合 checkpoint，不覆盖 Grasp 和 Contact 两个单任务基线。

### 9.6 loss reduction

沿用 Contact 已验证的 numerator/denominator 设计：

- base CE 恢复为有效 token numerator。
- pose CE 以四个参数槽恢复 numerator。
- 几何项保留逐 grasp tensor。
- 四卡和梯度累积使用整个 accumulation window、所有 data-parallel rank 的全局 denominator。
- DDP world-size 补偿只做一次。
- 空正样本 rank 返回图连通的零 loss，不能提前退出导致 collective 不一致。

当单 GT、pose weight 为 1、wrapped radius 为 0 且几何权重为 0 时，新 pose-aware CE 必须与被替换的原始四槽 CE 数值和梯度近似一致。这是启用 `r=1` 前的硬门槛。

## 10. 联合 top-k 解码

### 10.1 为什么需要联合排名

四个槽独立 argmax 可能产生：

- 合法中心配上接近 0 的退化宽度。
- angle 槽的非坐标 token 概率高但坐标子词表内部仍有伪 argmax。
- 四个单槽都高概率，但组合后 polygon 退化。
- `none` 与坐标结构竞争时选出不完整块。

因此 Fast 和 Hybrid 必须在四槽 top-k 的笛卡尔积上联合选取完整 pose。

### 10.2 候选生成

每槽保留 `K=4` 作为初始默认：

```text
cx candidates       K
cy candidates       K
theta candidates    K
width candidates    K
total combinations  K^4 = 256
```

候选分数：

```text
score = log P(cx) + log P(cy) + log P(theta) + log P(width)
```

不默认添加中心偏好、宽度先验或角度先验。任何数据先验都必须作为可关闭的 rerank 消融，不能污染 raw PBD 基线。

### 10.3 联合合法性 mask

对每个组合统一解码到像素空间并检查：

```text
center_valid = finite(cx, cy) and 0 <= cx <= W and 0 <= cy <= H
angle_valid  = finite(theta) and 0 <= canonical_theta < 180
width_valid  = finite(width) and width / D > minimum_width_diagonal
area_valid   = width * gripper_depth > epsilon
valid        = center_valid and angle_valid and width_valid and area_valid
```

不加入最大夹爪开口检查。也不要求四角全部位于图像内。

将非法组合 score 设为 `-inf`，从其余候选中选择总 log probability 最大者。如果全部非法：

- Fast 返回结构化 decode error，不得把 width clamp 到最小值。
- Hybrid 可以回退到受约束 AR，但 AR 完成后仍必须通过同一个几何验证。
- Slow 完成文本生成后同样使用共享 parser 和后验证，不能享有更宽松规则。

### 10.4 确定性和平局

同分时使用稳定规则：

1. 更高未取整 log score。
2. 更高 coordinate mass 的最小槽值。
3. 编码 tuple `(cx, cy, theta_bin, width_bin)` 字典序更小。

选择完成后才调用一次 `rect_to_points8()`。不要为每个候选生成并缓存角点作为模型输出，也不要让 Shapely 参与最终顶点顺序。

### 10.5 none 判定

`none` 只能在任务明确允许负样本时出现。结构槽按以下优先级处理：

- start token 必须是 `<grasp_rect>`。
- 内容首槽在 `none` 与 coordinate vocabulary 间比较。
- 选择 `none` 后立即强制 end token，其余 MTP 槽填 null。
- 选择 coordinate 后四个参数槽都必须由 coordinate vocabulary 产生。

应分别报告 false-none、missed-none 和正样本 grasp recall，避免负样本加入后全局拒绝塌缩。

## 11. Hybrid 和 AR 约束

新增 `geometry_type="grasp_rect"`，与 `bbox` 和 `contact` 并列。

AR 槽位约束：

```text
after ref_end: force grasp_rect_start
slot 1: none or coordinate token
slot 2: coordinate token, unless none branch already closed
slot 3: coordinate token
slot 4: coordinate token
slot 5: force grasp_rect_end
```

Hybrid 接受 Fast 结果前至少检查：

- start/end frame。
- 四个槽的 coordinate mass。
- 每槽条件熵。
- 联合候选存在合法组合。
- 最终共享几何验证通过。

坐标质量失败时回退 AR；几何失败时也回退一次。AR 仍失败则返回 invalid，不无限重试。

## 12. 服务层设计

### 12.1 新模式

建议新增：

```text
grasp_rect
```

prompt 只要求最终结构化答案，不要求输出 chain-of-thought。服务端提示与官方评测 prompt 可以不同，但任务语义必须一致。

### 12.2 parser

新增严格 parser：

```text
parse_grasp_rect_output(text, image_width, image_height)
```

返回状态：

```text
ok
none
invalid_structure
invalid_geometry
```

parser 流程：

1. 验证恰好一个完整块。
2. 解析四个 token 或 `none`。
3. 解码中心、角度、宽度。
4. 调用共享合法性验证。
5. 调用共享 `rect_to_points8()`。
6. 构建 API schema。

### 12.3 schema

建议新增 `GraspRectangle`，不要把字段塞进 `GraspContact`：

```text
center_1000
center_normalized
center_pixels_float
center_pixels
angle_token
angle_degrees_image
angle_radians_image
opening_width_token
opening_width_pixels
opening_width_diagonal_normalized
gripper_depth_pixels
rectangle_points_pixels_float
rectangle_points_pixels
collision_2d_status
collision_ratio_2d
outside_ratio_2d
clearance_pixels_2d
collision_detail
```

`LocateResponse` 增加独立 `grasp_rectangles` 列表，避免客户端根据字段是否存在猜任务类型。

### 12.4 visualization

- 按确定性四点顺序绘制 polygon。
- 绘制中心小十字和长边方向线。
- 不显示 token bin 等内部编码说明。
- Contact 与 Grasp Rect 使用不同颜色和图例。
- 原图、resize 后图和 API 返回角点必须有像素级一致性测试。

## 13. 二维碰撞和越界

Grasp Rect 已经确定完整二维 polygon，因此可直接计算：

```text
collision_ratio_2d = area(R intersect Mobs) / area(R)
outside_ratio_2d   = area(R outside image) / area(R)
clearance_px_2d    = min distance(R, Mobs)
```

约束与 Contact 相同：

- `Mobs` 必须排除当前 target mask。
- 只有实例覆盖被证明完整时 `collision_valid=True`。
- mask 缺失或不完整返回 `unknown`，不能当作 free。
- 官方 RealVLG mIoU/gAcc 不加入碰撞判定。
- collision-aware gAcc 作为独立表格。

与 Contact 不同的是，Grasp Rect polygon 不需要再假设 80 px 扫掠厚度；它使用官方四参数加固定 40 px depth 直接生成。

## 14. 严格评测协议

### 14.1 官方兼容路径

新增：

```text
training/scripts/evaluate_realvlg_grasp.py
```

必须能复现：

- 官方 scene 范围和 `0000.json` 首帧。
- 官方 `gripper_depth=40 px`。
- 与全部 GT `grasps` 比较。
- 以最大 IoU GT 为匹配项。
- 同一匹配项计算角度误差。
- `IoU > 0.25 && angle < 30 degrees` 的 gAcc。

### 14.2 修正官方问题

官方 evaluator 存在两个需要兼容但不能作为主指标的问题。

第一，解析失败样本不会加入 `iou_list/acc_list`，导致打印的 mIoU/gAcc 是仅合法输出条件下的指标。新评测器必须同时输出：

```text
format_valid_rate
mIoU_valid
gAcc_corrected_valid
mIoU_strict
gAcc_corrected_strict
```

strict 分母包含全部有 GT 的正样本，解析失败、`none` 和几何非法均按 0。checkpoint 选择只使用 corrected strict。

第二，官方 `angular_diff()` 根据数值绝对值是否小于 `pi` 猜测 degree/radian，但调用方已经传入 degree。接近 0 度的合法预测可能被误当 radian。修正角度误差统一使用：

```python
delta = (pred_theta_deg - gt_theta_deg) % 180.0
angle_error_deg = min(delta, 180.0 - delta)
```

可以额外输出 `gAcc_official_buggy_valid` 用于旧日志对账，但不得用于选模型。

### 14.3 诊断指标

每个 split 至少报告：

```text
format_valid_rate
positive_grasp_output_rate
mIoU_valid / mIoU_strict
gAcc_corrected_valid / gAcc_corrected_strict
gAcc_official_buggy_valid
center_error_pixels
center_error_diagonal_normalized
angle_error_degrees_mod_180
width_error_pixels
width_error_diagonal_normalized
corner_error_pixels_with_polygon_symmetry
outside_ratio_2d_geometry
geometry_invalid_rate
decode_fallback_rate
mean_latency_seconds
```

如果加入可靠负样本，再报告 `none precision/recall/F1` 和正样本 false-none rate。负样本不进入 polygon mIoU/gAcc 分母。

### 14.4 必须比较的基线

| 基线 | 目的 |
|---|---|
| RealVLG-R1 官方 Grasp 模型 | 官方任务基线 |
| 四参数 Slow/NTP | 精度基线 |
| 四参数 Fast/raw PBD | 验证并行吞吐 |
| 四参数 Fast/joint top-k | 验证结构化联合排名 |
| 四参数 Hybrid | 部署主路径 |
| 当前 Contact 模型转矩形 | 比较表示方式，不混入主表 |
| 单 GT medoid | 稳定监督基线 |
| multi-GT hard-min | 验证多解监督 |

Contact 转矩形只作为诊断基线：中心为两点中点，theta 为连线角度，width 为两点距离，depth 使用 Grasp 官方 40 px。它不是 Contact 官方 80 px 矩形，必须明确命名。

## 15. 分阶段实施与验收

不能跨过阶段门槛。跨阶段使用上一阶段通过验收的最佳 checkpoint 作为 `MODEL_PATH`，并建立新输出目录；同一阶段中断才使用 `RESUME_FROM_CHECKPOINT`。

### Phase 0：官方协议复现和数据审计

先不训练模型，完成 converter、几何函数和官方 evaluator 对账。

输出：

- scene/object/image 数量和 split 样本 ID hash。
- 每对象 grasp 候选数量。
- center、theta、width、depth 和 polygon area 分布。
- width 超 token 表示域比例。
- 退化、自交、角点顺序异常和越界比例。
- `points8 -> rect -> points8` IoU 分布。
- 固定 depth=40 与原始 GT depth 的差异。
- target/obstacle mask 完整率和 collision unknown 比例。

门槛：

- 官方评测样本 ID 与固定官方 snapshot 一致。
- 0 个缺失图像。
- 0 个静默 clip。
- 所有过滤都有原因计数。
- 合成输入上新 evaluator 与官方 evaluator 的 polygon IoU 一致。
- corrected 与 official buggy 角度指标差异有显式测试。
- 至少人工可视化 200 个随机 GT 和 50 个边界样本。

### Phase 1：64 样本过拟合

配置：

- 只用正样本。
- 每对象一个 medoid 主候选。
- 无数据增强。
- 原始 NTP + MTP CE。
- 无 multi-GT、几何、collision 或 negative loss。

门槛：

- 训练格式合法率 `>= 99%`。
- 四参数 token top-1 accuracy `>= 95%`。
- width 合法率 `100%`。
- Fast 完整六槽率 `>= 99%`。
- `miou_oracle_ratio >= 0.95`，即训练集 polygon mIoU 达到固定 40 px depth
  表示上限的 95%。
- `theta` seam 样本没有集中预测到 90 度。

### Phase 2：标准 PBD SFT

使用 Grasp Rect positive 和 grounding replay，仍为单 GT、原始 CE，并输出到独立 Grasp checkpoint。Contact checkpoint 不参与本阶段继续训练，也不被覆盖。可以从通过验收的 Contact checkpoint 与 LocateAnything 基座各跑一次短初始化 A/B，但两条实验都必须另存为 Grasp 权重，固定数据和步数后按 Grasp 验证指标选择，不能凭训练 loss 猜测。

门槛：

- Fast 和 Slow 都有稳定非零 corrected strict gAcc。
- Fast 格式合法率 `>= 98%`。
- grounding replay 指标下降不超过预先冻结的容忍值。
- 原有 Contact checkpoint hash 保持不变，并继续作为独立基线评测。
- center、theta、width 分布不塌缩到常数或少数 bin。
- Novel split 不因只优化 Seen 而持续恶化。

### Phase 3：pose-aware MTP CE

仍只用一个主候选，分两步启用 `L_pose_ce`，不同时开启几何损失：

1. `angle_wrap_radius=0`，验证新旧四槽 CE 数值和梯度等价。
2. 等价测试通过后固定 `angle_wrap_radius=1`，启用 wrapped angle CE。

门槛：

- 单 GT、`r=0` 条件下新旧 CE 数值和梯度近似一致。
- 全局 denominator 单卡/四卡/梯度累积等价测试通过。
- Fast corrected strict gAcc 不低于 Phase 2。
- `r=1` 后 angle seam 子集改善或保持。
- exact top-1 与 circular-within-1 accuracy 都被记录，预测分布没有无界变宽。

### Phase 4：中心、周期角度和宽度损失

从 Phase 3 的 `r=1` 最优 checkpoint 开始，将 center、double-angle 和 width 几何权重按累计正样本 grasp block 数从 0 ramp 到目标值。一次只增加这一组变量，不同时打开 multi-GT 或负样本。

门槛：

- center、angle、width 验证误差至少两项改善。
- corrected strict gAcc 不下降。
- 几何梯度贡献不超过基础损失的预设比例。
- angle resultant 和 geometry active rate 保持稳定。
- 最小宽度附近没有大量被强行推到同一 width bin。

### Phase 5：multi-GT 和可靠负样本

顺序：

1. 开启最多 K=8 的 multi-GT hard-min。
2. 验证稳定后加入少量可靠 `no_target`。
3. 最后才考虑数据源明确提供的 `ungraspable`。

门槛：

- multi-GT 后 strict gAcc 提升或保持。
- candidate assignment churn 可解释，没有每 step 大幅随机切换。
- `none` precision/recall 同时改善。
- 正样本 false-none rate 不超过阈值。
- 不出现全局输出 `none` 的拒绝塌缩。

### Phase 6：二维碰撞约束

仅在 mask 完整性审计通过后：

1. 先做安全候选过滤消融。
2. 再做低权重 outside/collision 连续 loss。
3. 最后才考虑二维 reward RL。

普通官方 strict gAcc 与 collision-aware strict gAcc 分开报告。RL 不能用于修复 parser、token 编码、角度 seam 或数据错配。

## 16. 训练监控

每个 logging step 至少记录：

```text
loss_total / loss_base / loss_pose_ce
loss_center / loss_angle / loss_width / loss_corner
loss_theta_hard / loss_theta_wrapped
cx_top1 / cy_top1 / theta_exact_top1 / theta_circular_within1 / width_top1
pose_four_slot_top1
coordinate_mass_min / coordinate_entropy_max
angle_resultant / geometry_active_rate
pred_center_mean_std
pred_angle_histogram_18bins
pred_width_mean_std
minimum_width_reject_rate
joint_decode_valid_combination_rate
hybrid_fallback_rate
positive / negative / grounding_replay samples
samples_per_optimizer_step
gradient_norm / learning_rate
```

固定验证集还需记录：

```text
format_valid_rate
mIoU_strict / gAcc_corrected_strict
center / angle / width error
angle_seam_subset_gAcc
geometry_invalid_rate
outside_rate
grounding replay metric
```

立即停止并回滚的信号：

- 格式合法率连续下降。
- width 大量集中到 0、最小合法 bin 或 1000。
- theta 大量集中到单一方向，或 seam GT 被预测成约 90 度。
- wrapped CE 下降但 exact theta top-1 持续恶化，说明容忍邻域正在掩盖分布变宽。
- 几何 loss 下降但 strict gAcc 明显下降。
- multi-GT assignment 高频随机切换。
- grounding replay 持续退化。
- Fast 合法但 Slow 非法，或反向出现规则不一致。
- collision loss 改善但普通 gAcc 和 oracle collision-aware gAcc 同时变差。

## 17. 测试计划

### 17.1 几何单元测试

- 水平、垂直、斜向矩形的 8 标量精确值。
- `theta`、`theta+180` 和负角度的 polygon 等价。
- area、中心、长边、短边不变量。
- `points8_to_rect` 回环。
- width 等于最小阈值时失败，略大于时成功。
- 超出图像的角点不被共享几何函数裁剪。
- 非有限数、负宽度和零宽度失败。
- angle 0/1000 bin 的圆周邻接。

### 17.2 数据测试

- 官方 split scene 范围和首帧规则。
- 原始 8 标量 GT 保留，不被训练 K 截断。
- medoid/FPS 确定性。
- conversation 与候选 0 严格一致。
- 无静默 clip、无跨 scene 图像/mask 合并。
- depth 分布和 round-trip stats 输出。

### 17.3 loss 测试

- 单 GT pose CE 与原四槽 CE 等价。
- wrapped radius `r=0` 与 hard angle CE 数值和梯度等价。
- `r=1` 的 target 0 邻域严格为 `{1000,0,1}`，target 1000 邻域严格为 `{999,1000,0}`。
- wrapped angle CE 使用完整词表概率，错误文字/结构 token 质量会增加损失。
- multi-GT 必须完整选择一个 pose。
- 角度 `0` 与 `179.9` 的周期损失接近 0。
- 普通线性 soft-argmax 不得出现在 angle 路径。
- 小预测宽度时 width recovery loss 仍有梯度。
- 低 coordinate mass/高熵时几何门控关闭。
- 单卡、四卡、不均匀 packing 和梯度累积等价。

### 17.4 解码测试

- 独立 top-1 的 width 退化时，联合 top-k 选择次优合法组合。
- width 必须严格大于最小阈值。
- 不存在最大夹爪开口参数或硬件上限拒绝分支。
- 全部组合非法时 Fast 返回 decode error。
- Hybrid 只回退一次，AR 结果仍走共享验证。
- 选定四参数后生成的 8 标量顺序固定。
- `none` 与正样本结构分支互斥。

### 17.5 evaluator 测试

- 合成矩形与官方 IoU 一致。
- 最大 IoU 和 angle 必须来自同一个 GT。
- invalid 在 strict 指标中按 0。
- corrected degree 角度与 official buggy 分开。
- GT K 不受训练 `max_candidates` 影响。
- valid、strict、negative 和 collision 分母互不混淆。

## 18. 文件修改清单

计划新增：

```text
training/scripts/convert_realvlg_grasp.py
training/scripts/evaluate_realvlg_grasp.py
training/scripts/train_realvlg_grasp.sh
training/configs/grasp_anything_realvlg_grasp.env
training/data/realvlg_grasp_meta.example.json
training/Eagle/Embodied/eaglevl/train/grasp_rect.py
tests/test_grasp_rect_geometry.py
tests/test_grasp_rect_parser.py
tests/test_grasp_rect_decode.py
tests/test_grasp_rect_loss.py
tests/test_realvlg_grasp_pipeline.py
tests/test_grasp_rect_data_bridge.py
```

计划修改服务层：

```text
src/locate_anything_service/prompts.py
src/locate_anything_service/schemas.py
src/locate_anything_service/parser.py
src/locate_anything_service/grasp_geometry.py
src/locate_anything_service/collision_2d.py
src/locate_anything_service/model.py
src/locate_anything_service/visualization.py
src/locate_anything_service/cli.py
tests/test_api.py
tests/test_model.py
tests/test_prompts.py
```

计划修改训练层：

```text
training/Eagle/Embodied/eaglevl/train/arguments.py
training/Eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py
training/Eagle/Embodied/eaglevl/model/locany/modeling_locateanything.py
training/Eagle/Embodied/eaglevl/utils/locany/generate_utils.py
training/Eagle/Embodied/eaglevl/utils/locany/modeling_locateanything.py
training/scripts/validate_training_meta.py
training/scripts/validate_phase_transition.py
```

Eagle 训练代码如果继续由项目 patch 管理，还需新增或扩展：

```text
training/patches/locateanything-grasp-rect.patch
```

启动脚本必须保留双向 `git apply --check` 和 revision 检查，防止服务器重新 clone Eagle 后静默丢失任务代码。

## 19. 风险和对应措施

### 19.1 角度 seam

风险：0 和 180 度物理等价，但 hard CE 将相邻圆周 bin 当作普通类别。

措施：角度编码使用 1001-bin 圆周；主线固定使用 `r=1` wrapped marginal CE；几何损失使用 double-angle circular mean；同时监控 exact top-1、circular-within-1 和 seam 子集。

### 19.2 官方角度单位错误

风险：小于 `pi` 的 degree 被误判为 radian，旧日志与正确结果不一致。

措施：主指标明确使用 degree 模 180 公式；官方错误实现只保留兼容字段。

### 19.3 固定 40 px depth 的信息损失

风险：GT 短边不全是 40 px，但模型只输出四参数，无法恢复每个 GT depth。

措施：Phase 0 报告 GT depth 分布和 round-trip IoU；主表严格按官方 40 px；额外诊断不改变官方协议。

### 19.4 width 语义误读

风险：把像素长边当作真实夹爪毫米开口，或错误新增最大硬件行程。

措施：API 字段始终带 `pixels`/`diagonal_normalized` 后缀；文档、schema 和指标不出现无标定的物理单位；不增加最大开口约束。

### 19.5 多 GT 平均成非法姿态

风险：四个参数分别匹配不同 GT。

措施：完整 pose hard-min；候选选择基于同一四槽总损失；评测的 IoU 和 angle 也绑定同一 GT。

### 19.6 任务互相覆盖

风险：Grasp Rect、Contact 和 bbox 都使用六槽，错误共用 token/mask 导致旧能力退化。

措施：独立 start/end token、task code、parser、schema、loss、decoder 和 checkpoint；主线只保留 grounding replay，不更新 Contact 权重。可选联合训练只能写入第三套独立 checkpoint。

### 19.7 条件指标虚高

风险：非法输出被官方 evaluator 跳过，模型格式越差，valid mIoU 反而可能越高。

措施：checkpoint 只按 strict corrected 指标选择；valid 指标仅用于与官方日志对账。

### 19.8 图像外矩形

风险：硬裁剪改变矩形中心、方向和 IoU；完全不报告又会掩盖部署问题。

措施：官方指标不裁剪；独立报告 outside ratio；部署过滤单独评估。

### 19.9 单任务权重与联合权重混淆

风险：把可选的 Grasp+Contact 联合实验覆盖到已有 Contact checkpoint，或用联合权重结果冒充 RealVLG 官方单任务对比。

措施：Grasp 和 Contact 主线使用独立 checkpoint、输出目录和模型标识；联合训练只能产生第三套独立 checkpoint；三套结果分表报告。任何从 Contact 初始化的 Grasp 实验在第一次 optimizer step 前后都记录 source checkpoint hash 和新输出目录。

## 20. 完成定义

- 官方 Grasp evaluator 的固定 commit、样本清单和兼容结果可复现。
- 参数编码、角度圆周、宽度量化和 8 标量矩形均有单元测试。
- 64 样本过拟合达到格式、四槽和 polygon 门槛。
- `r=1` wrapped angle CE 和 double-angle loss 均通过 seam 单元测试与子集验收。
- Grasp 主线保存为独立 checkpoint，原 Contact checkpoint hash 不变。
- Fast 使用一次联合 top-k 生成合法完整 pose。
- width 严格大于最小非退化阈值。
- 没有新增 RealVLG 未定义的最大夹爪开口限制。
- 所有模式在输出后调用同一个确定性 8 标量生成函数。
- Slow、Fast、Hybrid 使用相同 parser 和几何验证。
- 单卡、四卡和梯度累积 loss/gradient 等价测试通过。
- seen、similar、novel 均报告 corrected strict 指标。
- official buggy、valid、strict、negative 和 collision 指标分开。
- grounding replay 退化处于预先冻结的容忍范围。
- 可选联合 checkpoint 不覆盖 Grasp 或 Contact 单任务权重。
- multi-GT、几何损失和 collision 增强都有独立消融。
- checkpoint 包含新 token、adapter、配置、推理代码和 phase acceptance。
- 训练修改有 patch、测试和可复现配置。

## 21. 推荐实施顺序

```text
1. 固定官方 commit、样本清单和 evaluator 对账
2. 实现四参数编码、points8_to_rect 和 rect_to_points8
3. 完成 RealVLG Grasp 数据审计与 converter
4. 完成严格 evaluator 和 corrected strict 指标
5. 新增独立 grasp_rect token、parser、schema 和 API
6. 新增六槽 PBD data bridge 与 64 样本过拟合
7. 全量单 GT 标准 PBD SFT，加 grounding replay 并保存独立 Grasp checkpoint
8. 接入 pose-aware MTP CE，以 r=0 验证等价后固定 r=1 wrapped angle CE
9. 接入中心、double-angle 和宽度 ramp
10. 实现联合 top-k 合法性排名和 Hybrid 回退
11. 开启 multi-GT hard-min
12. 加入可靠 no_target
13. 完成 oracle-mask collision 评测和候选过滤消融
14. 视结果决定是否尝试低权重 collision loss 或二维 reward RL
15. 主线完成后再审计 Grasp/Contact 候选对应并消融跨任务一致性
```

每一步只引入一个主要变量。任何阶段未通过门槛，都回到上一阶段最佳 checkpoint 排查数据、mask、slot shift、角度编码或 reduction，不通过同时调大学习率和训练步数绕过失败。
