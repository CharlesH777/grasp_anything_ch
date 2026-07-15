# LocateAnything 单卡 LoRA 微调适配报告

> 本报告记录在 RTX 3090 Ti 24GB 单卡上对 NVIDIA LocateAnything-3B 进行 Pascal VOC LoRA 微调时，相对官方 Eagle 代码库所做的全部适配变更。

---

## 1 背景与约束

### 1.1 官方目标环境

NVIDIA 官方 Eagle 仓库的 LocateAnything 训练流程面向以下环境设计：

| 维度 | 官方假设 |
|------|----------|
| GPU | 8 × H100 80GB（单节点）或 2 × 8 H100（多节点） |
| 架构 | Hopper / Blackwell |
| 注意力后端 | MagiAttention（`magi`），仅支持 Hopper/Blackwell |
| 训练数据 | LocateAnything-Data：138M 查询，785M boxes，12M 图片 |
| 评测数据 | Rex-Omni EvalData（COCO / LVIS / ScreenSpot-Pro 等） |
| 分布式 | Slurm 或 torchrun 多机多卡 |
| 优化器分片 | DeepSpeed ZeRO Stage 1/2 |
| 序列长度 | 16K–32K tokens |

### 1.2 本项目实际环境

| 维度 | 实际 |
|------|------|
| GPU | 1 × RTX 3090 Ti 24GB（Ampere, compute capability 8.6） |
| 驱动 / CUDA | 580.159.03 / CUDA 12.8 (torch 2.8.0+cu128) |
| 注意力后端 | PyTorch SDPA（无 MagiAttention，FlashAttention-2 不可用于 packed sequence） |
| 训练数据 | Pascal VOC 2007+2012 trainval：16,551 图，47,223 目标，42,183 样本 |
| 评测数据 | VOC2007 test 固定 1000 张子集 |
| 分布式 | 单进程（`nproc_per_node=1`） |
| 优化器分片 | 无 DeepSpeed |

### 1.3 核心矛盾

1. **注意力后端不可用**：官方代码在多处硬编码 `flash_attention_2` 和 `magi`，Ampere 架构不支持 MagiAttention，而 MoonVit 的 SDPA 实现存在 packed sequence bug。
2. **显存限制**：24GB vs 80GB × 8 卡，序列长度和批次大小需大幅缩减。
3. **数据域不匹配**：官方用 138M 通用定位数据，本项目目标是 Pascal VOC 类别定位。
4. **流程安全性**：单卡单机环境下推理服务和训练共用 GPU，需要显存隔离机制。

---

## 2 注意力后端适配

### 2.1 问题诊断

官方训练脚本默认 `--attn_implementation magi`，MagiAttention 仅支持 Hopper/Blackwell 架构。官方文档明确指出：

> Non-H-series GPU: Use `--attn_implementation sdpa` with `--max_seq_length 4096`. SDPA does not support long-context (16K+) training.

然而，仅设置 `--attn_implementation sdpa` 并不够，因为官方代码在训练脚本和模型定义中有多处**硬编码覆盖**：

| 文件 | 官方硬编码 | 位置 |
|------|-----------|------|
| `locany_finetune_magi_stream.py` | `config.vision_config._attn_implementation = 'flash_attention_2'` | config 构建段（3 处） |
| 同上 | `text_config._attn_implementation = 'magi'` | LLM 加载段 |
| 同上 | `locateanything_config._attn_implementation = 'magi'` | 模型组装段 |
| 同上 | `vision_config._attn_implementation = 'flash_attention_2'` | MoonVit 加载段 |
| `modeling_locateanything.py` | `config.vision_config._attn_implementation = 'flash_attention_2'` | 模型初始化 |
| `modeling_vit.py` `sdpa_attention()` | packed sequence 全序列方阵 mask 逻辑 | SDPA 注意力函数 |

这意味着即使命令行传入 `--attn_implementation sdpa`，代码运行时仍会覆盖为 `flash_attention_2` 或 `magi`。

### 2.2 补丁内容

文件：`train/patches/locateanything-single-gpu-sdpa.patch`

#### 2.2.1 训练脚本修改（`locany_finetune_magi_stream.py`）

**a) 分布式 launcher 默认值**

```python
# 官方
launcher = os.environ.get('LAUNCHER', 'slurm')
# 修改后
launcher = os.environ.get('LAUNCHER', 'pytorch')
```

理由：单机环境不使用 Slurm，默认改为 `pytorch` launcher。

**b) 注意力实现跟随参数（5 处）**

将所有硬编码的 `'flash_attention_2'` 和 `'magi'` 替换为 `model_args.attn_implementation`，使命令行参数 `--attn_implementation sdpa` 在整个模型构建链路中一致生效：

| 修改点 | 官方 | 修改后 |
|--------|------|--------|
| config 构建段 vision_config | `'flash_attention_2'` | `model_args.attn_implementation` |
| config 构建段日志 | `'Vision attn: flash_attention_2'` | `'Vision attn: {model_args.attn_implementation}'` |
| MoonVit 加载段 | `'flash_attention_2'` | `model_args.attn_implementation` |
| LLM 加载段 | `'magi'` | `model_args.attn_implementation` |
| LocateAnything 组装段 | `'magi'` | `model_args.attn_implementation` |

**c) 模型初始化硬编码删除（`modeling_locateanything.py`）**

```python
# 官方：在 __init__ 中强制覆盖
if config.vision_config.model_type == 'moonvit':
    config.vision_config._attn_implementation = 'flash_attention_2'  # ← 删除此行
    self.vision_model = MoonVitPretrainedModel(config.vision_config)
```

理由：该行会在模型从 checkpoint 恢复时覆盖已设置的注意力实现，导致 `sdpa` 设置失效。

#### 2.2.2 MoonVit SDPA 注意力修复（`modeling_vit.py`）

**问题**：官方 `sdpa_attention()` 对 packed sequence 构建全序列 `[1, L, L]` 布尔 mask，其中 L 是所有样本 token 的总和。在长序列下：

- 显存开销 O(L²)，当 L 较大时直接 OOM
- 全序列方阵 mask 在数学上也是错误的：不同 packed 样本之间的 token 会互相 attend

**官方实现**：

```python
seq_length = q.shape[0]  # 全部 packed token 总数
attention_mask = torch.zeros([1, seq_length, seq_length], ...)
for i in range(1, len(q_cu_seqlens)):
    attention_mask[..., q_cu_seqlens[i-1]:q_cu_seqlens[i],
                   q_cu_seqlens[i-1]:q_cu_seqlens[i]] = True
q = q.transpose(0, 1)
attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask)
```

**修改后实现**：

```python
# 按 cu_seqlens 分段，每段独立调用 SDPA
for index in range(1, len(q_cu_seqlens)):
    q_segment = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
    k_segment = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
    v_segment = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
    output = F.scaled_dot_product_attention(
        q_segment, k_segment, v_segment, dropout_p=0.0, is_causal=False)
    outputs.append(output.squeeze(0).transpose(0, 1))
return torch.cat(outputs, dim=0).reshape(q.shape[0], -1)
```

**改进点**：

1. **显存从 O(L²) 降到 O(Σlᵢ²)**，其中 lᵢ 是各段长度，大幅减少峰值显存
2. **正确性**：不同 packed 样本之间不再有跨段 attention
3. **健壮性**：添加了 `q_cu_seqlens` / `k_cu_seqlens` 为 None 时的默认处理和长度一致性校验

---

## 3 训练配置适配

### 3.1 训练参数对比

下表逐项对比官方 LoRA 脚本（`locate-anything-lora-visual-prompt.sh`）与本项目的正式训练配置：

| 参数 | 官方 LoRA | 本项目正式 | 变化 | 原因 |
|------|----------|-----------|------|------|
| `attn_implementation` | `magi` | `sdpa` | 后端替换 | H100 → 3090 Ti |
| `nproc_per_node` | 8 | 1 | 单卡 | 硬件限制 |
| `max_steps` | 5000 | 2500（实际 2736） | 缩减 | 数据量小 + 单卡速度慢 |
| `max_seq_length` | 16384 | 2048（实际 3072） | 8× 缩减 | 24GB 显存限制 |
| `max_num_tokens` | 16384 | 2048/3072 | 5-8× 缩减 | 同上 |
| `packing_buffer_size` | 32 | 4 | 8× 缩减 | 显存限制 |
| `gradient_accumulation_steps` | 1 | 16 | 16× 增大 | 补偿单卡 micro batch，保持有效 batch ≈16 |
| `use_llm_lora` | 64 | 32 | 缩减 | 减少可训练参数量，降低显存 |
| `warmup` | `warmup_steps=500` | `warmup_ratio=0.03`（≈75 步） | 比例化 | 总步数缩减，固定 500 步 warmup 不合理 |
| `logging_steps` | 1 | 5 | 放宽 | 减少日志开销 |
| `sample_log_interval` | 1 | 50 | 放宽 | 同上 |
| `save_steps` | 100 | 250 | 放宽 | 减少 checkpoint I/O |
| `save_total_limit` | 3 | 2 | 缩减 | 磁盘空间 |
| `deepspeed` | `zero_stage1_config.json` | 无 | 移除 | 单卡不需要 |
| `use_onelogger` | True | False | 关闭 | 无分布式环境 |
| `num_train_epochs` | 1 | 未设置 | — | 用 max_steps 控制 |
| `seed` / `data_seed` | 未设置 | 42 | 固定 | 可复现性 |
| `tf32` | 未设置 | True | 启用 | Ampere TF32 加速 |

### 3.2 额外环境变量

本项目在训练脚本中额外设置：

```bash
export PYTHONPATH="${EAGLE_ROOT}/Embodied${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb=128"
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
```

其中 `PYTORCH_CUDA_ALLOC_CONF` 的 `expandable_segments` 和 `max_split_size_mb=128` 用于减少 CUDA 显存碎片，在 24GB 限制下提高分配成功率。

### 3.3 训练流程安全机制

官方脚本使用 `set -x` 直接执行，本项目采用**多层安全防护**：

| 防护层 | 机制 |
|--------|------|
| Dry-run 默认 | 未设置 `CONFIRM_TRAIN=YES` 时只打印命令不执行 |
| 探测模式 | `RUN_MODE=probe`：50 步、rank 16、seq 1536、不保存 checkpoint |
| 显存检查 | `nvidia-smi` 查询 free memory，需 ≥ 22,000 MiB |
| 推理服务冲突 | 检测 `locate-anything.service` systemd 服务是否在运行 |
| 自动停服 | `AUTO_STOP_INFERENCE=1` 时自动停止推理服务，等待显存释放 |
| 显存轮询 | 30 秒超时轮询，每秒检查一次 free memory |
| 退出恢复 | trap EXIT：训练结束后自动重启被停止的推理服务 |
| 补丁检查 | 检测 SDPA 补丁是否已应用 / 能否干净应用 |
| 数据校验 | 调用 `validate_training_meta.py` 校验 JSONL 和 meta.json 格式 |

---

## 4 数据适配

### 4.1 数据来源与划分

| 数据集 | split | 用途 | 图片数 | 目标数 |
|--------|-------|------|--------|--------|
| VOC2007 | trainval | 训练 | 5,011 | — |
| VOC2012 | trainval | 训练 | 11,540 | — |
| 合计 | — | 训练 | 16,551 | 47,223 |
| VOC2007 | test | 评测 | 1,000（固定随机子集，种子 20260711） | — |

选择 **trainval** 而非仅 train 的原因：单卡训练数据量有限，trainval 比 train 多约 40% 的数据（16,551 vs 11,905），在不影响评测（test 独立）的前提下尽量增加训练样本。

VOC2007 test 共 4,952 张，本项目固定随机抽取 1,000 张用于评测，保证可复现。

### 4.2 坐标归一化方案

LocateAnything 使用 `[0, 1000]` 范围的离散整数坐标，共 1001 个坐标 token（`<0>` 到 `<1000>`）。Pascal VOC XML 中标注为像素级绝对坐标。转换需要把 `[0, width]` 和 `[0, height]` 映射到 `[0, 1000]`。

**本项目的归一化公式**（`convert_pascal_voc.py`）：

```python
def normalize_min(value: int, size: int) -> int:
    return max(0, min(1000, round((value - 1) * 1000 / size)))

def normalize_max(value: int, size: int) -> int:
    return max(0, min(1000, round(value * 1000 / size)))
```

| 边界 | 公式 | 含义 |
|------|------|------|
| xmin / ymin | `round((value - 1) * 1000 / size)` | 从 1-based 像素坐标转换，偏移 -1 对齐像素中心 |
| xmax / ymax | `round(value * 1000 / size)` | 直接比例缩放 |

**验证示例**（图片 `000009.jpg`，500×375）：

| 目标 | VOC 像素坐标 | 归一化结果 | JSONL 实际值 | 匹配 |
|------|------------|----------|-----------|------|
| horse | (69, 172, 270, 330) | (136, 456, 540, 880) | `<136><456><540><880>` | ✅ |
| person | (150, 141, 229, 284) | (298, 373, 458, 757) | `<298><373><458><757>` | ✅ |

**min/max 不对称设计的原因**：VOC XML 的 `xmin/ymin` 是 1-based（像素从 1 开始计数），`xmax/ymax` 是包含边界。使用 `(value - 1)` 对 min 端做偏移，max 端不做偏移，确保框边界正确对齐像素网格。

**逆向验证**（`[0,1000]` → 像素）：

```
horse: <136><456><540><880> → (68, 171, 270, 330)  原始: (69, 172, 270, 330)
```

逆变换后 x1/y1 有 1 像素偏差，这是 1-based → 0-based 的量化误差，在 IoU 计算中可忽略。

### 4.3 样本生成策略

对每张 VOC 图像生成 **两类训练样本**：

#### 4.3.1 Detection 样本（每图 1 条）

```
human: "Detect all objects in <image-1>."
gpt:   "<ref>horse</ref><box><136><456><540><880></box>
        <ref>person</ref><box><298><373><458><757></box>
        <ref>person</ref><box><568><533><654><883></box>
        <ref>person</ref><box><514><525><594><877></box>"
```

- 包含图中**全部** GT 目标
- 每个目标用 `<ref>label</ref><box><x1><y1><x2><y2></box>` 格式
- 同类别多个实例逐个输出（重复 `<ref>label</ref>`）

#### 4.3.2 Grounding 样本（每图 N 条，N = GT-positive 类别数）

```
human: "Locate all the instances that match the following description: horse."
gpt:   "<ref>horse</ref><box><136><456><540><880></box>"

human: "Locate all the instances that match the following description: person."
gpt:   "<ref>person</ref><box><298><373><458><757></box>
        <ref>person</ref><box><568><533><654><883></box>
        <ref>person</ref><box><514><525><594><877></box>"
```

- 每条只包含**一个类别**的全部实例
- 类别顺序按图中出现顺序（`dict.fromkeys` 去重保序）

#### 4.3.3 生成统计

| 统计项 | 数量 |
|--------|------|
| 唯一图片 | 16,551 |
| detection 样本 | 16,551（每图 1 条） |
| grounding 样本 | 25,632（每图 N 条，N = 类别数） |
| 总样本 | 42,183 |
| 总目标框 | 47,223 |

每图平均类别数 = 25,632 / 16,551 ≈ 1.55，即平均每张图有 1.55 个不同类别。

各类别 grounding 样本数（按频率降序）：

| 排名 | 类别 | 样本数 | | 排名 | 类别 | 样本数 |
|-----|------|-------|-|-----|------|-------|
| 1 | person | 6,469 | | 11 | bus | 627 |
| 2 | car | 1,990 | | 12 | boat | 704 |
| 3 | chair | 1,870 | | 13 | horse | 777 |
| 4 | dog | 1,727 | | 14 | motorbike | 785 |
| 5 | cat | 1,428 | | 15 | train | 813 |
| 6 | bird | 1,106 | | 16 | bicycle | 826 |
| 7 | sofa | 1,067 | | 17 | pottedplant | 841 |
| 8 | bottle | 1,030 | | 18 | diningtable | 904 |
| 9 | tvmonitor | 874 | | 19 | aeroplane | 916 |
| 10 | cow | 455 | | 20 | sheep | 423 |

`person` 类占 grounding 样本的 25.2%，存在类别不平衡。`sheep` 最少仅 423 条。

### 4.4 数据格式与官方规范对照

#### 4.4.1 JSONL 行格式

每行是一个自包含 JSON 对象，使用 ShareGPT 对话格式：

```json
{
  "conversations": [
    {"from": "human", "value": "Detect all objects in <image-1>."},
    {"from": "gpt", "value": "<ref>chair</ref><box><524><560><648><904></box>..."}
  ],
  "image": "VOCdevkit/VOC2007/JPEGImages/000005.jpg"
}
```

与官方 `DATA_PREPARATION.md` 规范完全一致：
- ✅ `conversations` 列表，`from` 字段为 `human`/`gpt`
- ✅ `image` 字段为相对路径（相对于 meta.json 中的 `root`）
- ✅ `<image-1>` 占位符
- ✅ `<ref>label</ref><box><x1><y1><x2><y2></box>` 标签格式

#### 4.4.2 坐标 token 格式

| 来源 | `<box>` 内部格式 | 示例 |
|------|----------------|------|
| 官方 `TRAINING.md` 示例 | 括号包裹 | `<box>(100,200,400,500)</box>` |
| 官方 `DATA_PREPARATION.md` 规范 | 独立坐标 token | `<box><100><200><400><500></box>` |
| 官方评测脚本正则解析 | 独立坐标 token | `<box>\s*<\s*(\d+...)>\s*...` |
| 官方 tokenizer 定义 | 1001 个 token `<0>`~`<1000>` | — |
| **本项目** | **独立坐标 token** | **`<box><136><456><540><880></box>`** |

注意：官方 `TRAINING.md` 中的示例用了括号格式 `(100,200,400,500)`，但这与 `DATA_PREPARATION.md` 规范、tokenizer 的 1001 个独立坐标 token 设计、以及评测脚本的正则解析模式**不一致**。本项目遵循 `DATA_PREPARATION.md` 规范和实际 tokenizer 设计，使用独立坐标 token 格式，与官方评测脚本兼容。

#### 4.4.3 Prompt 措辞与官方任务格式对照

官方 `DATA_PREPARATION.md` 定义了多种任务格式，本项目的两种样本对应：

| 本项目样本 | 官方任务类型 | 官方 prompt 模板 | 本项目 prompt |
|-----------|------------|----------------|-------------|
| detection | Object Detection | `Locate all the instances that matches the following description: car</c>person.</` | `Detect all objects in <image-1>.` |
| grounding | Phrase Grounding (Multiple) | `Locate all the instances that match the following description: people wearing hats.` | `Locate all the instances that match the following description: {label}.` |

关键区别：
- 官方 Object Detection 用 `matches`（单数动词）+ `</c>` 分隔多类别
- 官方 Phrase Grounding (Multiple) 用 `match`（复数动词）+ 单一类别描述
- 本项目 grounding 样本采用 Phrase Grounding (Multiple) 格式（`match` + 单一类别），与官方规范一致
- 本项目 detection 样本采用 `Detect all objects in <image-1>.` 而非官方 detection 格式，这是更简洁的全图检测 prompt

### 4.5 数据配方（meta.json）

```json
{
  "pascal_voc_2007_2012": {
    "root": "/absolute/path/to/grasp_anything/datasets/pascal-voc",
    "annotation": "/absolute/path/to/grasp_anything/training/data/voc_train.jsonl",
    "repeat_time": 1,
    "data_augment": false,
    "visual_prompt": false
  }
}
```

| 字段 | 值 | 官方默认 | 差异说明 |
|------|-----|---------|---------|
| `root` | VOC 数据集根目录 | — | 绝对路径 |
| `annotation` | JSONL 文件路径 | — | 绝对路径 |
| `repeat_time` | 1 | 1.0 | 不重复采样 |
| `data_augment` | false | false | 一致 |
| `visual_prompt` | false | — | 显式设为 false，不使用视觉提示 |

### 4.6 数据校验（`validate_training_meta.py`）

官方假设用户自行保证数据格式正确。本项目自建校验器，在训练启动前检查：

| 校验项 | 检查内容 |
|--------|---------|
| meta.json 结构 | 非空 JSON 对象 |
| 数据集名称 | 非空字符串 |
| `root` 字段 | 绝对路径且目录存在 |
| `annotation` 字段 | 字符串或字符串列表（支持多文件合并） |
| JSONL 文件存在性 | 绝对路径、文件存在、非空 |
| JSONL 格式 | 首行可解析为 JSON 对象 |
| conversations 字段 | 非空列表 |
| image 字段 | 至少包含 `image`/`image_list`/`data`/`video` 之一 |

### 4.7 数据质量保证

转换脚本中的完整性检查：

| 检查 | 处理 |
|------|------|
| 图片尺寸缺失或无效 | `raise ValueError`（拒绝转换） |
| 框坐标越界（`xmin > xmax` 或 `ymin > ymax`） | `raise ValueError`（拒绝转换） |
| 框坐标超出图片范围 | `raise ValueError`（拒绝转换） |
| 图片无目标对象 | 跳过该图，计入 `empty_images` 统计 |
| VOC2012 无 test split | 跳过 test split（不报错） |

### 4.8 额外数据划分文件

除了实际训练用的 `data/voc_train.jsonl`（trainval split），`data/splits/` 目录还保存了独立划分：

| 文件 | split | 样本数 | 用途 |
|------|-------|--------|------|
| `splits/voc_train.jsonl` | train | 21,098 | 备用（仅 train split） |
| `splits/voc_val.jsonl` | val | 21,085 | 备用（仅 val split） |
| `voc_train.jsonl` | trainval | 42,183 | **实际训练使用** |

这些划分文件可用于后续的 train/val 分离实验，但当前训练使用的是合并的 trainval。

---

## 5 评测适配

### 5.1 评测整体架构差异

官方评测体系基于 Rex-Omni EvalData，使用 DDP 分布式推理 + `fastevaluate` 库计算 COCO/LVIS AP，或使用 `other_metric.py` 计算 grounding 任务指标。本项目完全自建 `evaluate_voc2007.py`，在架构层面有以下差异：

| 维度 | 官方评测 | 本项目评测 |
|------|----------|-----------|
| 推理框架 | DDP 多卡分布式（`torchrun --nproc_per_node=8`） | 单卡串行 |
| 数据集 | COCO / LVIS / ScreenSpot-Pro / RefCOCOg / HierText 等 | VOC2007 test 固定 1000 张子集 |
| AP 计算库 | `fastevaluate`（第三方库，来自 Rex-Omni） | 自实现 `voc_ap()` + `box_iou()` |
| Grounding 指标 | `other_metric.py`：IoU ≥ 0.5 判定正确/错误（二值），或 point-in-bbox | 标准 VOC AP（多阈值 IoU 0.5–0.95） |
| 速度测量 | 解析推理日志中的 "Statistic Info" 行 | `time.perf_counter()` 逐图计时，排除前 5 张预热 |
| 图片缩放 | 支持 `--short_side_size` 可选缩放 | 不缩放，原图推理 |
| 断点续跑 | 无 | `load_completed()` 从 JSONL 已完成行续跑 |

### 5.2 查询策略差异（核心区别）

**这是评测协议中最重大的差异。**

**官方策略**：每张图发**一次查询**，将所有类别用 `</c>` 分隔符拼在一条 prompt 中：

```python
# 官方：一次查询所有类别
category_set_str = "</c>".join(categories)  # e.g. "person</c>car</c>bicycle"
user_text = f"Locate all the instances that matches the following description: {category_set_str}."
```

**本项目策略**：对每张图的每个 GT-positive 类别**分别发一次独立查询**：

```python
# 本项目：逐类别查询
for category in categories:
    prompt = f"Locate all the instances that match the following description: {category}."
    # 单独推理一次
```

**影响分析**：

| 维度 | 单次全类别查询（官方） | 逐类别查询（本项目） |
|------|----------------------|---------------------|
| 查询数/图 | 1 | N（GT-positive 类别数，平均约 1.5） |
| 吞吐量 | 高（每图只跑一次前向） | 低（多次前向） |
| 误报风险 | 高（模型可能输出未查询类别的框） | 低（只取与查询类别匹配的预测） |
| 公平性 | 模型需在单次输出中区分所有类别 | 每次只关注一个类别，任务更简单 |
| 指标性质 | 更接近真实检测场景 | 更接近条件定位（oracle 类别设定） |

**预测过滤**：本项目进一步过滤模型输出，只保留与当前查询类别匹配的预测框（`if item["label"] == category`），丢弃模型可能输出的其他类别框。官方不做此过滤，直接使用模型输出的所有框。

### 5.3 Prompt 措辞差异

| 来源 | prompt 文本 |
|------|-----------|
| 官方评测脚本 | `"Locate all the instances that matches the following description: {categories}."` |
| 官方训练脚本 | `"Locate all the instances that matches the following description: {categories}."` |
| 本项目评测脚本 | `"Locate all the instances that match the following description: {category}."` |
| 本项目训练数据 | `"Locate all the instances that match the following description: {label}."` |

差异：官方用 `matches`（单数），本项目用 `match`（复数）。这是训练数据和评测脚本中一致使用的不一致措辞，属于细微但实际存在的偏差。

### 5.4 生成参数差异

| 参数 | 官方检测评测 | 官方 grounding 评测 | 本项目评测 |
|------|------------|-------------------|-----------|
| `max_new_tokens` | 4096 | 8192 | **512** |
| `include_eos_token` | False | True | **未设置** |
| `n_future_tokens`（hybrid/fast） | 6 | 6 | 6 |
| `do_sample` | True | True | True |
| `temperature` | 0.7 | 0.7 | 0.7 |
| `top_p` | 0.9 | 0.9 | 0.9 |
| `repetition_penalty` | 1.1 | 1.1 | 1.1 |

`max_new_tokens` 差异显著：官方允许 4096–8192 tokens 的生成长度，本项目仅 512。由于本项目逐类别查询（每次只需输出一个类别的框），512 tokens 通常足够，但如果图片中某类别目标非常多，可能截断输出。

### 5.5 `pixel_values` 数据类型差异

| 来源 | `pixel_values` 处理 |
|------|-------------------|
| 官方评测脚本（`inference_compat.py`） | 仅 `.to(device)`，**不显式转换 dtype** |
| 官方 worker（`locateanything_worker.py`） | `.to(self.dtype)` 即 bfloat16 |
| 本项目评测脚本 | `.to("cuda", dtype=torch.bfloat16)` **显式转换** |
| 本项目推理服务 | `.to(pixel_dtype)` 其中 `pixel_dtype = bfloat16` |

官方评测脚本中 `pixel_values` 保持 processor 输出的默认 dtype（通常 float32），未显式转为 bfloat16。本项目在两处（评测脚本和推理服务）都显式转为 bfloat16。这可能导致数值精度上的细微差异。

### 5.6 种子与采样控制差异

| 维度 | 官方评测 | 本项目评测 |
|------|----------|-----------|
| 全局种子 | 不设置 | `torch.manual_seed(42)` + `random.seed(42)` |
| 逐查询种子 | 不设置 | `torch.manual_seed(42 + image_id * 100 + category_index)` |
| 目的 | — | 可复现性 + 避免相同图片相同类别产生相同采样 |

官方评测不设置任何种子，依赖 `do_sample=True` 的随机采样。本项目为每个 (image_id, category_index) 组合设置唯一种子，保证可复现性的同时避免完全确定的输出。

### 5.7 坐标校验范围差异

| 来源 | 坐标有效范围 |
|------|-----------|
| 官方评测脚本 | `0 <= x <= 10000`（允许超出 [0, 1000] 的异常值） |
| 本项目评测脚本 | `max(0.0, min(1000.0, x))`（严格 clamp 到 [0, 1000]） |

官方允许模型输出 [0, 10000] 范围内的坐标值（尽管训练数据是 [0, 1000]），不做修正。本项目将所有坐标严格 clamp 到 [0, 1000] 后再转换为像素坐标。

### 5.8 VOC `difficult` 标志处理

Pascal VOC 标注中每个目标有 `difficult` 标志（0 或 1），标记标注困难或模糊的目标。

| 维度 | 官方评测 | 本项目评测 |
|------|----------|-----------|
| difficult 处理 | 不涉及（COCO/LVIS 无此概念） | 遵循标准 VOC 协议：difficult 目标不计入 GT 总数，不参与 AP 计算，但匹配到的 difficult 框既不算 TP 也不算 FP |

这是本项目相对官方评测特有的逻辑，符合 Pascal VOC 标准 AP 定义。

### 5.9 AP 计算实现差异

**官方 COCO/LVIS**：使用 `fastevaluate` 库（来自 Rex-Omni 仓库），该库内部使用 `pycocotools` 进行标准 COCO AP 计算。

**官方 grounding 任务**：使用 `other_metric.py`，对 box_eval 任务使用 IoU ≥ 0.5 二值判定（correct/incorrect），对 point_eval 使用 point-in-bbox 或 point-in-mask 判定。**不计算 AP，只计算 accuracy。**

**本项目 VOC**：完全自实现 VOC AP：

```python
def voc_ap(recall, precision):
    """标准 VOC AP：先对 precision 做单调递减插值，再对 recall-prcision 曲线积分"""
    recall = [0.0] + recall + [1.0]
    precision = [0.0] + precision + [0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return sum(
        (recall[index] - recall[index - 1]) * precision[index]
        for index in range(1, len(recall))
        if recall[index] != recall[index - 1]
    )
```

- IoU 阈值：0.50, 0.55, ..., 0.95 共 10 个
- 排序依据：模型不输出置信度，使用稳定输出顺序作为 AP 排名依据
- 逐类别计算 AP 后取平均得到 mAP

### 5.10 标签归一化差异

| 维度 | 官方评测 | 本项目评测 |
|------|----------|-----------|
| 类别名匹配 | 直接使用 COCO/LVIS 类别名（如 `person`、`car`），不归一化 | 实现别名映射：`airplane→aeroplane`、`motorcycle→motorbike`、`tv→tvmonitor`、`dining table→diningtable`、`potted plant→pottedplant` |
| 空格/下划线 | 不处理 | 统一去除空格和下划线 |

VOC 类别名与常见英文名有差异（如 `aeroplane` vs `airplane`），模型可能输出任一形式。本项目实现了别名映射确保匹配，官方评测因使用 COCO 类别名（与模型训练数据一致）不需要此处理。

### 5.11 评测协议设计

**评测协议设计**：

1. 对每张图解析 XML 标注，获取 GT-positive 类别列表
2. 对每个 GT-positive 类别分别发一次单类别 grounding 查询
3. 解析模型输出中的 `<ref>label</ref><box>...` 标签，提取预测框
4. 坐标从 `[0, 1000]` 转换回像素，与 GT 做 IoU 匹配

**关键设计决策**：

| 决策 | 选择 | 原因 |
|------|------|------|
| 查询方式 | GT-positive 条件查询 | 不查询不存在类别，避免误报污染；衡量"已知类别名称条件下的定位能力" |
| AP 排序 | 稳定输出顺序 | LocateAnything 不输出原生置信度分数 |
| IoU 阈值 | 0.5–0.95 共 10 个 | 标准 COCO 协议 |
| 预热排除 | 前 5 张 | 排除 CUDA kernel JIT 和 KV cache 预热 |
| 断点续跑 | `load_completed()` | 支持中断后从 JSONL 续跑 |
| 标签归一化 | 别名映射 | `airplane→aeroplane`、`motorcycle→motorbike` 等 |

### 5.12 评测结果

#### 总体指标（VOC2007 test 1000 张子集）

| 指标 | 官方原模型 | LoRA step-1797 | 变化 |
|------|--------:|--------:|------:|
| mAP50:95 | 61.05% | **67.51%** | +6.46 pp |
| AP50 | 80.28% | **88.22%** | +7.94 pp |
| AP75 | 66.75% | **72.77%** | +6.02 pp |
| Micro Precision @ IoU50 | 81.89% | **88.91%** | +7.02 pp |
| Micro Recall @ IoU50 | 88.10% | **93.35%** | +5.25 pp |
| Micro F1 @ IoU50 | 84.88% | **91.08%** | +6.19 pp |

#### 性能指标

| 指标 | 官方原模型 | LoRA step-1797 | 变化 |
|------|--------:|--------:|------:|
| 图片吞吐 | 3.057 img/s | 2.110 img/s | -30.99% |
| 查询吞吐 | 4.686 query/s | 3.234 query/s | -30.99% |
| 平均延迟 | 0.327 s | 0.474 s | +44.90% |
| P50 延迟 | 0.245 s | 0.375 s | +52.86% |
| P95 延迟 | 0.730 s | 1.214 s | +66.29% |
| 峰值显存 | 7,830 MiB | 8,058 MiB | +229 MiB |

#### 每类别 AP50

| 类别 | 官方原模型 | LoRA step-1797 | 变化 |
|------|--------:|--------:|------:|
| aeroplane | 97.67% | 96.15% | -1.52 pp |
| bicycle | 82.90% | 91.19% | +8.29 pp |
| bird | 99.60% | 97.61% | -2.00 pp |
| boat | 46.38% | 86.31% | **+39.93 pp** |
| bottle | 54.78% | 69.75% | +14.97 pp |
| bus | 97.92% | 97.92% | 0.00 pp |
| car | 75.91% | 95.43% | **+19.52 pp** |
| cat | 98.82% | 98.82% | 0.00 pp |
| chair | 59.70% | 73.75% | +14.05 pp |
| cow | 83.12% | 86.77% | +3.64 pp |
| diningtable | 91.46% | 90.23% | -1.23 pp |
| dog | 97.07% | 100.00% | +2.93 pp |
| horse | 93.49% | 94.71% | +1.22 pp |
| motorbike | 75.14% | 86.48% | +11.34 pp |
| person | 71.75% | 82.49% | +10.74 pp |
| pottedplant | 46.96% | 59.97% | +13.01 pp |
| sheep | 76.93% | 80.85% | +3.92 pp |
| sofa | 81.52% | 95.67% | **+14.15 pp** |
| train | 87.45% | 89.06% | +1.61 pp |
| tvmonitor | 87.04% | 91.25% | +4.20 pp |

微调模型在 15 个类别上提升，2 个类别持平，3 个类别轻微下降。最显著提升出现在 boat (+39.93 pp)、car (+19.52 pp)、sofa (+14.15 pp)、bottle (+14.97 pp)、chair (+14.05 pp)。

### 5.13 训练收敛

Loss 从 step 5 的 2.23 下降至 step 2735 的 0.79，收敛平稳：

| Step | Loss | Learning Rate |
|-----:|-----:|-------------:|
| 5 | 2.2259 | 1.07e-06 |
| 230 | 0.9048 | 1.98e-05 |
| 455 | 0.9338 | 1.88e-05 |
| 680 | 0.8599 | 1.71e-05 |
| 905 | 0.8765 | 1.48e-05 |
| 1130 | 0.9054 | 1.20e-05 |
| 1355 | 0.8670 | 9.14e-06 |
| 1580 | 0.8850 | 6.31e-06 |
| 1805 | 0.8318 | 3.85e-06 |
| 2030 | 0.8379 | 3.85e-06 |
| 2255 | 0.8239 | 3.85e-06 |
| 2480 | 0.8344 | 3.85e-06 |
| 2705 | 0.8128 | 3.85e-06 |
| 2735 | 0.7859 | 3.85e-06 |

### 5.14 评测限制说明

1. **GT-positive 条件定位**：评测器针对图片中真实存在的类别发出查询，不衡量不存在类别的误报。
2. **无原生置信度**：LocateAnything 文本输出不包含每个框的置信度分数，AP 排序使用稳定输出顺序，不应与传统检测器论文中的标准 VOC Detection AP 直接横向比较。
3. **采样解码**：使用 `do_sample=True, temperature=0.7, top_p=0.9`，结果存在采样波动。

---

## 6 推理服务工程化

官方仅提供 `locateanything_worker.py`（Python API）和 `batch_infer.py`（CLI）。本项目额外构建了完整的生产级服务封装。

### 6.1 服务架构

```
┌──────────────┐    ┌──────────────────────────────────┐
│  Web UI      │───▶│  serve_ui.py (FastAPI 代理)       │
│  index.html  │    │  :8001  透明转发 /healthz, /v1/locate │
└──────────────┘    └──────────┬───────────────────────┘
                               │
                    ┌──────────▼───────────────────────┐
                    │  api.py (FastAPI 主服务)            │
                    │  :8000                             │
                    │  ┌────────────┐  ┌──────────────┐  │
                    │  │ config.py  │  │ model.py     │  │
                    │  │ Settings   │  │ Runtime      │  │
                    │  └────────────┘  └──────┬───────┘  │
                    │  ┌────────────┐  ┌──────▼───────┐  │
                    │  │ prompts.py │  │ parser.py    │  │
                    │  │ 9 种模式    │  │ 坐标解析     │  │
                    │  └────────────┘  └──────┬───────┘  │
                    │  ┌────────────┐  ┌──────▼───────┐  │
                    │  │ schemas.py │  │ visualization│  │
                    │  │ Pydantic   │  │ PIL 标注     │  │
                    │  └────────────┘  └──────────────┘  │
                    └────────────────────────────────────┘
```

### 6.2 注意力后端选择差异

官方提供了多种推理注意力后端，本项目只使用 SDPA：

| 后端 | 官方支持 | 本项目 | 说明 |
|------|---------|-------|------|
| `magi` | ✅（Hopper/Blackwell） | ❌ | MagiAttention，需要专用安装 |
| `la_flash` | ✅（非 Hopper GPU） | ❌ | 官方 HF release 携带的 `batch_utils/` + `kernel_utils/`，使用 FlashAttention varlen sparse range |
| `sdpa` | ✅ | ✅ | PyTorch 原生 SDPA |
| `eager` | ✅ | ❌ | PyTorch eager attention |

官方在非 Hopper GPU 上推荐使用 `la_flash`（batch_infer.py 的默认值即 `sdpa`，但 worker.py 默认 `la_flash`）。本项目在推理服务和评测脚本中均使用 `sdpa`，未启用 `la_flash` 运行时，主要原因是 `la_flash` 需要将 HF 模型仓库中的额外代码目录加入 `PYTHONPATH`，增加了部署复杂度。

### 6.3 batch 推理差异

| 维度 | 官方 | 本项目 |
|------|------|-------|
| batch 推理 | `batch_infer.py` 支持 batch_size > 1，使用 `batch_utils` 的 `generate_batch_hybrid()` | 不支持 batch，每次只处理 1 张图 |
| hybrid 调度 | `batch_utils` 中的 eager/hold_ar/ar_first/pipeline/adaptive 策略 | 直接调用模型 `.generate()` 的 `generation_mode` 参数 |
| 统计 | `get_last_hybrid_stats()` 返回 MTP/NTP 切换统计 | 仅记录总生成时间和 box 数量 |

### 6.4 关键设计

| 组件 | 设计 |
|------|------|
| 模型加载 | 延迟加载，`/healthz` 在模型加载完成后才返回 `status=ok` |
| 并发控制 | `threading.Lock` 串行访问模型，避免 KV cache 状态冲突 |
| Prompt 映射 | 9 种模式（detect / ground_single / ground_multi / ground_text / detect_text / gui_box / gui_point / point / raw）→ 官方 prompt 文本 |
| 坐标解析 | 正则解析 `<box>` / `<ref>` / `<c>` 标签，同时输出三套坐标：`[0,1000]` 离散值、`[0,1]` 归一化、像素坐标 |
| 可视化 | PIL 画绿框 + 红点 + 标签，输出 base64 PNG |
| 请求限制 | 自定义 ASGI middleware，Content-Length + 实际 body 双层防护 |
| 部署 | Docker Compose（GPU + shm 8GB + HF cache volume）/ systemd user service |
| 模型版本 | revision 固定到 commit hash，不跟随 `main` |

---

## 7 变更文件索引

### 7.1 本项目新增文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `train/patches/locateanything-single-gpu-sdpa.patch` | 补丁 | SDPA 适配 + packed attention 修复 |
| `train/scripts/train_locateanything_lora.sh` | 脚本 | 安全优先训练入口 |
| `train/configs/locateanything_voc_lora.env` | 配置 | 全部超参 |
| `train/scripts/convert_pascal_voc.py` | 工具 | VOC XML → JSONL 转换 |
| `train/scripts/validate_training_meta.py` | 工具 | 训练数据格式校验 |
| `train/scripts/evaluate_voc2007.py` | 工具 | VOC2007 评测脚本 |
| `train/data/voc_train_meta.json` | 数据 | 训练配方 |
| `train/data/voc_train.jsonl` | 数据 | 训练数据（42,183 条） |
| `train/data/voc_train_stats.json` | 数据 | 训练统计 |
| `locate-anything/src/locate_anything_service/` | 服务 | 推理服务源码 |
| `locate-anything/compose.yaml` | 部署 | Docker Compose |
| `locate-anything/Dockerfile` | 部署 | 容器镜像 |
| `locate-anything/scripts/serve_ui.py` | UI | Web UI 代理 |
| `locate-anything/scripts/index.html` | UI | Web 界面 |
| `locate-anything/launch/locate-ui.sh` | 启动 | 一键启动 |

### 7.2 官方原始文件（未修改，作为对照）

| 文件 | 说明 |
|------|------|
| `train/Eagle/Embodied/shell/locate-anything-lora-visual-prompt.sh` | 官方 LoRA 训练脚本（对照基准） |
| `train/Eagle/Embodied/shell/locate-anything-streaming.sh` | 官方全参数 SFT 脚本 |
| `train/Eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py` | 官方训练主程序（被补丁修改） |
| `train/Eagle/Embodied/eaglevl/model/moon_vit/modeling_vit.py` | 官方 ViT 模型（被补丁修改） |
| `train/Eagle/Embodied/eaglevl/model/locany/modeling_locateanything.py` | 官方模型定义（被补丁修改） |
| `train/Eagle/Embodied/evaluation/inference_detection_ddp.py` | 官方检测评测推理脚本 |
| `train/Eagle/Embodied/evaluation/inference_grounding_ddp.py` | 官方 grounding 评测推理脚本 |
| `train/Eagle/Embodied/evaluation/inference_compat.py` | 官方推理兼容层 |
| `train/Eagle/Embodied/evaluation/metrics/coco_lvis_metric.py` | 官方 COCO/LVIS AP 指标 |
| `train/Eagle/Embodied/evaluation/metrics/other_metric.py` | 官方 grounding 指标 |
| `train/Eagle/Embodied/locateanything_worker.py` | 官方推理 worker API |
| `train/Eagle/Embodied/document/TRAINING.md` | 官方训练文档 |
| `train/Eagle/Embodied/document/DATA_PREPARATION.md` | 官方数据准备文档 |
