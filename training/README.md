# LocateAnything 单卡 LoRA 历史工作区

> 本文档记录旧 VOC 单卡实验，不是当前 grasp-contact 训练入口。当前环境使用
> `bash scripts/bootstrap.sh --training`，固定 Eagle revision 和完整 contact patch；
> 参见仓库根目录的 `DEPLOYMENT.md` 与 `GRASP_CONTACT_README.md`。

此目录针对当前 RTX 3090 Ti 24GB 主机准备。脚本默认只打印命令，不会自动启动训练。

## 文件结构

```text
training/
├── scripts/train_locateanything_lora.sh
├── configs/locateanything_voc_lora.env
├── patches/locateanything-single-gpu-sdpa.patch
├── data/
│   └── voc_train_meta.example.json
├── outputs/
└── logs/
```

## 准备要求

1. 官方代码位于仓库内的 `training/Eagle`。
2. 独立训练环境位于仓库内的 `training/.venv`。
3. 训练 JSONL 位于 `data/voc_train.jsonl`。
4. 将 `data/voc_train_meta.example.json` 复制为 `data/voc_train_meta.json`。
5. 正式训练前停止 LocateAnything 推理服务，释放约 8GB 显存。

旧 VOC 环境尚未准备时也应使用固定 bootstrap，不要 clone 浮动的 Eagle main：

```bash
bash scripts/bootstrap.sh --training --venv training/.venv
```

训练环境固定版本见 `pyproject.toml`；需要其他 CUDA wheel 时通过 `TORCH_INDEX_URL` 覆盖下载源，但不能绕过 `validate_training_environment.py` 的版本检查。

## 安全预览

默认运行 `probe` 模式，但不会真正训练：

```bash
bash scripts/train_locateanything_lora.sh
```

脚本会检查：

- 官方 Eagle 代码是否存在；
- 模型和 metadata 是否存在；
- 推理服务是否仍在占用 GPU；
- 可用显存是否至少为 22GB；
- SDPA 兼容补丁能否干净应用。

## 50-step 探测

完成数据和环境准备后才执行：

```bash
RUN_MODE=probe \
AUTO_STOP_INFERENCE=1 \
CONFIRM_TRAIN=YES \
bash scripts/train_locateanything_lora.sh
```

探测配置：LoRA rank 16、序列长度 1536、梯度累积 8、50 step，不保存 checkpoint。

## 正式训练

只有探测确认峰值显存和 loss 正常后再执行：

```bash
RUN_MODE=train \
AUTO_STOP_INFERENCE=1 \
CONFIRM_TRAIN=YES \
bash scripts/train_locateanything_lora.sh
```

正式配置：LLM LoRA rank 32、序列长度 2048、batch 1、梯度累积 16、2500 step、BF16、gradient checkpointing、SDPA。视觉编码器、MLP projector 和原始 LLM 参数全部冻结。

可通过环境变量临时覆盖配置，例如：

```bash
RUN_MODE=train TRAIN_MAX_SEQ_LENGTH=1536 TRAIN_LORA_RANK=16 bash scripts/train_locateanything_lora.sh
```

没有设置 `CONFIRM_TRAIN=YES` 时，脚本永远只显示最终命令。
