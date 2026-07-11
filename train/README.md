# LocateAnything 单卡 LoRA 训练工作区

此目录针对当前 RTX 3090 Ti 24GB 主机准备。脚本默认只打印命令，不会自动启动训练。

## 文件结构

```text
train/
├── scripts/train_locateanything_lora.sh
├── configs/locateanything_voc_lora.env
├── patches/locateanything-single-gpu-sdpa.patch
├── data/
│   └── voc_train_meta.example.json
├── outputs/
└── logs/
```

## 准备要求

1. 官方代码位于 `/home/charles/WORK/train/Eagle`。
2. 独立训练环境位于 `/home/charles/WORK/train/.venv`。
3. 训练 JSONL 位于 `data/voc_train.jsonl`。
4. 将 `data/voc_train_meta.example.json` 复制为 `data/voc_train_meta.json`。
5. 正式训练前停止 LocateAnything 推理服务，释放约 8GB 显存。

官方代码尚未准备时：

```bash
git clone https://github.com/NVlabs/Eagle.git /home/charles/WORK/train/Eagle
python3.11 -m venv /home/charles/WORK/train/.venv
source /home/charles/WORK/train/.venv/bin/activate
pip install --upgrade pip
pip install -e /home/charles/WORK/train/Eagle/Embodied
```

训练环境还需要安装与驱动兼容的 PyTorch CUDA 版本。当前推荐保持 `torch==2.8.0`、CUDA 12.8 wheel，与已验证推理环境一致。

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
