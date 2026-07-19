# Grasp Anything 远端运行手册

> 历史机器记录，仅用于追溯 2026-07-16 的实验环境。新机器部署不要复制本文固定路径；使用根目录 `DEPLOYMENT.md` 和 `.env.example`。

## 登录与 Codex

使用 `/home/charles/login_zhenghengcao.sh` 登录，工程目录为 `/home/zhenghengcao/grasp_anything`。Codex 配置、API key 和规则已同步到 `~/.codex/`，CLI 为 `~/.local/bin/codex`，版本 `0.144.4`。登录后运行 `codex login status` 检查状态。不要提交 `~/.codex/auth.json`。

## 远端路径

Python 为 `/home/zhenghengcao/la_env/bin/python`；Eagle 为 `/home/zhenghengcao/grasp_anything/training/Eagle/Embodied`；模型为 `/home/zhenghengcao/LocateAnything-3B`；数据根目录为 `/data2/zhenghengcao/datasets/RealVLG-11B/GraspNet_VLG`；准备数据、输出和日志分别在 `/data2/zhenghengcao/grasp_anything_2d/prepared`、`outputs`、`logs`。

训练前执行：`export PYTHONPATH=/home/zhenghengcao/grasp_anything/training/Eagle/Embodied:$PYTHONPATH`。

## 数据状态

当前存在 `contact_train_grasp_v2.jsonl`、`contact_overfit64_grasp_v2.jsonl`、`overfit64_grasp_v2_meta.json` 和 contact-only 的 `full_contact_meta.json`。后者缺少 grounding replay，因此不能作为正式稳定 Phase 2 meta；脚本现在会在启动前拒绝这种配置。

## 训练配置

配置文件是 `training/configs/grasp_anything_realvlg_contact.remote.env`，默认 `CONTACT_PHASE=overfit`。训练冻结 ViT、LLM 基座和完整 lm_head，更新 LLM LoRA、视觉 projector/MLP 与 grasp task-token adapter。`pair` 才开启交换不变损失，`geometry` 固定一个候选并逐步开启 center/angle/width，只有 `multigt` 读取最多 8 个候选。

## 启动与监控

```bash
cd ~/grasp_anything
export PYTHONPATH=$PWD/training/Eagle/Embodied:$PYTHONPATH
CONFIG_FILE=$PWD/training/configs/grasp_anything_realvlg_contact.remote.env bash training/scripts/train_realvlg_contact.sh
```

后台日志写入 `/data2/zhenghengcao/grasp_anything_2d/logs/contact-training.log`。用 `nvidia-smi`、`pgrep -af 'torch.distributed.run|locany_finetune_magi_stream'` 和 `tail -f` 监控。日志必须出现 `Running training`；`contact_loss_enabled=True` 只应出现在 pair/geometry/negative/multigt，overfit/sft 应为 False。

## 停止、恢复与磁盘

停止使用 `pkill -TERM -f locany_finetune_magi_stream.py` 和 `pkill -TERM -f torch.distributed.run`。DeepSpeed checkpoint 的 optimizer 状态可能比模型权重大数倍，建议 `save_total_limit=2` 或 `3`；推理备份不需要 `global_step*/`，断点续训必须保留。

## 全量切换条件

只有在 `full_contact_meta.json` 存在、meta 校验通过、overfit64 二维指标正常且磁盘空间充足后，才把 `META_PATH` 切换到该文件。切换后先小步试跑并检查日志，再长期训练。

## 当前审查结论（2026-07-16）

- 远端当前没有正在运行的训练进程。
- `CONTACT_MAX_CANDIDATES=8` 已同步；overfit、sft、pair、geometry、negative 阶段固定使用 1 个候选，只有 multigt 阶段读取配置中的 8。
- `full_contact_meta.json` 已存在并通过校验，JSONL 为 `contact_train_grasp_v2.jsonl`。
- 该 full meta 目前是 contact-only，缺少 grounding replay；新启动门禁会拒绝把它用于正式 SFT。
- Eagle contact 代码已集成但 Git 工作树不是干净状态；训练脚本会检测已集成代码并跳过重复 patch。
- 当前全量 JSONL 没有 `grasp_contact_negative` 样本。需要拒答训练时，必须先重新生成并检查负样本数量。

正式启动前按顺序验收：

1. Phase 1：`overfit64_grasp_v2_meta.json`、`CONTACT_PHASE=overfit`、300 steps，确认 `<grasp>` 格式率和四坐标输出。
2. Phase 2：先创建含显式 `sampling_weight` 且 grounding replay 不低于 15% 的正式 meta，再用通过验收的 Phase 1 checkpoint 作为 `MODEL_PATH`，切换 `CONTACT_PHASE=sft`。
3. 依次运行 `pair`、`geometry`、单候选 `negative`，最后运行 `multigt`，每次只改变一个阶段参数。
4. 检查 `contact_positive_samples`、`geometry_active_count`、LoRA 梯度和 finite loss。
5. checkpoint 保存成功并完成二维评估后，再开启长期训练。

Phase 1 验收通过后，Phase 2 使用 checkpoint 权重启动新的优化器和数据流：

```bash
cd /home/zhenghengcao/grasp_anything/training/scripts
bash start_phase2_sft.sh
```

默认从 `overfit/checkpoint-300` 加载并运行 `13000` 步；但当前 contact-only
`full_contact_meta.json` 会被门禁拒绝，必须先把 `META_PATH` 指向含 grounding
replay 的正式 meta。完整输出写入 `/data2/zhenghengcao/grasp_anything_2d/logs/`。
切换数据阶段时不要设置 `RESUME_FROM_CHECKPOINT`；该变量会错误恢复 overfit64
的优化器和 dataloader 状态。需要先检查命令时使用
`DRY_RUN=1 bash start_phase2_sft.sh`。
