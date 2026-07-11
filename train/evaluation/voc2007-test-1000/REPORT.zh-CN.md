# LocateAnything VOC2007 Test 对比评测报告

## 实验结论

在固定的 1000 张 VOC2007 test 子集上，step-1797 微调模型相较官方原始权重取得了明显的定位准确度提升：mAP50:95 提升 6.46 个百分点，AP50 提升 7.94 个百分点，IoU50 Micro-F1 提升 6.19 个百分点。代价是吞吐下降约 31.0%，平均端到端推理延迟增加约 44.9%，峰值显存增加约 229 MiB。

综合判断：本次微调有效收敛，并显著改善了 Pascal VOC 类别条件定位能力；如果主要目标是准确率，建议使用 step-1797。如果主要目标是吞吐，官方原始权重更快。

## 实验配置

- 数据集：VOC2007 test 固定随机子集
- 图片数：1000
- 随机子集种子：20260711
- 类别数：20，全部类别均被覆盖
- 官方模型：NVIDIA LocateAnything-3B，revision `c32291ca5e996f5a7a485845b4f57a233936bba0`
- 微调模型：`checkpoint-1797`
- GPU：NVIDIA GeForce RTX 3090 Ti
- 精度：BF16
- Generation mode：hybrid
- 最大生成长度：512 tokens
- 解码：官方采样参数，`do_sample=True`、`temperature=0.7`、`top_p=0.9`
- 提示方式：每张图片针对其 GT-positive 类别分别执行一次单类别 grounding 查询
- 性能统计：排除前 5 张预热图片，不包含模型加载时间

## 总体指标

| 指标 | 官方原模型 | step-1797 | 变化 |
|---|---:|---:|---:|
| mAP50:95 | 61.05% | 67.51% | +6.46 pp |
| AP50 | 80.28% | 88.22% | +7.94 pp |
| AP75 | 66.75% | 72.77% | +6.02 pp |
| Micro Precision @ IoU50 | 81.89% | 88.91% | +7.02 pp |
| Micro Recall @ IoU50 | 88.10% | 93.35% | +5.25 pp |
| Micro F1 @ IoU50 | 84.88% | 91.08% | +6.19 pp |

## 性能指标

| 指标 | 官方原模型 | step-1797 | 变化 |
|---|---:|---:|---:|
| 图片吞吐 | 3.057 img/s | 2.110 img/s | -30.99% |
| 查询吞吐 | 4.686 query/s | 3.234 query/s | -30.99% |
| 平均图片延迟 | 0.327 s | 0.474 s | +44.90% |
| P50 图片延迟 | 0.245 s | 0.375 s | +52.86% |
| P95 图片延迟 | 0.730 s | 1.214 s | +66.29% |
| 峰值 GPU 显存 | 7829.7 MiB | 8058.3 MiB | +228.5 MiB |

## 每类别 AP50

| 类别 | 官方原模型 | step-1797 | 变化 |
|---|---:|---:|---:|
| aeroplane | 97.67% | 96.15% | -1.52 pp |
| bicycle | 82.90% | 91.19% | +8.29 pp |
| bird | 99.60% | 97.61% | -2.00 pp |
| boat | 46.38% | 86.31% | +39.93 pp |
| bottle | 54.78% | 69.75% | +14.97 pp |
| bus | 97.92% | 97.92% | 0.00 pp |
| car | 75.91% | 95.43% | +19.52 pp |
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
| sofa | 81.52% | 95.67% | +14.15 pp |
| train | 87.45% | 89.06% | +1.61 pp |
| tvmonitor | 87.04% | 91.25% | +4.20 pp |

微调模型在 15 个类别上提升，2 个类别持平，3 个类别轻微下降。最明显的提升出现在 boat、car、bottle、sofa、chair、pottedplant、motorbike 和 person。aeroplane、bird、diningtable 存在小幅回退，需要在后续独立验证或更大 test 样本上确认是否属于采样波动。

## 评测限制

本评测遵循项目官方的 GT-positive category localization 思路：评测器会针对图片中真实存在的类别分别发出定位请求。因此指标衡量的是“已知类别名称条件下的定位能力”，不会衡量不存在类别的误报。

LocateAnything 的文本输出不包含每个框的原生置信度。报告中的 AP 使用稳定输出顺序作为排序依据，适合在完全相同协议下比较官方模型与微调模型，但不应直接与传统检测器论文中的标准 VOC Detection AP 横向比较。

## 文件索引

- 测试子集：`evaluation/voc2007-test-1000/image_ids.txt`
- 官方模型预测：`evaluation/voc2007-test-1000/base/predictions.jsonl`
- 官方模型指标：`evaluation/voc2007-test-1000/base/metrics.json`
- 微调模型预测：`evaluation/voc2007-test-1000/lora-step1797/predictions.jsonl`
- 微调模型指标：`evaluation/voc2007-test-1000/lora-step1797/metrics.json`
- 评测脚本：`scripts/evaluate_voc2007.py`

