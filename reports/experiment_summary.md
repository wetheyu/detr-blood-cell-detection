# 实验结果摘要

本报告根据 `output_2/log.txt` 自动生成，记录早期训练日志摘要。

## 最优验证结果

| 指标 | 数值 |
| --- | ---: |
| epoch | 2 |
| AP | 0.5571 |
| AP50 | 0.8963 |
| AP75 | 0.5985 |
| APS | 0.3902 |
| APM | 0.5473 |
| APL | 0.4439 |

## 最后一轮结果

| 指标 | 数值 |
| --- | ---: |
| epoch | 7 |
| train_loss | 5.4773 |
| val_loss | 5.1589 |
| AP | 0.5484 |
| AP50 | 0.9021 |
| AP75 | 0.5908 |

## 结果分析

- AP50 明显高于 AP，说明模型能较好定位目标大致区域，但严格 IoU 下仍有提升空间。
- 训练损失和验证损失走势见 `reports/figures/training_curves.png`。
- AP 曲线见 `reports/figures/ap_curve.png`，用于观察验证集指标是否稳定。
- 正式调参结果汇总见 `reports/experiment_runs_summary.md`。
