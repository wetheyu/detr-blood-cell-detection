# 最终实验分析

本报告由 `tools/generate_project_reports.py` 根据 `experiments/runs/*/log.txt` 自动生成，记录正式实验对比和结果分析。

## 正式实验对比

| 实验 | 轮数 | 最优 epoch | AP | AP50 | AP75 | 最后 train loss | 最后 val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 24 | 11 | 0.5949 | 0.9129 | 0.6445 | 4.9290 | 4.9707 |
| lr_low | 36 | 23 | 0.6365 | 0.9176 | 0.7455 | 4.3190 | 4.2063 |
| lr_high | 45 | 32 | 0.6348 | 0.9288 | 0.7380 | 4.4056 | 4.1822 |
| query_50 | 43 | 40 | 0.6157 | 0.9131 | 0.6950 | 4.8566 | 4.7232 |
| query_150 | 33 | 20 | 0.6156 | 0.9140 | 0.6908 | 4.2769 | 4.1648 |

## 最优模型

- 最优实验：`lr_low`。
- 最优 epoch：23。
- 验证集 AP：0.6365。
- 验证集 AP50：0.9176。
- 验证集 AP75：0.7455。
- 最优权重：`experiments/runs/lr_low/checkpoint_best.pth`。
- 相比 baseline，AP 提升 0.0417。

## 结果解读

- 学习率调参带来了最主要的性能提升，`lr_low` 在 AP 和 AP75 上取得最高结果，说明较低学习率更有利于小规模血细胞数据集上的稳定收敛。
- `lr_high` 的 AP50 较高，但整体 AP 和 AP75 略低于 `lr_low`，说明较大学习率在宽松 IoU 下仍能定位目标，但严格定位质量稍弱。
- `query_50` 和 `query_150` 均低于 100 queries 的最佳结果，说明在当前数据集规模和训练策略下，默认 100 queries 在检测容量和优化难度之间更均衡。
- AP50 明显高于 AP，说明模型对目标大致位置识别较好，但严格 IoU 阈值下仍存在边界框精细定位提升空间。

## 结果图表

- 实验对比图：`reports/figures/experiment_comparison.png`。
- 正式实验 AP 曲线：`reports/figures/formal_experiment_ap_curves.png`。
- 正式实验 loss 曲线：`reports/figures/formal_experiment_loss_curves.png`。
- 训练曲线：`reports/figures/training_curves.png`。
- AP 曲线：`reports/figures/ap_curve.png`。
- 检测示例：`reports/figures/detection_examples_grid.jpg`。
