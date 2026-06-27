# 实验设计与调参方案

## 评价指标

采用 COCO 目标检测指标：

- AP：IoU=0.50:0.95 的平均精度，作为主要指标。
- AP50：IoU=0.50 的平均精度，体现宽松定位下的检测能力。
- AP75：IoU=0.75 的平均精度，体现更严格定位质量。
- APS/APM/APL：小、中、大目标上的 AP。
- AR：平均召回率，用于辅助分析漏检情况。

## 实验 1：基线模型

目的：验证 DETR-R50 在本数据集上的基础效果。

推荐设置：

- Backbone：ResNet-50。
- Query 数量：100。
- 学习率：`1e-4`。
- Backbone 学习率：`1e-5`。
- Batch size：GPU 环境使用 2；CPU 只用于小样本 smoke test。
- 初始权重：`detr-r50_4.pth`。

输出：

- `reports/experiment_summary.md`
- `reports/figures/training_curves.png`
- `reports/figures/ap_curve.png`

## 实验 2：学习率调参

目的：比较不同学习率对收敛速度和验证 AP 的影响。

候选设置：

| 实验名 | lr | lr_backbone | 预期观察 |
| --- | --- | --- | --- |
| `baseline` | `1e-4` | `1e-5` | 标准设置。 |
| `lr_low` | `5e-5` | `5e-6` | 更稳定但可能收敛较慢。 |
| `lr_high` | `2e-4` | `2e-5` | 收敛快，但可能震荡。 |

## 实验 3：Query 数量调参

目的：分析 DETR 中 object queries 数量对密集血细胞检测的影响。

候选设置：

| 实验名 | num_queries | 预期观察 |
| --- | --- | --- |
| `query_50` | 50 | 计算量更低，但密集目标可能漏检。 |
| `baseline` | 100 | 默认设置。 |
| `query_150` | 150 | 容纳更多候选目标，但可能增加误检和训练成本。 |

## 实验 4：CPU 快速链路测试

目的：在没有 GPU 时验证代码、数据、权重、训练循环和评估循环是否完整可运行。

设置：

- `--train_limit 8`
- `--val_limit 8`
- `--epochs 1`
- `--batch_size 1`
- `--num_workers 0`

注意：这个实验不计入最终性能，只用于确认训练和评估链路可运行。

## 实验结果表

| 实验 | lr | queries | AP | AP50 | AP75 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1e-4 | 100 | 0.5949 | 0.9129 | 0.6445 | 标准设置。 |
| lr_low | 5e-5 | 100 | 0.6365 | 0.9176 | 0.7455 | 最优模型，学习率降低后定位质量更好。 |
| lr_high | 2e-4 | 100 | 0.6348 | 0.9288 | 0.7380 | AP50 最高，但 AP/AP75 略低于 `lr_low`。 |
| query_50 | 1e-4 | 50 | 0.6157 | 0.9131 | 0.6950 | 减少查询数量后性能下降。 |
| query_150 | 1e-4 | 150 | 0.6156 | 0.9140 | 0.6908 | 增加查询数量未带来收益。 |

## 自动运行方式

推荐使用统一实验运行器：

```powershell
.\scripts\run_experiment.ps1 configs\experiments\baseline.json
.\scripts\run_experiment.ps1 configs\experiments\lr_low.json
.\scripts\run_experiment.ps1 configs\experiments\lr_high.json
.\scripts\run_experiment.ps1 configs\experiments\query_50.json
.\scripts\run_experiment.ps1 configs\experiments\query_150.json
```

也可以一键运行全部正式实验：

```powershell
.\scripts\train_all_experiments.ps1
```

每组实验完成后运行：

```powershell
.\scripts\generate_reports.ps1
```

`reports/experiment_runs_summary.md` 会自动汇总 `experiments/runs/*/log.txt`。

最终实验分析见 `reports/final_experiment_analysis.md`，正式实验曲线见 `reports/figures/formal_experiment_ap_curves.png` 和 `reports/figures/formal_experiment_loss_curves.png`。
