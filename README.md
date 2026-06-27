# DETR 血细胞目标检测项目

本项目基于 `facebookresearch/detr`，面向血细胞图片目标检测任务完成了数据整理、模型训练、调参实验、结果评估、可视化预测和 Web 演示。

## 项目状态

- 检测类别：`cells`、`Platelets`、`RBC`、`WBC`
- 训练集：765 张图片，10,342 条标注
- 验证集：73 张图片，967 条标注
- 最优实验：`lr_low`
- 最优结果：AP=0.637，AP50=0.918，AP75=0.745
- 最优权重：`experiments/runs/lr_low/checkpoint_best.pth`

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `main.py` | DETR 训练与评估入口 |
| `predict.py` | 验证集批量预测与可视化 |
| `web.py` | Gradio 单图检测演示 |
| `project_config.py` | 类别、数据路径、默认权重配置 |
| `configs/experiments/` | 基线、学习率、query 数量等实验配置 |
| `scripts/` | 训练、评估、预测、报告生成脚本 |
| `docs/` | 项目计划、实验设计、运行手册、技术报告结构 |
| `reports/` | 实验分析、图表、检测示例与复现记录 |
| `experiments/README.md` | 实验管理与复现说明 |

## 环境

本机已配置 GPU 环境：

```text
D:\DETR\envs\detr-gpu
```

主要依赖：

- Python 3.8
- PyTorch 1.7.1
- TorchVision 0.8.2
- CUDA 10.1
- pycocotools-windows

检查环境：

```powershell
cd D:\DETR\detr
.\scripts\check_gpu_env.ps1
```

重新配置环境可参考 `docs/runbook.md` 和 `environment_gpu_cuda101.yml`。

## 常用命令

评估最优模型：

```powershell
.\scripts\evaluate_best.ps1
```

生成验证集预测图：

```powershell
.\scripts\predict_val_gpu.ps1
```

启动 Web 演示：

```powershell
.\scripts\launch_web_best.ps1
```

重新生成报告：

```powershell
.\scripts\generate_reports.ps1
```

重新运行全部正式实验：

```powershell
.\scripts\train_all_experiments.ps1
```

## 实验结果

| 实验 | AP | AP50 | AP75 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 0.595 | 0.913 | 0.645 | 基线模型 |
| `lr_low` | 0.637 | 0.918 | 0.745 | 最优模型 |
| `lr_high` | 0.635 | 0.929 | 0.738 | 较高学习率 |
| `query_50` | 0.616 | 0.913 | 0.695 | query 数量减少 |
| `query_150` | 0.616 | 0.914 | 0.691 | query 数量增加 |

详细分析见：

- `reports/final_experiment_analysis.md`
- `reports/experiment_runs_summary.md`
- `reports/figures/experiment_comparison.png`
- `reports/figures/formal_experiment_ap_curves.png`
- `reports/figures/formal_experiment_loss_curves.png`
- `reports/figures/detection_examples_grid.jpg`

## 项目文档

- 项目规划：`docs/project_plan.md`
- 实验设计：`docs/experiment_plan.md`
- 运行手册：`docs/runbook.md`
- 环境记录：`docs/environment_setup_result.md`
- 技术报告结构：`docs/technical_report_outline.md`
- 项目检查清单：`docs/project_checklist.md`

## GitHub 提交说明

当前版本适合上传 GitHub 进行代码审阅和项目展示。提交范围包括代码、配置、脚本、文档和报告；不要提交以下内容：

- `coco/`
- `output*/`
- `experiments/runs/`
- `*.pth`
- `flagged/`

如需完整复现，需要另外提供数据集和最优权重 `experiments/runs/lr_low/checkpoint_best.pth`。

## 来源与许可

本项目基于 Facebook Research 开源 DETR 改造，原项目采用 Apache 2.0 License，许可说明见 `LICENSE`。
