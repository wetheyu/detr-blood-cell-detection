# 可复现实验检查

## 本机环境

| 项目 | 值 |
| --- | --- |
| Conda 环境 | `D:\DETR\envs\detr-gpu` |
| Python | 3.8.18 |
| PyTorch | 1.7.1 |
| Torchvision | 0.8.2 |
| CUDA | 10.1，可用 |
| GPU | GeForce GTX 1070 with Max-Q Design |

## 已完成检查

| 检查项 | 命令/文件 | 结果 |
| --- | --- | --- |
| 语法检查 | `python -m py_compile main.py predict.py web.py project_config.py tools\generate_project_reports.py tools\run_experiment.py models\detr.py` | 通过 |
| 模块导入 | `import main, predict, web, project_config` | 通过 |
| 报告生成 | `scripts\generate_reports.ps1` | 通过 |
| 实验运行器 | `scripts\run_experiment.ps1 configs\experiments\baseline.json --dry-run` | 通过 |
| GPU 环境检查 | `scripts\check_gpu_env.ps1` | 通过，`torch.cuda.is_available()` 为 True |
| GPU smoke eval | `experiments/runs/gpu_env_smoke` | 2 张图评估通过 |
| 8 图评估 smoke test | `experiments/runs/baseline_eval_smoke` | AP 0.533，AP50 0.870 |
| 8 图训练 smoke test | `experiments/runs/smoke_cpu` | 训练/评估链路通过 |
| 完整训练 baseline | `experiments/runs/baseline` | AP 0.5949，AP50 0.9129，AP75 0.6445 |
| 学习率调参 | `experiments/runs/lr_low`、`experiments/runs/lr_high` | 最优 AP 0.6365 |
| Query 数量调参 | `experiments/runs/query_50`、`experiments/runs/query_150` | 100 queries 最优 |
| 最优模型 GPU 复评 | `experiments/runs/best_eval` | AP 0.637，AP50 0.918，AP75 0.745 |
| 最优模型批量预测 | `experiments/runs/predict_val_best` | 73 张验证集预测图生成完成 |

## 说明

当前机器已配置 GPU 版 PyTorch。完整训练和调参使用 `scripts/train_baseline_gpu.ps1`、`scripts/train_tune_lr_low.ps1`、`scripts/train_tune_lr_high.ps1`、`scripts/train_tune_query_50.ps1`、`scripts/train_tune_query_150.ps1` 运行，训练结束后重新执行 `scripts/generate_reports.ps1` 汇总结果。
