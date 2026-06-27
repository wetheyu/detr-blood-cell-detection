# 项目报告

本目录保存自动生成的项目报告、指标表格和可视化图表。

- `dataset_summary.md`：数据集统计。
- `experiment_summary.md`：早期 `output_2/log.txt` 训练日志摘要。
- `experiment_runs_summary.md`：`experiments/runs/*/log.txt` 聚合指标。
- `final_experiment_analysis.md`：正式实验对比和结论。
- `detection_examples.md`：检测结果示例。
- `tables/`：数据和指标 CSV 表格。
- `figures/`：可视化图表。

重新生成：

```powershell
.\scripts\generate_reports.ps1
```
