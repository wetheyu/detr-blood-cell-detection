# 数据集统计报告

本报告由 `tools/generate_project_reports.py` 自动生成，记录数据集划分、类别分布和目标尺度分布。

## 数据集划分

| 划分 | 图片数 | 标注数 | 类别数 | 平均标注/图 |
| --- | ---: | ---: | ---: | ---: |
| train | 765 | 10342 | 4 | 13.52 |
| val | 73 | 967 | 4 | 13.25 |

## 类别分布

| 划分 | 类别 | 标注数 |
| --- | --- | ---: |
| train | cells | 0 |
| train | Platelets | 739 |
| train | RBC | 8814 |
| train | WBC | 789 |
| val | cells | 0 |
| val | Platelets | 76 |
| val | RBC | 819 |
| val | WBC | 72 |

## 目标尺度分布

| 划分 | small | medium | large |
| --- | ---: | ---: | ---: |
| train | 472 | 8837 | 1033 |
| val | 51 | 847 | 69 |

## 数据分析

- 训练集和验证集均为 COCO detection 格式，核心字段包括 `images`、`annotations`、`categories`。
- 平均每图标注数量较高，体现出血细胞目标密集分布的特点。
- 类别分布和尺度分布支撑类别不均衡、小目标检测和边界框定位误差分析。
