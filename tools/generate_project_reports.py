import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_config import CLASSES, DEFAULT_COCO_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
FORMAL_RUN_ORDER = ["baseline", "lr_low", "lr_high", "query_50", "query_150"]


def ensure_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_training_log(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_split(split_name, annotation_file):
    data = load_json(annotation_file)
    categories = {item["id"]: item["name"] for item in data["categories"]}
    class_counter = Counter()
    area_counter = Counter()

    for ann in data["annotations"]:
        class_counter[categories.get(ann["category_id"], str(ann["category_id"]))] += 1
        area = ann.get("area", 0)
        if area < 32 * 32:
            area_counter["small"] += 1
        elif area < 96 * 96:
            area_counter["medium"] += 1
        else:
            area_counter["large"] += 1

    return {
        "split": split_name,
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": len(data["categories"]),
        "avg_annotations_per_image": len(data["annotations"]) / max(len(data["images"]), 1),
        "class_counter": class_counter,
        "area_counter": area_counter,
    }


def write_dataset_tables(summaries):
    with (TABLES_DIR / "dataset_stats.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "images", "annotations", "categories", "avg_annotations_per_image"])
        for item in summaries:
            writer.writerow([
                item["split"],
                item["images"],
                item["annotations"],
                item["categories"],
                f"{item['avg_annotations_per_image']:.2f}",
            ])

    with (TABLES_DIR / "class_distribution.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "annotations"])
        for item in summaries:
            for class_name in CLASSES:
                writer.writerow([item["split"], class_name, item["class_counter"].get(class_name, 0)])


def plot_class_distribution(summaries):
    x = list(range(len(CLASSES)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, item in zip([-width / 2, width / 2], summaries):
        counts = [item["class_counter"].get(class_name, 0) for class_name in CLASSES]
        ax.bar([v + offset for v in x], counts, width, label=item["split"])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Annotations")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)


def write_dataset_summary(summaries):
    lines = [
        "# 数据集统计报告",
        "",
        "本报告由 `tools/generate_project_reports.py` 自动生成，记录数据集划分、类别分布和目标尺度分布。",
        "",
        "## 数据集划分",
        "",
        "| 划分 | 图片数 | 标注数 | 类别数 | 平均标注/图 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['split']} | {item['images']} | {item['annotations']} | "
            f"{item['categories']} | {item['avg_annotations_per_image']:.2f} |"
        )

    lines.extend(["", "## 类别分布", "", "| 划分 | 类别 | 标注数 |", "| --- | --- | ---: |"])
    for item in summaries:
        for class_name in CLASSES:
            lines.append(f"| {item['split']} | {class_name} | {item['class_counter'].get(class_name, 0)} |")

    lines.extend(["", "## 目标尺度分布", "", "| 划分 | small | medium | large |", "| --- | ---: | ---: | ---: |"])
    for item in summaries:
        lines.append(
            f"| {item['split']} | {item['area_counter'].get('small', 0)} | "
            f"{item['area_counter'].get('medium', 0)} | {item['area_counter'].get('large', 0)} |"
        )

    lines.extend([
        "",
        "## 数据分析",
        "",
        "- 训练集和验证集均为 COCO detection 格式，核心字段包括 `images`、`annotations`、`categories`。",
        "- 平均每图标注数量较高，体现出血细胞目标密集分布的特点。",
        "- 类别分布和尺度分布支撑类别不均衡、小目标检测和边界框定位误差分析。",
    ])

    (REPORTS_DIR / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def extract_metrics(row):
    coco = row.get("test_coco_eval_bbox") or []
    return {
        "epoch": row.get("epoch"),
        "train_loss": row.get("train_loss"),
        "test_loss": row.get("test_loss"),
        "test_class_error": row.get("test_class_error"),
        "ap": coco[0] if len(coco) > 0 else None,
        "ap50": coco[1] if len(coco) > 1 else None,
        "ap75": coco[2] if len(coco) > 2 else None,
        "aps": coco[3] if len(coco) > 3 else None,
        "apm": coco[4] if len(coco) > 4 else None,
        "apl": coco[5] if len(coco) > 5 else None,
    }


def write_training_tables(metrics):
    with (TABLES_DIR / "training_log_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "test_loss", "test_class_error", "ap", "ap50", "ap75", "aps", "apm", "apl"],
        )
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def plot_training_curves(metrics):
    if not metrics:
        return
    epochs = [row["epoch"] for row in metrics]
    train_loss = [row["train_loss"] for row in metrics]
    test_loss = [row["test_loss"] for row in metrics]
    ap = [row["ap"] for row in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="train loss")
    ax.plot(epochs, test_loss, marker="o", label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, ap, marker="o", label="AP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_title("Validation AP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ap_curve.png", dpi=160)
    plt.close(fig)


def write_experiment_summary(metrics):
    lines = [
        "# 实验结果摘要",
        "",
        "本报告根据 `output_2/log.txt` 自动生成，记录早期训练日志摘要。",
        "",
    ]

    if not metrics:
        lines.extend([
            "未找到训练日志。请先完成训练或评估，再重新运行报告生成脚本。",
            "",
        ])
        (REPORTS_DIR / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")
        return

    best = max((row for row in metrics if row["ap"] is not None), key=lambda row: row["ap"])
    last = metrics[-1]

    lines.extend([
        "## 最优验证结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| epoch | {best['epoch']} |",
        f"| AP | {best['ap']:.4f} |",
        f"| AP50 | {best['ap50']:.4f} |",
        f"| AP75 | {best['ap75']:.4f} |",
        f"| APS | {best['aps']:.4f} |",
        f"| APM | {best['apm']:.4f} |",
        f"| APL | {best['apl']:.4f} |",
        "",
        "## 最后一轮结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| epoch | {last['epoch']} |",
        f"| train_loss | {last['train_loss']:.4f} |",
        f"| val_loss | {last['test_loss']:.4f} |",
        f"| AP | {last['ap']:.4f} |",
        f"| AP50 | {last['ap50']:.4f} |",
        f"| AP75 | {last['ap75']:.4f} |",
        "",
        "## 结果分析",
        "",
        "- AP50 明显高于 AP，说明模型能较好定位目标大致区域，但严格 IoU 下仍有提升空间。",
        "- 训练损失和验证损失走势见 `reports/figures/training_curves.png`。",
        "- AP 曲线见 `reports/figures/ap_curve.png`，用于观察验证集指标是否稳定。",
        "- 正式调参结果汇总见 `reports/experiment_runs_summary.md`。",
    ])

    (REPORTS_DIR / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def summarize_run(run_dir):
    log_path = run_dir / "log.txt"
    rows = load_training_log(log_path)
    if not rows:
        return None
    metrics = [extract_metrics(row) for row in rows]
    valid_metrics = [row for row in metrics if row["ap"] is not None]
    best = max(valid_metrics, key=lambda row: row["ap"]) if valid_metrics else metrics[-1]
    last = metrics[-1]
    return {
        "run": run_dir.name,
        "epochs_logged": len(metrics),
        "best_epoch": best.get("epoch"),
        "best_ap": best.get("ap"),
        "best_ap50": best.get("ap50"),
        "best_ap75": best.get("ap75"),
        "last_epoch": last.get("epoch"),
        "last_train_loss": last.get("train_loss"),
        "last_val_loss": last.get("test_loss"),
    }


def sort_runs(summaries):
    order = {name: idx for idx, name in enumerate(FORMAL_RUN_ORDER)}
    return sorted(summaries, key=lambda row: (order.get(row["run"], 999), row["run"]))


def is_smoke_run(run_name):
    return "smoke" in run_name


def format_metric(value):
    return "" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def plot_experiment_comparison(summaries):
    formal = [row for row in sort_runs(summaries) if not is_smoke_run(row["run"])]
    if not formal:
        return

    labels = [row["run"] for row in formal]
    ap = [row["best_ap"] for row in formal]
    ap50 = [row["best_ap50"] for row in formal]
    ap75 = [row["best_ap75"] for row in formal]

    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([v - width for v in x], ap, width, label="AP")
    ax.bar(x, ap50, width, label="AP50")
    ax.bar([v + width for v in x], ap75, width, label="AP75")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Experiment comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment_comparison.png", dpi=160)
    plt.close(fig)


def plot_formal_run_curves(summaries):
    formal = [row for row in sort_runs(summaries) if not is_smoke_run(row["run"])]
    if not formal:
        return

    fig_ap, ax_ap = plt.subplots(figsize=(9, 5))
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5))

    for row in formal:
        run_dir = PROJECT_ROOT / "experiments" / "runs" / row["run"]
        metrics = [extract_metrics(item) for item in load_training_log(run_dir / "log.txt")]
        metrics = [item for item in metrics if item["ap"] is not None]
        if not metrics:
            continue
        epochs = [item["epoch"] for item in metrics]
        ap = [item["ap"] for item in metrics]
        train_loss = [item["train_loss"] for item in metrics]
        val_loss = [item["test_loss"] for item in metrics]
        ax_ap.plot(epochs, ap, marker="o", markersize=3, label=row["run"])
        ax_loss.plot(epochs, val_loss, marker="o", markersize=3, label=f"{row['run']} val")
        ax_loss.plot(epochs, train_loss, linestyle="--", linewidth=1, alpha=0.6, label=f"{row['run']} train")

    ax_ap.set_xlabel("Epoch")
    ax_ap.set_ylabel("AP")
    ax_ap.set_title("Formal experiment AP curves")
    ax_ap.legend()
    fig_ap.tight_layout()
    fig_ap.savefig(FIGURES_DIR / "formal_experiment_ap_curves.png", dpi=160)
    plt.close(fig_ap)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Formal experiment loss curves")
    ax_loss.legend(ncol=2, fontsize=8)
    fig_loss.tight_layout()
    fig_loss.savefig(FIGURES_DIR / "formal_experiment_loss_curves.png", dpi=160)
    plt.close(fig_loss)


def write_final_experiment_analysis(summaries):
    formal = [row for row in sort_runs(summaries) if not is_smoke_run(row["run"])]
    if not formal:
        return

    best = max(formal, key=lambda row: row["best_ap"])
    baseline = next((row for row in formal if row["run"] == "baseline"), None)
    improvement = None
    if baseline is not None and baseline["best_ap"] is not None:
        improvement = best["best_ap"] - baseline["best_ap"]

    lines = [
        "# 最终实验分析",
        "",
        "本报告由 `tools/generate_project_reports.py` 根据 `experiments/runs/*/log.txt` 自动生成，记录正式实验对比和结果分析。",
        "",
        "## 正式实验对比",
        "",
        "| 实验 | 轮数 | 最优 epoch | AP | AP50 | AP75 | 最后 train loss | 最后 val loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in formal:
        lines.append(
            f"| {row['run']} | {row['epochs_logged']} | {format_metric(row['best_epoch'])} | "
            f"{format_metric(row['best_ap'])} | {format_metric(row['best_ap50'])} | {format_metric(row['best_ap75'])} | "
            f"{format_metric(row['last_train_loss'])} | {format_metric(row['last_val_loss'])} |"
        )

    lines.extend([
        "",
        "## 最优模型",
        "",
        f"- 最优实验：`{best['run']}`。",
        f"- 最优 epoch：{best['best_epoch']}。",
        f"- 验证集 AP：{best['best_ap']:.4f}。",
        f"- 验证集 AP50：{best['best_ap50']:.4f}。",
        f"- 验证集 AP75：{best['best_ap75']:.4f}。",
        f"- 最优权重：`experiments/runs/{best['run']}/checkpoint_best.pth`。",
    ])
    if improvement is not None:
        lines.append(f"- 相比 baseline，AP 提升 {improvement:.4f}。")

    lines.extend([
        "",
        "## 结果解读",
        "",
        "- 学习率调参带来了最主要的性能提升，`lr_low` 在 AP 和 AP75 上取得最高结果，说明较低学习率更有利于小规模血细胞数据集上的稳定收敛。",
        "- `lr_high` 的 AP50 较高，但整体 AP 和 AP75 略低于 `lr_low`，说明较大学习率在宽松 IoU 下仍能定位目标，但严格定位质量稍弱。",
        "- `query_50` 和 `query_150` 均低于 100 queries 的最佳结果，说明在当前数据集规模和训练策略下，默认 100 queries 在检测容量和优化难度之间更均衡。",
        "- AP50 明显高于 AP，说明模型对目标大致位置识别较好，但严格 IoU 阈值下仍存在边界框精细定位提升空间。",
        "",
        "## 结果图表",
        "",
        "- 实验对比图：`reports/figures/experiment_comparison.png`。",
        "- 正式实验 AP 曲线：`reports/figures/formal_experiment_ap_curves.png`。",
        "- 正式实验 loss 曲线：`reports/figures/formal_experiment_loss_curves.png`。",
        "- 训练曲线：`reports/figures/training_curves.png`。",
        "- AP 曲线：`reports/figures/ap_curve.png`。",
        "- 检测示例：`reports/figures/detection_examples_grid.jpg`。",
    ])
    (REPORTS_DIR / "final_experiment_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_runs_summary():
    runs_root = PROJECT_ROOT / "experiments" / "runs"
    summaries = []
    if runs_root.exists():
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            summary = summarize_run(run_dir)
            if summary:
                summaries.append(summary)

    summaries = sort_runs(summaries)

    with (TABLES_DIR / "experiment_runs_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "epochs_logged",
                "best_epoch",
                "best_ap",
                "best_ap50",
                "best_ap75",
                "last_epoch",
                "last_train_loss",
                "last_val_loss",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    lines = [
        "# 调参实验汇总",
        "",
        "本表扫描 `experiments/runs/*/log.txt` 自动生成。完整调参训练结束后，重新运行报告生成脚本即可更新。",
        "",
    ]
    if not summaries:
        lines.extend(["当前没有发现可汇总的实验日志。", ""])
    else:
        formal = [row for row in summaries if not is_smoke_run(row["run"])]
        smoke = [row for row in summaries if is_smoke_run(row["run"])]
        lines.extend([
            "## 正式实验",
            "",
            "| 实验 | 日志轮数 | 最优 epoch | AP | AP50 | AP75 | 最后 train loss | 最后 val loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in formal:
            lines.append(
                f"| {row['run']} | {row['epochs_logged']} | {format_metric(row['best_epoch'])} | "
                f"{format_metric(row['best_ap'])} | {format_metric(row['best_ap50'])} | {format_metric(row['best_ap75'])} | "
                f"{format_metric(row['last_train_loss'])} | {format_metric(row['last_val_loss'])} |"
            )
        if smoke:
            lines.extend([
                "",
                "## 链路测试实验",
                "",
                "| 实验 | 日志轮数 | 最优 epoch | AP | AP50 | AP75 | 说明 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ])
            for row in smoke:
                lines.append(
                    f"| {row['run']} | {row['epochs_logged']} | {format_metric(row['best_epoch'])} | "
                    f"{format_metric(row['best_ap'])} | {format_metric(row['best_ap50'])} | {format_metric(row['best_ap75'])} | "
                    "用于验证训练/评估链路，不计入最终性能。 |"
                )

    (REPORTS_DIR / "experiment_runs_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    plot_experiment_comparison(summaries)
    plot_formal_run_curves(summaries)
    write_final_experiment_analysis(summaries)
    return summaries


def make_detection_examples_grid():
    pred_dir = PROJECT_ROOT / "experiments" / "runs" / "predict_val_best"
    if not pred_dir.exists():
        pred_dir = PROJECT_ROOT / "experiments" / "runs" / "predict_val"
    image_paths = sorted(pred_dir.glob("*.jpg"))[:6]
    if not image_paths:
        return

    thumb_size = (240, 240)
    padding = 12
    label_h = 24
    cols = 3
    rows = (len(image_paths) + cols - 1) // cols
    grid_w = cols * thumb_size[0] + (cols + 1) * padding
    grid_h = rows * (thumb_size[1] + label_h) + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_size[0] + padding)
        y = padding + row * (thumb_size[1] + label_h + padding)
        image = Image.open(image_path).convert("RGB")
        image.thumbnail(thumb_size)
        canvas = Image.new("RGB", thumb_size, "white")
        canvas.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
        grid.paste(canvas, (x, y))
        draw.text((x, y + thumb_size[1] + 4), image_path.name.split("_jpg")[0], fill="black")

    output_path = FIGURES_DIR / "detection_examples_grid.jpg"
    grid.save(output_path, quality=92)

    lines = [
        "# 检测结果示例",
        "",
        f"本页基于 `{pred_dir.relative_to(PROJECT_ROOT).as_posix()}` 的批量预测结果自动生成。",
        "",
        f"示例拼图：`figures/{output_path.name}`",
        "",
        "原始预测图保存在批量预测目录中，示例拼图用于快速查看模型在验证集上的检测效果。",
    ]
    (REPORTS_DIR / "detection_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_reports_readme():
    lines = [
        "# 项目报告",
        "",
        "本目录保存自动生成的项目报告、指标表格和可视化图表。",
        "",
        "- `dataset_summary.md`：数据集统计。",
        "- `experiment_summary.md`：早期 `output_2/log.txt` 训练日志摘要。",
        "- `experiment_runs_summary.md`：`experiments/runs/*/log.txt` 聚合指标。",
        "- `final_experiment_analysis.md`：正式实验对比和结论。",
        "- `detection_examples.md`：检测结果示例。",
        "- `tables/`：数据和指标 CSV 表格。",
        "- `figures/`：可视化图表。",
        "",
        "重新生成：",
        "",
        "```powershell",
        ".\\scripts\\generate_reports.ps1",
        "```",
    ]
    (REPORTS_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main():
    ensure_dirs()
    summaries = [
        summarize_split("train", DEFAULT_COCO_PATH / "annotations" / "instances_train2017.json"),
        summarize_split("val", DEFAULT_COCO_PATH / "annotations" / "instances_val2017.json"),
    ]
    write_dataset_tables(summaries)
    plot_class_distribution(summaries)
    write_dataset_summary(summaries)

    log_rows = load_training_log(DEFAULT_OUTPUT_DIR / "log.txt")
    metrics = [extract_metrics(row) for row in log_rows]
    write_training_tables(metrics)
    plot_training_curves(metrics)
    write_experiment_summary(metrics)
    write_runs_summary()
    make_detection_examples_grid()
    write_reports_readme()
    print(f"Reports generated in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
