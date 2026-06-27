# 运行手册

## 1. 环境检查

当前机器已经配置了推荐环境：

- `D:\DETR\envs\detr-gpu`：新配置的 GPU 环境，用于训练、评估、报告生成和 Web 演示。

GPU 环境检查：

```powershell
.\scripts\check_gpu_env.ps1
```

当前 GPU 环境检查结果：

- PyTorch：1.7.1
- Torchvision：0.8.2
- CUDA runtime：10.1
- GPU：GeForce GTX 1070 with Max-Q Design
- `torch.cuda.is_available()`：`True`

## 2. 安装环境

当前环境已安装在：

```powershell
D:\DETR\envs\detr-gpu
```

PyTorch 模型缓存目录：

```powershell
D:\DETR\torch_cache
```

如需重新创建，可参考：

```powershell
$env:CONDA_PKGS_DIRS='D:\DETR\conda_pkgs'
conda create -p D:\DETR\envs\detr-gpu python=3.8 -y
conda install -p D:\DETR\envs\detr-gpu pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch -y
conda install -p D:\DETR\envs\detr-gpu scipy matplotlib pandas -y
$env:PIP_CACHE_DIR='D:\DETR\pip_cache'
conda run -p D:\DETR\envs\detr-gpu python -m pip install pycocotools gradio onnx onnxruntime opencv-python submitit
```

## 3. 评估最优模型

```powershell
.\scripts\evaluate_best.ps1
```

## 4. 快速训练链路测试

```powershell
.\scripts\train_smoke_cpu.ps1
```

该脚本使用 `detr-gpu` 环境运行，但显式指定 CPU 和小样本数据，只用于检查训练链路，不作为最终性能结果。

## 5. GPU 完整训练

```powershell
.\scripts\train_baseline_gpu.ps1
```

一键运行全部正式实验：

```powershell
.\scripts\train_all_experiments.ps1
```

实验脚本支持自动续训，并会保存 `checkpoint_best.pth`。如果 GPU 显存不足，保持 batch size 为 1 或 2。

## 6. 批量预测

```powershell
.\scripts\predict_val_gpu.ps1
```

## 7. Web 演示

```powershell
.\scripts\launch_web_best.ps1
```

## 8. 生成项目报告

```powershell
.\scripts\generate_reports.ps1
```

生成内容包括数据集统计、正式实验对比、最优模型分析、CSV 表格、曲线图和检测示例拼图。
