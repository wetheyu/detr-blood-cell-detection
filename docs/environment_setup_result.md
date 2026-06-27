# 环境配置结果

## GPU 环境

已配置独立 GPU 环境：

```powershell
D:\DETR\envs\detr-gpu
```

验证命令：

```powershell
.\scripts\check_gpu_env.ps1
```

验证结果：

| 项目 | 结果 |
| --- | --- |
| Python | 3.8.18 |
| PyTorch | 1.7.1 |
| Torchvision | 0.8.2 |
| CUDA runtime | 10.1 |
| GPU | GeForce GTX 1070 with Max-Q Design |
| `torch.cuda.is_available()` | True |

## 已安装依赖

- `pytorch`
- `torchvision`
- `scipy`
- `matplotlib`
- `pandas`
- `pycocotools`
- `gradio`
- `onnx`
- `onnxruntime`
- `opencv-python`
- `submitit`

## 缓存目录

为了避免 C 盘空间不足，环境和缓存放在 D 盘：

| 内容 | 路径 |
| --- | --- |
| conda 环境 | `D:\DETR\envs\detr-gpu` |
| conda 包缓存 | `D:\DETR\conda_pkgs` |
| pip 缓存 | `D:\DETR\pip_cache` |
| PyTorch 模型缓存 | `D:\DETR\torch_cache` |

这些目录已经在外层 `.gitignore` 中忽略。

## GPU smoke test

已运行 2 张验证图的 GPU 快速评估：

```powershell
conda run -p D:\DETR\envs\detr-gpu python main.py --eval --val_limit 2 --batch_size 1 --num_workers 0 --no_aux_loss --resume .\output_2\checkpoint.pth --coco_path .\coco --output_dir .\experiments\runs\gpu_env_smoke --device cuda
```

结果：评估链路通过，模型使用 `cuda`，显存峰值约 336 MB。

## 最优模型完整验证集 GPU 评估

已运行完整验证集评估：

```powershell
.\scripts\evaluate_best.ps1
```

结果：

| 指标 | 数值 |
| --- | ---: |
| AP | 0.637 |
| AP50 | 0.918 |
| AP75 | 0.745 |
| APS | 0.476 |
| APM | 0.654 |
| APL | 0.485 |
| AR@100 | 0.745 |

评估循环耗时约 18 秒，显存峰值约 1203 MB。
