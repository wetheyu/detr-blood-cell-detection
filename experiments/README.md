# Experiments

This directory keeps tracked experiment plans and ignores heavy run outputs.

Generated run directories under `experiments/runs/` are intentionally ignored by
git because they contain checkpoints, COCO eval files, and prediction images.

## Environment

- Conda env: `D:\DETR\envs\detr-gpu`
- Python: 3.8.18
- PyTorch: 1.7.1
- Torchvision: 0.8.2
- CUDA: 10.1, available
- GPU: GeForce GTX 1070 with Max-Q Design

## Final Formal Experiments

All formal experiments were run with GPU training, auxiliary DETR losses enabled,
`batch_size=2`, `lr_drop=15`, and early stopping.

| Run | Best epoch | AP | AP50 | AP75 | Best checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 11 | 0.5949 | 0.9129 | 0.6445 | `experiments/runs/baseline/checkpoint_best.pth` |
| lr_low | 23 | 0.6365 | 0.9176 | 0.7455 | `experiments/runs/lr_low/checkpoint_best.pth` |
| lr_high | 32 | 0.6348 | 0.9288 | 0.7380 | `experiments/runs/lr_high/checkpoint_best.pth` |
| query_50 | 40 | 0.6157 | 0.9131 | 0.6950 | `experiments/runs/query_50/checkpoint_best.pth` |
| query_150 | 20 | 0.6156 | 0.9140 | 0.6908 | `experiments/runs/query_150/checkpoint_best.pth` |

The selected final model is `lr_low`.

## Commands

Run all formal experiments:

```powershell
.\scripts\train_all_experiments.ps1
```

Evaluate the selected final model:

```powershell
.\scripts\evaluate_best.ps1
```

Generate validation prediction visualizations with the selected model:

```powershell
.\scripts\predict_val_gpu.ps1
```

Regenerate project reports:

```powershell
.\scripts\generate_reports.ps1
```

## Engineering Smoke Tests

Smoke tests are used only to prove that the code, dataset, checkpoint loading,
training loop, and evaluation loop are operational.

| Run | Purpose | Result |
| --- | --- | --- |
| `gpu_env_smoke` | 2-image GPU evaluation | Passed |
| `baseline_eval_smoke` | 8-image quick evaluation | AP 0.533, AP50 0.870 |
| `smoke_cpu` | 8-image CPU training loop | Training/evaluation passed |
