$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Root
$EnvPath = Join-Path $ProjectRoot "envs\detr-gpu"
$env:TORCH_HOME = Join-Path $ProjectRoot "torch_cache"
$CondaExe = "C:\Users\Administrator\anaconda3\Scripts\conda.exe"
Set-Location $Root

& $CondaExe run -p $EnvPath python main.py `
  --epochs 1 `
  --lr_drop 1 `
  --batch_size 1 `
  --num_workers 0 `
  --train_limit 8 `
  --val_limit 8 `
  --no_aux_loss `
  --resume .\detr-r50_4.pth `
  --coco_path .\coco `
  --output_dir .\experiments\runs\smoke_cpu `
  --device cpu
