$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Root
$EnvPath = Join-Path $ProjectRoot "envs\detr-gpu"
$env:TORCH_HOME = Join-Path $ProjectRoot "torch_cache"
$CondaExe = "C:\Users\Administrator\anaconda3\Scripts\conda.exe"
Set-Location $Root

& $CondaExe run -p $EnvPath python main.py `
  --eval `
  --batch_size 6 `
  --no_aux_loss `
  --resume .\output_2\checkpoint.pth `
  --coco_path .\coco `
  --output_dir .\experiments\runs\baseline_eval_full `
  --device cuda
