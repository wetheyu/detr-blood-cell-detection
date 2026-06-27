$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Root
$EnvPath = Join-Path $ProjectRoot "envs\detr-gpu"
$env:TORCH_HOME = Join-Path $ProjectRoot "torch_cache"
$CondaExe = "C:\Users\Administrator\anaconda3\Scripts\conda.exe"
Set-Location $Root

& $CondaExe run -p $EnvPath python web.py `
  --resume .\experiments\runs\lr_low\checkpoint_best.pth `
  --device cuda
