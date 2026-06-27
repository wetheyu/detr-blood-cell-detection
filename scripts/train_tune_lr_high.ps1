$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Root
$EnvPath = Join-Path $ProjectRoot "envs\detr-gpu"
$env:TORCH_HOME = Join-Path $ProjectRoot "torch_cache"
$CondaExe = "C:\Users\Administrator\anaconda3\Scripts\conda.exe"
Set-Location $Root

& $CondaExe run -p $EnvPath python tools\run_experiment.py configs\experiments\lr_high.json
