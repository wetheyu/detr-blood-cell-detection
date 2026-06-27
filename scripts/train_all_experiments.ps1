$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $Root
$EnvPath = Join-Path $ProjectRoot "envs\detr-gpu"
$env:TORCH_HOME = Join-Path $ProjectRoot "torch_cache"
$CondaExe = "C:\Users\Administrator\anaconda3\Scripts\conda.exe"
Set-Location $Root

$Configs = @(
  "configs\experiments\baseline.json",
  "configs\experiments\lr_low.json",
  "configs\experiments\lr_high.json",
  "configs\experiments\query_50.json",
  "configs\experiments\query_150.json"
)

foreach ($Config in $Configs) {
  Write-Host "Running experiment: $Config"
  & $CondaExe run -p $EnvPath python tools\run_experiment.py $Config
}

& $CondaExe run -p $EnvPath python tools\generate_project_reports.py
