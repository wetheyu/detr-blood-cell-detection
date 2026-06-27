$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvPath = Join-Path (Split-Path -Parent $Root) "envs\detr-gpu"
$CondaExe = "C:\Users\Administrator\anaconda3\Scripts\conda.exe"
Set-Location $Root

& $CondaExe run -p $EnvPath python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
& $CondaExe run -p $EnvPath python -c "import scipy, matplotlib, pandas, pycocotools, gradio, onnx, onnxruntime, cv2, submitit; print('project deps ok')"
