$ErrorActionPreference = "Stop"

$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

if (Test-Path -LiteralPath $cudaRoot) {
    $env:CUDA_HOME = $cudaRoot
    $env:CUDA_PATH = $cudaRoot
    $env:Path = "$cudaRoot\bin;$cudaRoot\libnvvp;$env:Path"
}

python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU YOK')"
python train_supervision.py -c config/custom/unetformer.py
