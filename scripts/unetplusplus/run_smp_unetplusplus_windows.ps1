$ErrorActionPreference = "Stop"

python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU YOK')"

python smp_unetplusplus_train.py `
  --epochs 50 `
  --batch-size 4 `
  --image-size 512 `
  --encoder resnet34 `
  --encoder-weights imagenet `
  --output-dir C:\Users\U-III\Desktop\geoseg\unetplusplus
