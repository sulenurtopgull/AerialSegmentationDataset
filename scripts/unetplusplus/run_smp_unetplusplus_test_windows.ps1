$ErrorActionPreference = "Stop"

python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU YOK')"

python smp_unetplusplus_test.py `
  --checkpoint C:\Users\U-III\Desktop\geoseg\unetplusplus\best.pth `
  --batch-size 4 `
  --save-predictions `
  --save-triplets `
  --output-dir C:\Users\U-III\Desktop\geoseg\unetplusplus\test_results
