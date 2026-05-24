$ErrorActionPreference = "Stop"

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU YOK')"

python unetformer_custom_test.py `
  --checkpoint model_weights\custom\unetformer-custom-512crop-e50\best.pth `
  --output-dir unetformer\test_results `
  --batch-size 8 `
  --save-predictions `
  --save-triplets
