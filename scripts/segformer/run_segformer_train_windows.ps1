$ErrorActionPreference = "Stop"

& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\smp_segformer_train.py" `
  --encoder mit_b2 `
  --encoder-weights imagenet `
  --image-size 512 `
  --epochs 50 `
  --batch-size 4 `
  --num-workers 0
