$ErrorActionPreference = "Stop"

& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\smp_segformer_test.py" `
  --checkpoint "C:\Users\CBS-7\Desktop\opencode\segformer\best.pth" `
  --output-dir "C:\Users\CBS-7\Desktop\opencode\segformer\test_results" `
  --batch-size 4 `
  --num-workers 0 `
  --save-predictions `
  --save-triplets
