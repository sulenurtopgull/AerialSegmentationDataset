$ErrorActionPreference = "Stop"

& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\xai_segformer.py" `
  --checkpoint "C:\Users\CBS-7\Desktop\opencode\segformer\best.pth" `
  --triplet-dir "C:\Users\CBS-7\Desktop\opencode\segformer\test_results\triplets" `
  --output-root "C:\Users\CBS-7\Desktop\opencode\segformer\xai_results" `
  --limit 10 `
  --methods gradcam gradcamplusplus eigencam
