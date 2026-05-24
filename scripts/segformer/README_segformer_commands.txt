0. Ortam kontrolu ve gerekiyorsa SegFormer destekli segmentation_models_pytorch kurulumu
Mevcut geoseg ortaminda once kontrol et:
& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\check_segformer_environment.py"

Mevcut kontrolde geoseg ortami Python 3.8 ve segmentation_models_pytorch 0.3.3 oldugu icin smp.Segformer yoktur. SegFormer icin Python >= 3.9 onerilir.

Secenek A - mevcut ortam Python 3.9+ ise guncelle:
& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" -m pip install -U segmentation-models-pytorch timm

Secenek B - mevcut ortam Python 3.8 ise yeni ortam olustur:
conda create -n segformer_smp python=3.10 -y
conda activate segformer_smp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -U segmentation-models-pytorch timm pillow matplotlib tqdm numpy

Yeni ortam kullanilirsa asagidaki komutlarda python yolu yerine su yol kullanilabilir:
C:\Users\CBS-7\anaconda3\envs\segformer_smp\python.exe

1. Egitim
PowerShell komutu:
& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\smp_segformer_train.py" --encoder mit_b2 --encoder-weights imagenet --image-size 512 --epochs 50 --batch-size 4 --num-workers 0

Beklenen ciktilar:
C:\Users\CBS-7\Desktop\opencode\segformer\best.pth
C:\Users\CBS-7\Desktop\opencode\segformer\last.pth

2. Test, metrik, confusion matrix, prediction ve triplet gorselleri
PowerShell komutu:
& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\smp_segformer_test.py" --checkpoint "C:\Users\CBS-7\Desktop\opencode\segformer\best.pth" --output-dir "C:\Users\CBS-7\Desktop\opencode\segformer\test_results" --batch-size 4 --num-workers 0 --save-predictions --save-triplets

Beklenen ciktilar:
C:\Users\CBS-7\Desktop\opencode\segformer\test_results\test_metrics.txt
C:\Users\CBS-7\Desktop\opencode\segformer\test_results\confusion_matrix_counts.csv
C:\Users\CBS-7\Desktop\opencode\segformer\test_results\confusion_matrix_normalized.csv
C:\Users\CBS-7\Desktop\opencode\segformer\test_results\confusion_matrix_normalized.png
C:\Users\CBS-7\Desktop\opencode\segformer\test_results\predictions
C:\Users\CBS-7\Desktop\opencode\segformer\test_results\triplets

3. XAI: Grad-CAM, Grad-CAM++, Eigen-CAM
PowerShell komutu:
& "C:\Users\CBS-7\anaconda3\envs\geoseg\python.exe" "C:\Users\CBS-7\Desktop\opencode\segformer\xai_segformer.py" --checkpoint "C:\Users\CBS-7\Desktop\opencode\segformer\best.pth" --triplet-dir "C:\Users\CBS-7\Desktop\opencode\segformer\test_results\triplets" --output-root "C:\Users\CBS-7\Desktop\opencode\segformer\xai_results" --limit 10 --methods gradcam gradcamplusplus eigencam

Beklenen ciktilar:
C:\Users\CBS-7\Desktop\opencode\segformer\xai_results\gradcam
C:\Users\CBS-7\Desktop\opencode\segformer\xai_results\gradcamplusplus
C:\Users\CBS-7\Desktop\opencode\segformer\xai_results\eigencam
