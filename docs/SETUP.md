# Setup and Installation Guide

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.9 | 3.10 |
| RAM | 8 GB | 16 GB |
| GPU | Optional (CPU supported) | NVIDIA GPU with 8GB+ VRAM |
| Storage | 5 GB free | 10 GB free |

---

## 1. Clone Repository

```bash
git clone https://github.com/daudshah/xis-cv-assessment.git
cd xis-cv-assessment
```

---

## 2. Environment Setup

### Option A: Conda (Recommended)

```bash
conda create -n xis_cv python=3.10
conda activate xis_cv
pip install -r requirements.txt
```

### Option B: Pip virtualenv

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Requirements

The `requirements.txt` file contains:

```
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.40.0
accelerate>=0.20.0
pycocotools>=2.0.6
roboflow>=1.1.0
gradio>=4.0.0
matplotlib>=3.7.0
Pillow>=9.5.0
scipy>=1.10.0
albumentations>=1.3.0
```

Install all:

```bash
pip install -r requirements.txt
```

---

## 4. Run Camera Calibration

Place your calibration images in `calibration/images/` then run:

```bash
python calibration/calibrate.py
```

**Expected output:**
```
Found 26 calibration images
Processing...
  Image 01: SUCCESS - corners detected
  ...
Reprojection Error: 0.3928 px
Quality: GOOD
Calibration saved: calibration/calibration.pkl
```

The saved `calibration.pkl` file is used by all downstream scripts.

---

## 5. Run Measurement Pipeline

Place card images in `measurement/test_images/` then run:

```bash
python measurement/measure.py
```

**Expected output:**
```
Calibration loaded. Reprojection error: 0.3928 px
Model loaded successfully.
Running accuracy validation on 8 images...
Processing: IMG_xxxxx.jpg
  Undistortion applied
  Segmentation confidence: 0.9044
  Pixel width:  2309.6 px
  Pixel height: 1487.6 px
  Pixels per mm: 27.2646
  Measured width:  84.71 mm
  Measured height: 54.56 mm
  Width  error: 0.89 mm (1.04%)
  Height error: 0.56 mm (1.04%)
Width  MAE:  0.74 mm
Height MAE:  0.47 mm
```

Annotated images saved to `measurement/outputs/`.

---

## 6. Launch Gradio Interface

```bash
python interface/app.py
```

Open `http://127.0.0.1:7860` in your browser. Upload any card image and click **Measure Card** to run the full pipeline and see results.

---

## 7. Training (Optional Reproduction)

Training was performed on Kaggle with T4 GPU. To reproduce:

1. Upload `mask2former_training.ipynb` to Kaggle
2. Enable GPU T4x2 accelerator in Settings
3. Set your Roboflow API key in Cell 3
4. Run all cells

Training completes in approximately 60 minutes. Best model weights are saved automatically.

---

## 8. Project File Paths

All hardcoded paths in the scripts use the following structure. Update if your installation directory differs:

```python
MODEL_PATH       = r"path/to/xis-cv-assessment/models/weights/best_model"
CALIBRATION_PATH = r"path/to/xis-cv-assessment/calibration/calibration.pkl"
```

---

## 9. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No images found` | Wrong folder path | Check TEST_IMAGES_FOLDER in measure.py |
| `CUDA out of memory` | GPU too small | Reduce batch_size to 1 in config |
| `calibration.pkl not found` | Calibration not run | Run calibrate.py first |
| `conda not recognized` | Anaconda not in PATH | Use full path to activate.bat |
| `Module not found` | Dependencies missing | Run pip install -r requirements.txt |
