# XIS AI Assessment — Card Segmentation & Metric Measurement

**Candidate:** Daud Shah
**Role:** Computer Vision Engineer
**Company:** XIS.AI
**Submission Date:** June 12, 2026

---

## Project Overview

This repository contains a complete end-to-end computer vision pipeline that:

- Segments a physical transport card using a fine-tuned Mask2Former model
- Applies intrinsic camera calibration to remove lens distortion from all images
- Computes real-world width and height measurements in millimetres from pixel data
- Validates measurement accuracy against ISO/IEC 7810 ground truth dimensions
- Provides a Gradio web interface for live end-to-end demonstration

---

## Object Selected

**Zoo Peshawar Bus Transport Cards** (green and yellow variants)

- Real-world dimensions: 85.6 mm × 54.0 mm (ISO/IEC 7810 ID-1 standard)
- Single class: `card`
- Two colour variants for dataset diversity

---

## Key Results

| Metric | Value |
|---|---|
| Camera reprojection error | 0.3928 px (GOOD) |
| Model mAP@0.5 | 1.0000 |
| Model mAP@0.5:0.95 | 1.0000 |
| Mean IoU | 0.9832 |
| Width MAE | 0.74 mm |
| Height MAE | 0.47 mm |
| Width MPE | 0.87% |
| Height MPE | 0.87% |

---

## Repository Structure

```
xis-cv-assessment/
│
├── Calibration/
│   ├── images/                          # 26 checkerboard calibration photos
│ 
│   ├── Calibration_checkerboard.ipynb   # Calibration notebook
│   ├── calibration.pkl                  # Saved camera parameters
│   ├── calibration_report.txt           # Calibration results
│   └── undistortion_comparison.png      # Before/after undistortion visual
│
├── dataset/
│   └── card segment.v1i.coco-segmentation/
│       ├── train/                       # 150 augmented training images
│       ├── valid/                       # 14 validation images
│       ├── test/                        # 7 test images
│       └── raw-images/                  # 71 original collected images
│
├── models/
│   ├── best_model/                      # Saved Mask2Former weights
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   ├── traning_code/
│   │   └── mask2former-training.ipynb   # Full Kaggle training notebook
│   ├── training_log.json                # Epoch-by-epoch metrics
│   └── test_results.json               # Final test evaluation
│
├── inference/
│   ├── predict.py                       # Inference script
│   └── outputs/                         # Sample annotated results
│
├── measurement/
│   ├── measure.py                       # Pixel-to-mm pipeline
│   ├── test_images/                     # 8 validation photos
│   └── outputs/
│       ├── accuracy_report.json
│       ├── accuracy_chart.png
│       └── *_measured.jpg
│
├── interface/
│   └── app.py                           # Gradio web interface
│
├── docs/
│   ├── CALIBRATION_REPORT.md
│   ├── DATASET_CARD.md
│   ├── TRAINING_REPORT.md
│   ├── MEASUREMENT_REPORT.md
│   └── SETUP.md
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/daudshah/xis-cv-assessment.git
cd xis-cv-assessment

# 2. Create environment
conda create -n xis_cv python=3.10
conda activate xis_cv
pip install -r requirements.txt

# 3. Run camera calibration
python Calibration/calibrate.py

# 4. Run inference on single image
python inference/predict.py --image path/to/card.jpg

# 5. Run inference on folder
python inference/predict.py --folder path/to/images/

# 6. Run measurement pipeline
python measurement/measure.py

# 7. Launch Gradio interface
python interface/app.py
# Open http://127.0.0.1:7860
```

---

## Pipeline Architecture

```
Infinix Note 50 Camera
        │
        ▼
Checkerboard Photos (26 images)
        │
        ▼
calibrate.py → calibration.pkl (reprojection error: 0.3928 px)
        │
        ▼
Card Photos (71 raw images)
        │
        ▼
cv2.undistort() applied to all images
        │
        ▼
Roboflow Labelling → COCO Segmentation Format
(150 train / 14 val / 7 test)
        │
        ▼
Mask2Former Fine-tuning on Kaggle T4 GPU
(50 epochs, mAP@0.5: 1.0, IoU: 0.9832)
        │
        ▼
Segmentation Mask → Minimum Area Rectangle
        │
        ▼
pixels_per_mm ratio → mm Measurement
(MAE: 0.74mm, MPE: 0.87%)
        │
        ▼
Annotated Output + Accuracy Report + Gradio Interface
```

---

## Technology Stack

| Component | Tool |
|---|---|
| Camera Calibration | OpenCV 4.x |
| Data Collection | Infinix Note 50 |
| Data Labelling | Roboflow |
| Segmentation Model | Mask2Former (Swin-Small) |
| Training Framework | PyTorch + HuggingFace Transformers |
| Training Hardware | Kaggle T4 GPU |
| Measurement | OpenCV + NumPy |
| Interface | Gradio |
| Version Control | Git + GitHub |

---

## Documentation

| File | Contents |
|---|---|
| CALIBRATION_REPORT.md | Camera matrix, distortion coefficients, reprojection error |
| DATASET_CARD.md | Object selection, collection strategy, augmentation, statistics |
| TRAINING_REPORT.md | Model selection, hyperparameters, metrics, challenges |
| MEASUREMENT_REPORT.md | Pixel-to-mm methodology, error table, limitations |
| SETUP.md | Full installation and run instructions |
