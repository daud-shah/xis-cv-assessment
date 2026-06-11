# Dataset Card

**Dataset Name:** Zoo Peshawar Bus Transport Card Segmentation
**Version:** 1.0
**Created:** June 11, 2026
**Labelling Tool:** Roboflow
**Task:** Instance Segmentation

---

## 1. Object Selection

**Object:** Zoo Peshawar Bus Transport Cards

**Justification for selection:**

The Zoo Peshawar bus card was selected for the following reasons:

- **Availability:** Multiple cards in two distinct colours (green and yellow) were immediately available, enabling colour diversity in the dataset without additional acquisition cost.
- **Geometry:** The card is a rigid, flat rectangle with clearly defined edges, making it ideal for segmentation and accurate bounding box extraction.
- **Known dimensions:** As a standard ISO/IEC 7810 ID-1 format card (85.6 × 54.0 mm), ground truth dimensions are precisely known, enabling rigorous measurement validation.
- **Labelling ease:** The rectangular shape with high contrast against most backgrounds allows precise polygon annotation with minimal ambiguity.

**Real-world dimensions:**

| Dimension | Value |
|---|---|
| Width | 85.6 mm |
| Height | 54.0 mm |
| Standard | ISO/IEC 7810 ID-1 |

---

## 2. Data Collection Strategy

All images were captured using the **Infinix Note 50** rear camera after completing camera calibration. The calibrated camera was used exclusively to ensure all images are compatible with the undistortion pipeline.

**Collection parameters:**

| Parameter | Value |
|---|---|
| Camera | Infinix Note 50 |
| Resolution | 4064×3048 px |
| Total images collected | 71 |
| Collection duration | 30 minutes |
| Environment | Indoor and outdoor |

**Variation strategy:**

To maximise dataset diversity and model generalisation, images were captured with deliberate variation across the following dimensions:

- **Distance:** close (card fills 70% of frame), medium (40%), far (20%)
- **Angle:** flat, tilted left, tilted right, tilted up, tilted down
- **Orientation:** card horizontal, card vertical
- **Background:** wooden table, floor tiles, white paper, dark surface, carpet, book cover
- **Lighting:** natural daylight, indoor fluorescent, mixed lighting
- **Card colour:** green variant (35 images), yellow variant (36 images)

---

## 3. Labelling

**Tool:** Roboflow (polygon segmentation)

**Method:** Each image was manually annotated by drawing a precise polygon mask around the card boundary. Annotations follow the card outline exactly, including rounded corners where visible.

**Class definition:**

| Class ID | Class Name | Description |
|---|---|---|
| 0 | card | Zoo Peshawar bus transport card (both colours) |

Both the green and yellow card variants are labelled under the single `card` class. This design decision ensures the model learns card shape rather than card colour, improving generalisation to unseen card variants.

---

## 4. Dataset Statistics

### Split Distribution

| Split | Images | Percentage |
|---|---|---|
| Train | 50 | 70% |
| Validation | 14 | 20% |
| Test | 7 | 10% |
| **Total** | **71** | **100%** |

### After Augmentation

| Split | Images (augmented) |
|---|---|
| Train | ~150 |
| Validation | 14 |
| Test | 7 |

---

## 5. Preprocessing

| Step | Setting | Reason |
|---|---|---|
| Auto-Orient | Applied | Corrects phone photo rotation metadata |
| Resize | 640×640 (stretch) | Standard input size for segmentation models |

---

## 6. Augmentation Strategy

Augmentation was applied to the training split only to artificially increase dataset size and improve model robustness to real-world conditions.

| Augmentation | Setting | Justification |
|---|---|---|
| Flip Horizontal | On | Card valid in either orientation |
| Flip Vertical | On | Simulates overhead camera angles |
| Rotation | ±15° | Cards appear at slight angles in real use |
| Shear | ±10° horizontal and vertical | Simulates perspective variation |
| Brightness | ±25% | Handles indoor and outdoor lighting |
| Exposure | ±15% | Handles over and underexposed shots |
| Blur | Up to 1.5 px | Robustness to slight camera motion |

---

## 7. Export Format

**Format:** COCO Segmentation JSON

Dataset exported from Roboflow via API in COCO segmentation format for compatibility with HuggingFace Transformers and the Mask2Former training pipeline.

---

## 8. Roboflow Dataset

**Roboflow Project:** Card Segment  
**Workspace:** object-detection-sn8ac  
**License:** CC BY 4.0  
**Dataset URL:** https://universe.roboflow.com/object-detection-sn8ac/card-segment  
**Browse Images:** https://app.roboflow.com/object-detection-sn8ac/card-segment/browse

**Project Link:** [View on Roboflow Universe](https://universe.roboflow.com/object-detection-sn8ac/card-segment)

The dataset is publicly available on Roboflow Universe and can be accessed for viewing, cloning, or forking. All 71 images with their polygon segmentation annotations are hosted on the Roboflow platform.

---

## 9. Dataset Usage

### Download from Roboflow

```bash
# Install Roboflow SDK
pip install roboflow

# Download dataset
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='your_api_key')
project = rf.workspace('object-detection-sn8ac').project('card-segment')
dataset = project.versions(1).download('coco')
"
```

### Local Dataset Structure

The dataset in this repository is organized as:

```
dataset/
└── card segment.v1i.coco-segmentation/
    ├── README.dataset.txt
    ├── README.roboflow.txt
    ├── train/
    │   └── _annotations.coco.json
    ├── valid/
    │   └── _annotations.coco.json
    └── test/
        └── _annotations.coco.json
```

---

## 10. Citation

If you use this dataset in your research or project, please cite:

```bibtex
@dataset{card-segment,
  title={Card Segment Instance Segmentation Dataset},
  author={Your Name},
  year={2026},
  organization={Roboflow Universe},
  url={https://universe.roboflow.com/object-detection-sn8ac/card-segment}
}
```

---

## 11. License

This dataset is licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

**You are free to:**
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially

**Under the following terms:**
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made
