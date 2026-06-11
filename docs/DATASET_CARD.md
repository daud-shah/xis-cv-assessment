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
