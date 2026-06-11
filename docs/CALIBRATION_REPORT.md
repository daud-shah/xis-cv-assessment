# Camera Calibration Report

**Camera:** Infinix Note 50
**Date:** June 11, 2026
**Method:** Checkerboard Intrinsic Calibration (OpenCV)

---

## 1. Calibration Objective

Lens distortion in phone cameras introduces geometric deformation into every captured image. This deformation causes pixel measurements to deviate from real-world distances. Specifically, the Infinix Note 50 exhibits barrel distortion where straight lines appear curved toward the edges of the frame.

Intrinsic calibration computes the camera matrix and distortion coefficients that mathematically model this deformation. Once known, `cv2.undistort()` removes the distortion from any subsequent image, restoring geometric accuracy required for metric measurement.

---

## 2. Calibration Target

| Parameter | Value |
|---|---|
| Pattern type | Checkerboard |
| Grid size | 9×7 squares |
| Inner corners | 8×6 |
| Square size | 15 mm |
| Source | calib.io pattern generator |

---

## 3. Image Collection

| Parameter | Value |
|---|---|
| Total images captured | 26 |
| Successful detections | 17 |
| Failed detections | 9 |
| Image resolution | 4064×3048 px |

Images were captured with the Infinix Note 50 from varied angles, distances, and positions. The checkerboard was displayed on a laptop screen for the initial capture set. Failed images resulted from motion blur or partial board occlusion and were automatically excluded by OpenCV.

---

## 4. Calibration Results

### Camera Matrix (Intrinsic Matrix K)

```
K = [[3002.57    0.00    2024.69]
     [   0.00  3015.24  1532.58]
     [   0.00     0.00     1.00]]
```

| Parameter | Value | Description |
|---|---|---|
| fx | 3002.57 px | Horizontal focal length |
| fy | 3015.24 px | Vertical focal length |
| cx | 2024.69 px | Principal point X |
| cy | 1532.58 px | Principal point Y |

### Distortion Coefficients

```
D = [k1=0.0820, k2=-0.6147, p1=0.0008, p2=-0.0020, k3=0.9276]
```

| Coefficient | Value | Type |
|---|---|---|
| k1 | 0.0820 | Radial (barrel distortion present) |
| k2 | -0.6147 | Radial correction |
| p1 | 0.0008 | Tangential (near zero, good) |
| p2 | -0.0020 | Tangential (near zero, good) |
| k3 | 0.9276 | Higher-order radial |

---

## 5. Reprojection Error

| Metric | Value | Quality |
|---|---|---|
| Mean reprojection error | **0.3928 px** | **GOOD** |

The reprojection error measures how accurately the computed camera model can project 3D checkerboard corners back to their detected 2D positions. A value of 0.3928 px is within the acceptable range defined by the assessment (below 0.5 px).

---

## 6. Distortion Analysis

The k1 coefficient of 0.082 confirms mild barrel distortion in the Infinix Note 50 lens. Without correction, a card placed at 30 cm distance would show approximately 2 to 3 mm measurement error due to this distortion. After applying `cv2.undistort()`, this systematic error is eliminated.

---

## 7. Why Raw Images Produce Incorrect Measurements

Raw distorted images cannot be used for accurate metric measurement for three reasons:

**Radial distortion** causes pixel coordinates near image edges to shift inward or outward relative to their true geometric position. A card placed near the edge of frame appears smaller or larger than it truly is.

**Non-uniform distortion** means the error is not constant. A card at the image centre experiences less distortion than the same card at the corner, making calibration-free measurements inconsistent.

**Pixel-to-mm ratio instability** occurs because the apparent size of an object in pixels depends on its position in the distorted frame. Without undistortion, the pixels_per_mm ratio computed at one position does not hold at another.

Applying `cv2.undistort()` using the stored camera matrix and distortion coefficients corrects all three effects, producing geometrically accurate images where pixel distances correspond reliably to real-world distances.

---

## 8. Undistortion Application

```python
import cv2
import pickle

with open("calibration/calibration.pkl", "rb") as f:
    calib = pickle.load(f)

undistorted = cv2.undistort(
    image,
    calib["camera_matrix"],
    calib["dist_coeffs"]
)
```

This operation is applied to every image before segmentation or measurement in the pipeline.
