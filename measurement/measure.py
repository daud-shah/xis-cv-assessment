"""
Pixel-to-MM Measurement Pipeline
XIS Computer Vision Assessment
Step 3: Real-world metric measurement from calibrated images
Camera: Infinix Note 50
"""

import cv2
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

# ─── CONFIGURATION ────────────────────────────────────────────
MODEL_PATH       = r"D:\MY WORK\Project\xis-cv-assessment\models\best_model"
CALIBRATION_PATH = r"D:\MY WORK\Project\xis-cv-assessment\Calibration\calibration.pkl"
OUTPUT_DIR       = "measurement/outputs"

# Ground truth card dimensions (ISO/IEC 7810 ID-1 standard)
CARD_REAL_WIDTH_MM  = 85.6
CARD_REAL_HEIGHT_MM = 54.0

# Your measured dimensions
CARD_MEASURED_WIDTH_MM  = 85.0
CARD_MEASURED_HEIGHT_MM = 53.0

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5
# ──────────────────────────────────────────────────────────────


def load_calibration(calibration_path: str) -> dict:
    """Load camera calibration parameters from pkl file."""
    with open(calibration_path, "rb") as f:
        calib_data = pickle.load(f)
    print(f"Calibration loaded:")
    print(f"  Reprojection error: {calib_data['reprojection_error']:.4f} px")
    print(f"  Image size:         {calib_data['image_size']}")
    return calib_data


def load_model(model_path: str):
    """Load trained Mask2Former model and processor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from: {model_path}")
    print(f"Device: {device}")

    processor = Mask2FormerImageProcessor.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    print("Model loaded successfully.")
    return model, processor, device


def undistort_image(image_bgr: np.ndarray, calib_data: dict) -> np.ndarray:
    """
    Apply camera undistortion to remove lens distortion.
    This is mandatory before any measurement.
    Raw distorted images produce incorrect pixel-to-mm conversions
    because lens distortion causes geometric deformation that
    directly translates to measurement errors.
    """
    camera_matrix = calib_data["camera_matrix"]
    dist_coeffs   = calib_data["dist_coeffs"]
    undistorted   = cv2.undistort(image_bgr, camera_matrix, dist_coeffs)
    return undistorted


def get_segmentation_mask(
    image_bgr: np.ndarray,
    model,
    processor,
    device,
    image_size: int = 512,
) -> np.ndarray:
    """
    Run Mask2Former inference and return binary segmentation mask.
    Uses direct logit extraction for reliable single-class detection.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    encoding     = processor(images=pil_image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask   = encoding["pixel_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )

    # Direct logit extraction
    class_logits = outputs.class_queries_logits
    mask_logits  = outputs.masks_queries_logits

    card_scores = class_logits[0, :, 0].softmax(dim=0)
    best_query  = card_scores.argmax().item()
    confidence  = card_scores[best_query].item()

    mask = mask_logits[0, best_query]
    mask = torch.sigmoid(mask).cpu().numpy()

    h, w      = image_bgr.shape[:2]
    mask_full = cv2.resize(mask, (w, h))
    binary    = (mask_full > 0.5).astype(np.uint8) * 255

    return binary, confidence


def extract_card_dimensions(mask: np.ndarray) -> dict:
    """
    Extract card pixel dimensions from segmentation mask.
    Uses minimum area rectangle for accurate rotated card measurement.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)

    # Minimum area rectangle handles rotated cards
    rect        = cv2.minAreaRect(largest)
    box         = cv2.boxPoints(rect)
    box         = box.astype(np.intp)

    # Width and height from rectangle
    pixel_w = rect[1][0]
    pixel_h = rect[1][1]

    # Ensure width > height convention
    if pixel_w < pixel_h:
        pixel_w, pixel_h = pixel_h, pixel_w

    # Bounding box for simple measurement
    x, y, bw, bh = cv2.boundingRect(largest)

    return {
        "pixel_width"    : pixel_w,
        "pixel_height"   : pixel_h,
        "bbox_width"     : bw,
        "bbox_height"    : bh,
        "contour"        : largest,
        "rotated_box"    : box,
        "center"         : rect[0],
        "area_pixels"    : cv2.contourArea(largest),
    }


def compute_pixels_per_mm(
    pixel_width: float,
    real_width_mm: float,
    pixel_height: float,
    real_height_mm: float,
) -> float:
    """
    Compute pixels_per_mm ratio using both dimensions for accuracy.
    Average of width and height ratios gives more robust estimate.
    """
    ratio_w = pixel_width  / real_width_mm
    ratio_h = pixel_height / real_height_mm
    return (ratio_w + ratio_h) / 2.0


def pixels_to_mm(pixels: float, pixels_per_mm: float) -> float:
    """Convert pixel measurement to millimetres."""
    return pixels / pixels_per_mm


def compute_errors(
    measured_mm: float,
    ground_truth_mm: float
) -> dict:
    """Compute absolute and percentage errors."""
    abs_error  = abs(measured_mm - ground_truth_mm)
    pct_error  = (abs_error / ground_truth_mm) * 100.0
    return {
        "absolute_error_mm" : abs_error,
        "percentage_error"  : pct_error,
    }


def measure_card(
    image_path: str,
    model,
    processor,
    device,
    calib_data: dict,
    ground_truth_w: float = CARD_REAL_WIDTH_MM,
    ground_truth_h: float = CARD_REAL_HEIGHT_MM,
    save_output: bool = True,
) -> dict:
    """
    Full end-to-end measurement pipeline for a single image.

    Steps:
    1. Load image
    2. Undistort using calibration parameters
    3. Run Mask2Former segmentation
    4. Extract pixel dimensions
    5. Compute pixels_per_mm ratio
    6. Convert to millimetres
    7. Compute error against ground truth
    8. Save annotated output

    Args:
        image_path:     Path to input image
        model:          Loaded Mask2Former model
        processor:      Mask2Former processor
        device:         torch device
        calib_data:     Camera calibration data
        ground_truth_w: Real card width in mm
        ground_truth_h: Real card height in mm
        save_output:    Save annotated image

    Returns:
        dict with all measurements and errors
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")

    # Step 1: Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  ERROR: Cannot load image")
        return None

    # Step 2: Undistort
    undistorted = undistort_image(image_bgr, calib_data)
    print(f"  Undistortion applied")

    # Step 3: Segmentation
    mask, confidence = get_segmentation_mask(
        undistorted, model, processor, device
    )
    print(f"  Segmentation confidence: {confidence:.4f}")

    # Step 4: Extract dimensions
    dims = extract_card_dimensions(mask)
    if dims is None:
        print(f"  ERROR: No card detected")
        return None

    print(f"  Pixel width:  {dims['pixel_width']:.1f} px")
    print(f"  Pixel height: {dims['pixel_height']:.1f} px")

    # Step 5: Compute pixels_per_mm
    pixels_per_mm = compute_pixels_per_mm(
        dims["pixel_width"],  ground_truth_w,
        dims["pixel_height"], ground_truth_h,
    )
    print(f"  Pixels per mm: {pixels_per_mm:.4f}")

    # Step 6: Convert to mm
    width_mm  = pixels_to_mm(dims["pixel_width"],  pixels_per_mm)
    height_mm = pixels_to_mm(dims["pixel_height"], pixels_per_mm)
    print(f"  Measured width:  {width_mm:.2f} mm")
    print(f"  Measured height: {height_mm:.2f} mm")

    # Step 7: Compute errors
    err_w = compute_errors(width_mm,  ground_truth_w)
    err_h = compute_errors(height_mm, ground_truth_h)
    print(f"  Width  error: {err_w['absolute_error_mm']:.2f} mm ({err_w['percentage_error']:.2f}%)")
    print(f"  Height error: {err_h['absolute_error_mm']:.2f} mm ({err_h['percentage_error']:.2f}%)")

    # Step 8: Save annotated output
    if save_output:
        annotated = undistorted.copy()

        # Draw mask overlay
        mask_color              = np.zeros_like(annotated)
        mask_color[mask > 0]    = [0, 255, 0]
        annotated               = cv2.addWeighted(
            annotated, 0.7, mask_color, 0.3, 0
        )

        # Draw rotated bounding box
        cv2.drawContours(
            annotated, [dims["rotated_box"]], 0, (0, 255, 0), 2
        )

        # Draw measurement labels
        cx, cy = int(dims["center"][0]), int(dims["center"][1])

        labels = [
            f"Width:  {width_mm:.1f} mm",
            f"Height: {height_mm:.1f} mm",
            f"Conf:   {confidence:.3f}",
        ]
        for i, label in enumerate(labels):
            y_pos = 60 + i * 60
            # Black background for readability
            cv2.putText(
                annotated, label,
                (8, y_pos + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), 6
            )
             # Green text on top
            cv2.putText(
                annotated, label,
                (8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3
                )
    

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_name = os.path.basename(image_path).replace(
            ".jpg", "_measured.jpg"
        )
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"  Saved: {out_path}")

    return {
        "image"              : os.path.basename(image_path),
        "confidence"         : confidence,
        "pixel_width"        : dims["pixel_width"],
        "pixel_height"       : dims["pixel_height"],
        "pixels_per_mm"      : pixels_per_mm,
        "width_mm"           : width_mm,
        "height_mm"          : height_mm,
        "gt_width_mm"        : ground_truth_w,
        "gt_height_mm"       : ground_truth_h,
        "width_abs_error_mm" : err_w["absolute_error_mm"],
        "width_pct_error"    : err_w["percentage_error"],
        "height_abs_error_mm": err_h["absolute_error_mm"],
        "height_pct_error"   : err_h["percentage_error"],
    }


def run_accuracy_validation(
    image_folder: str,
    model,
    processor,
    device,
    calib_data: dict,
) -> None:
    """
    Run measurement on 10+ card images and compute MAE and MPE.
    Required by Step 3 of the assessment.
    """
    import glob

    images = (
        glob.glob(os.path.join(image_folder, "*.jpg")) +
        glob.glob(os.path.join(image_folder, "*.JPG")) +
        glob.glob(os.path.join(image_folder, "*.png"))
    )

    # Remove duplicates from case-insensitive matches on Windows
    images = list(set(os.path.normpath(p) for p in images))

    if len(images) == 0:
        print(f"No images found in {image_folder}")
        return

    print(f"\nRunning accuracy validation on {len(images)} images...")
    print("="*60)

    results = []
    for img_path in images:
        result = measure_card(
            img_path, model, processor, device, calib_data
        )
        if result:
            results.append(result)

    if len(results) == 0:
        print("No valid measurements obtained.")
        return

    # Compute MAE and MPE
    mae_width  = np.mean([r["width_abs_error_mm"]  for r in results])
    mae_height = np.mean([r["height_abs_error_mm"] for r in results])
    mpe_width  = np.mean([r["width_pct_error"]     for r in results])
    mpe_height = np.mean([r["height_pct_error"]    for r in results])

    print("\n" + "="*60)
    print("ACCURACY VALIDATION REPORT")
    print("="*60)
    print(f"Total images measured:    {len(results)}")
    print(f"\nWidth  MAE:  {mae_width:.2f} mm")
    print(f"Height MAE:  {mae_height:.2f} mm")
    print(f"Width  MPE:  {mpe_width:.2f}%")
    print(f"Height MPE:  {mpe_height:.2f}%")

    print("\nPer-image results:")
    print(f"{'Image':<30} {'W_mm':>8} {'H_mm':>8} {'W_err':>8} {'H_err':>8}")
    print("-"*64)
    for r in results:
        print(
            f"{r['image']:<30} "
            f"{r['width_mm']:>7.1f} "
            f"{r['height_mm']:>7.1f} "
            f"{r['width_abs_error_mm']:>7.2f} "
            f"{r['height_abs_error_mm']:>7.2f}"
        )

    # Save accuracy report
    report = {
        "total_images"  : len(results),
        "mae_width_mm"  : mae_width,
        "mae_height_mm" : mae_height,
        "mpe_width_pct" : mpe_width,
        "mpe_height_pct": mpe_height,
        "results"       : results,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "accuracy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAccuracy report saved: {report_path}")

    plot_accuracy_results(results, mae_width, mae_height)


def plot_accuracy_results(results, mae_width, mae_height):
    """Plot measurement results vs ground truth."""

    images   = [r["image"][:15] for r in results]
    w_meas   = [r["width_mm"]   for r in results]
    h_meas   = [r["height_mm"]  for r in results]
    w_gt     = [r["gt_width_mm"]  for r in results]
    h_gt     = [r["gt_height_mm"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Measurement Accuracy — System vs Ground Truth",
        fontsize=14, fontweight="bold"
    )

    x = range(len(results))

    axes[0].bar(x, w_meas, alpha=0.7, label="Measured", color="#3498db")
    axes[0].axhline(
        y=CARD_REAL_WIDTH_MM, color="red",
        linestyle="--", label=f"Ground Truth ({CARD_REAL_WIDTH_MM}mm)"
    )
    axes[0].set_title(f"Width (MAE: {mae_width:.2f} mm)")
    axes[0].set_ylabel("Width (mm)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(images, rotation=45, ha="right", fontsize=7)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, h_meas, alpha=0.7, label="Measured", color="#2ecc71")
    axes[1].axhline(
        y=CARD_REAL_HEIGHT_MM, color="red",
        linestyle="--", label=f"Ground Truth ({CARD_REAL_HEIGHT_MM}mm)"
    )
    axes[1].set_title(f"Height (MAE: {mae_height:.2f} mm)")
    axes[1].set_ylabel("Height (mm)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(images, rotation=45, ha="right", fontsize=7)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "accuracy_chart.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    print("Accuracy chart saved.")


if __name__ == "__main__":

    # Load calibration and model
    calib_data           = load_calibration(CALIBRATION_PATH)
    model, processor, device = load_model(MODEL_PATH)

    # Run validation on test images
    # Point this to your card images folder
    TEST_IMAGES_FOLDER = r"D:\MY WORK\Project\xis-cv-assessment\measurement\test_images"

    run_accuracy_validation(
        TEST_IMAGES_FOLDER,
        model, processor, device,
        calib_data,
    )