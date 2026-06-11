"""
Inference Script
XIS Computer Vision Assessment
Accepts a single image or a folder of images.
Runs undistortion, Mask2Former inference, outputs annotated results.

Usage:
    Single image:
        python inference/predict.py --image path/to/image.jpg

    Folder of images:
        python inference/predict.py --folder path/to/images/

    Custom model/calibration paths:
        python inference/predict.py --image img.jpg \
            --model_path path/to/model \
            --calibration_path path/to/calibration.pkl
"""

import cv2
import numpy as np
import pickle
import torch
import os
import glob
import argparse
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

# ─── DEFAULT CONFIGURATION ────────────────────────────────────
MODEL_PATH       = r"D:\MY WORK\Project\xis-cv-assessment\models\best_model"
CALIBRATION_PATH = r"D:\MY WORK\Project\xis-cv-assessment\Calibration\calibration.pkl"
OUTPUT_DIR       = "inference/outputs"
# ──────────────────────────────────────────────────────────────


def load_calibration(path: str) -> dict:
    """Load saved camera calibration parameters from pkl file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path, "rb") as f:
        calib = pickle.load(f)
    print(f"Calibration loaded | Reprojection error: {calib['reprojection_error']:.4f} px")
    return calib


def load_model(model_path: str):
    """Load fine-tuned Mask2Former model and processor."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = Mask2FormerImageProcessor.from_pretrained(
        model_path, ignore_mismatched_sizes=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_path, ignore_mismatched_sizes=True
    ).to(device)
    model.eval()
    print("Model loaded successfully.")
    return model, processor, device


def run_inference(
    image_path: str,
    model,
    processor,
    device,
    calib_data: dict,
    save_output: bool = True,
) -> dict:
    """
    Full inference pipeline for a single image.

    Steps:
    1. Load image from disk
    2. Apply undistortion using calibration parameters
    3. Run Mask2Former forward pass
    4. Extract segmentation mask via direct logit extraction
    5. Annotate and save output image

    Args:
        image_path:   Path to input image (.jpg or .png)
        model:        Loaded Mask2Former model
        processor:    Mask2Former image processor
        device:       torch device (cuda or cpu)
        calib_data:   Camera calibration dictionary
        save_output:  Save annotated image to OUTPUT_DIR

    Returns:
        dict: image_path, output_path, confidence, undistorted flag
    """
    print(f"\nProcessing: {os.path.basename(image_path)}")

    # Step 1: Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  ERROR: Cannot load image from {image_path}")
        return None

    print(f"  Resolution: {image_bgr.shape[1]}x{image_bgr.shape[0]} px")

    # Step 2: Undistort
    undistorted = cv2.undistort(
        image_bgr,
        calib_data["camera_matrix"],
        calib_data["dist_coeffs"]
    )
    print(f"  Undistortion applied")

    # Step 3: Prepare for model
    h, w      = undistorted.shape[:2]
    image_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    encoding     = processor(images=pil_image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask   = encoding["pixel_mask"].to(device)

    # Step 4: Run inference
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )

    # Direct logit extraction - reliable for single class
    class_logits = outputs.class_queries_logits
    mask_logits  = outputs.masks_queries_logits
    card_scores  = class_logits[0, :, 0].softmax(dim=0)
    best_query   = card_scores.argmax().item()
    confidence   = card_scores[best_query].item()

    mask      = torch.sigmoid(mask_logits[0, best_query]).cpu().numpy()
    mask_full = cv2.resize(mask, (w, h))
    binary    = (mask_full > 0.5).astype(np.uint8) * 255

    print(f"  Confidence: {confidence:.4f}")

    # Step 5: Annotate and save
    output_path = None
    if save_output:
        annotated              = undistorted.copy()
        mask_color             = np.zeros_like(annotated)
        mask_color[binary > 0] = [0, 255, 0]
        annotated              = cv2.addWeighted(
            annotated, 0.7, mask_color, 0.3, 0
        )

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 3)

        line1 = f"Card Detected"
        line2 = f"Confidence: {confidence:.3f}"
        line3 = f"Undistorted: Yes"

        for i, line in enumerate([line1, line2, line3]):
            y = 50 + i * 55
            cv2.putText(annotated, line, (8, y+2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
            cv2.putText(annotated, line, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base        = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}_inference.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"  Saved: {output_path}")

    return {
        "image_path" : image_path,
        "output_path": output_path,
        "confidence" : confidence,
        "undistorted": True,
    }


def run_on_folder(folder_path, model, processor, device, calib_data):
    """Run inference on all images in a folder."""
    images = (
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.JPG")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )
    images = list(set(os.path.normpath(p) for p in images))

    if len(images) == 0:
        print(f"No images found in: {folder_path}")
        return

    print(f"Found {len(images)} images in folder.")
    print("="*50)

    results  = []
    success  = 0
    failed   = 0

    for img_path in sorted(images):
        result = run_inference(
            img_path, model, processor, device, calib_data
        )
        if result:
            results.append(result)
            success += 1
        else:
            failed += 1

    print("\n" + "="*50)
    print(f"FOLDER INFERENCE COMPLETE")
    print(f"Successful: {success}/{len(images)}")
    print(f"Failed:     {failed}/{len(images)}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="XIS CV Assessment - Card Segmentation Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python inference/predict.py --image path/to/card.jpg

  Folder of images:
    python inference/predict.py --folder path/to/images/

  Custom paths:
    python inference/predict.py --image card.jpg \\
        --model_path models/weights/best_model \\
        --calibration_path calibration/calibration.pkl
        """
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Path to a single input image"
    )
    group.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing images"
    )

    # Optional path overrides
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--calibration_path",
        type=str,
        default=CALIBRATION_PATH,
        help="Path to calibration.pkl file"
    )

    args = parser.parse_args()

    print("XIS CV Assessment - Inference Pipeline")
    print("="*50)

    # Load calibration and model
    calib_data            = load_calibration(args.calibration_path)
    model, processor, device = load_model(args.model_path)

    # Run inference
    if args.image:
        result = run_inference(
            args.image, model, processor, device, calib_data
        )
        if result:
            print("\n" + "="*50)
            print("INFERENCE COMPLETE")
            print(f"Confidence:  {result['confidence']:.4f}")
            print(f"Output:      {result['output_path']}")
    else:
        run_on_folder(
            args.folder, model, processor, device, calib_data
        )


if __name__ == "__main__":
    main()