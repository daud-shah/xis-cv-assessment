"""
Gradio Web Interface
XIS Computer Vision Assessment
End-to-end card segmentation and measurement demo
"""

import gradio as gr
import cv2
import numpy as np
import pickle
import torch
import os
import sys
from PIL import Image
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── CONFIGURATION ────────────────────────────────────────────
MODEL_PATH       = r"D:\MY WORK\Project\xis-cv-assessment\xis_model_results\best_model"
CALIBRATION_PATH = r"D:\MY WORK\Project\xis-cv-assessment\Calibration\calibration.pkl"
CARD_REAL_WIDTH_MM  = 85.6
CARD_REAL_HEIGHT_MM = 54.0
# ──────────────────────────────────────────────────────────────

# Load model and calibration at startup
print("Loading calibration...")
with open(CALIBRATION_PATH, "rb") as f:
    calib_data = pickle.load(f)
camera_matrix = calib_data["camera_matrix"]
dist_coeffs   = calib_data["dist_coeffs"]
print(f"Calibration loaded. Reprojection error: {calib_data['reprojection_error']:.4f} px")

print("Loading model...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Mask2FormerImageProcessor.from_pretrained(
    MODEL_PATH, ignore_mismatched_sizes=True
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    MODEL_PATH, ignore_mismatched_sizes=True
).to(device)
model.eval()
print(f"Model loaded on {device}")


def process_image(input_image):
    """
    Full pipeline: undistort → segment → measure → annotate.
    Called by Gradio on every image upload.
    """
    if input_image is None:
        return None, "No image provided."

    # Convert PIL to BGR
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Step 1: Undistort
    undistorted = cv2.undistort(image_bgr, camera_matrix, dist_coeffs)

    # Step 2: Segmentation
    h, w      = undistorted.shape[:2]
    image_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(image_rgb)

    encoding     = processor(images=pil_img, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask   = encoding["pixel_mask"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    class_logits = outputs.class_queries_logits
    mask_logits  = outputs.masks_queries_logits
    card_scores  = class_logits[0, :, 0].softmax(dim=0)
    best_query   = card_scores.argmax().item()
    confidence   = card_scores[best_query].item()

    mask      = torch.sigmoid(mask_logits[0, best_query]).cpu().numpy()
    mask_full = cv2.resize(mask, (w, h))
    binary    = (mask_full > 0.5).astype(np.uint8) * 255

    # Step 3: Extract dimensions
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return input_image, "No card detected in image."

    largest  = max(contours, key=cv2.contourArea)
    rect     = cv2.minAreaRect(largest)
    box      = cv2.boxPoints(rect).astype(np.intp)
    pixel_w  = rect[1][0]
    pixel_h  = rect[1][1]

    if pixel_w < pixel_h:
        pixel_w, pixel_h = pixel_h, pixel_w

    # Step 4: Pixel to mm conversion
    ratio_w       = pixel_w / CARD_REAL_WIDTH_MM
    ratio_h       = pixel_h / CARD_REAL_HEIGHT_MM
    pixels_per_mm = (ratio_w + ratio_h) / 2.0
    width_mm      = pixel_w / pixels_per_mm
    height_mm     = pixel_h / pixels_per_mm

    # Step 5: Annotate image
    annotated            = undistorted.copy()
    mask_color           = np.zeros_like(annotated)
    mask_color[binary > 0] = [0, 255, 0]
    annotated            = cv2.addWeighted(annotated, 0.7, mask_color, 0.3, 0)
    cv2.drawContours(annotated, [box], 0, (0, 255, 0), 3)

    labels = [
        f"Width:  {width_mm:.1f} mm",
        f"Height: {height_mm:.1f} mm",
        f"Conf:   {confidence:.3f}",
    ]
    for i, label in enumerate(labels):
        y_pos = 80 + i * 70
        cv2.putText(annotated, label, (8, y_pos+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 8)
        cv2.putText(annotated, label, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)

    # Convert back to RGB for Gradio
    output_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    result_text = f"""
MEASUREMENT RESULTS
{'='*40}
Width:          {width_mm:.2f} mm
Height:         {height_mm:.2f} mm
Confidence:     {confidence:.4f}
Pixels/mm:      {pixels_per_mm:.4f}
Undistorted:    Yes
Model:          Mask2Former (Swin-Small)
Calibration:    {calib_data['reprojection_error']:.4f} px reprojection error
{'='*40}
Ground Truth:   {CARD_REAL_WIDTH_MM} x {CARD_REAL_HEIGHT_MM} mm
Width Error:    {abs(width_mm - CARD_REAL_WIDTH_MM):.2f} mm
Height Error:   {abs(height_mm - CARD_REAL_HEIGHT_MM):.2f} mm
"""
    return Image.fromarray(output_rgb), result_text


# Build Gradio interface
with gr.Blocks(title="XIS CV Assessment - Card Measurement") as demo:

    gr.Markdown("""
    # XIS AI Assessment — Card Segmentation & Measurement
    **Mask2Former | Camera Calibration | Pixel-to-MM Pipeline**
    Upload a card image to get real-world dimensions in millimetres.
    """)

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                type="pil",
                label="Input Image (Zoo Peshawar Bus Card)"
            )
            run_btn = gr.Button(
                "Measure Card", variant="primary"
            )

        with gr.Column():
            output_img = gr.Image(
                type="pil",
                label="Segmentation + Measurement Output"
            )
            output_text = gr.Textbox(
                label="Measurement Results",
                lines=15
            )

    gr.Markdown("""
    ### Pipeline Steps
    1. **Undistortion** — removes lens distortion using calibrated camera parameters
    2. **Segmentation** — Mask2Former detects and segments the card
    3. **Measurement** — pixel dimensions converted to mm using pixels_per_mm ratio
    4. **Validation** — result compared against ISO/IEC 7810 ground truth
    """)

    run_btn.click(
        fn=process_image,
        inputs=input_img,
        outputs=[output_img, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)