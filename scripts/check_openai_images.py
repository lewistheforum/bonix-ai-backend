"""
Check the exact images being sent to OpenAI in the 3-step pipeline.

Generates an HTML file showing:
  1. The standardized image sent to OpenAI for classification (Step 1)
  2. The annotated YOLO detection image sent to OpenAI for analysis (Step 3)
"""
import sys
import os
import io
import base64
import cv2
import numpy as np
from PIL import Image

# Ensure app is importable
sys.path.insert(0, os.path.dirname(__file__))

from app.services.fracture_detection.yolo_patch import patch_ultralytics
patch_ultralytics()
from ultralytics import YOLO


def standardize_image(image_bytes: bytes) -> Image.Image:
    """Same logic as fracture_detection_service._standardize_image"""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "I;16" or img.mode == "I":
        img = img.point(lambda i: i * (1. / 256)).convert("L")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_jpeg_base64(pil_img: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def main(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        return

    print(f"Reading {image_path}...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    orig_img = Image.open(io.BytesIO(image_bytes))
    print(f"Original: format={orig_img.format}, mode={orig_img.mode}, size={orig_img.size}")

    # ── Step 1 image: standardized JPEG sent for classification ──
    step1_img = standardize_image(image_bytes)
    step1_b64 = to_jpeg_base64(step1_img)
    print(f"Step 1 (classify) image: base64 length = {len(step1_b64)} chars")

    # ── Step 2: run YOLO detection ──
    model_path = os.path.join(os.path.dirname(__file__), "app", "models", "huggingface", "wrist_fracture_model.pt")
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    results = model(step1_img)
    result = results[0]

    # Collect detections
    detections_html = ""
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = result.names[cls_id]
            detections_html += f"<li><strong>{name}</strong>: {conf*100:.2f}% confidence</li>\n"
    else:
        detections_html = "<li>No detections</li>"

    # ── Step 3 image: annotated image sent for analysis ──
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)
    step3_b64 = to_jpeg_base64(annotated_pil)
    print(f"Step 3 (analyze) image: base64 length = {len(step3_b64)} chars")

    # ── Generate HTML preview ──
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>OpenAI Image Upload Preview</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 30px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #e94560; text-align: center; }}
        .step {{ background: #16213e; border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
        .step h2 {{ color: #0f3460; background: #e94560; display: inline-block; padding: 6px 16px; border-radius: 6px; color: #fff; }}
        .step img {{ max-width: 100%; height: auto; border: 2px solid #0f3460; border-radius: 8px; margin-top: 15px; }}
        .info {{ background: #0f3460; padding: 12px 18px; border-radius: 8px; margin: 10px 0; font-size: 14px; }}
        ul {{ line-height: 1.8; }}
        .badge {{ display: inline-block; background: #e94560; padding: 3px 10px; border-radius: 4px; font-size: 12px; margin-right: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 OpenAI Image Upload Preview</h1>
        <p style="text-align:center; opacity:0.7;">Original file: <strong>{image_path}</strong></p>

        <div class="step">
            <h2>Step 1 — Classification Image</h2>
            <div class="info">
                Sent to OpenAI Vision to check: <em>"Is it a wrist X-ray?"</em><br/>
                <span class="badge">JPEG</span> <span class="badge">Base64: {len(step1_b64)} chars</span>
            </div>
            <img src="data:image/jpeg;base64,{step1_b64}" alt="Step 1 Classification" />
        </div>

        <div class="step">
            <h2>Step 2 — YOLO Detections</h2>
            <div class="info">Detected objects:</div>
            <ul>
                {detections_html}
            </ul>
        </div>

        <div class="step">
            <h2>Step 3 — Annotated Analysis Image</h2>
            <div class="info">
                Sent to OpenAI for medical analysis with bounding boxes drawn.<br/>
                <span class="badge">JPEG</span> <span class="badge">Base64: {len(step3_b64)} chars</span>
            </div>
            <img src="data:image/jpeg;base64,{step3_b64}" alt="Step 3 Analysis" />
        </div>
    </div>
</body>
</html>"""

    output_file = "preview_openai_images.html"
    with open(output_file, "w") as f:
        f.write(html)

    print(f"\n✅ Created '{output_file}'")
    print(f"   Open: file://{os.path.abspath(output_file)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_openai_images.py <path_to_image>")
        sys.exit(1)
    main(sys.argv[1])
