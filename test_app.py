import json
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv

# Use current working directory
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"


def calculate_psnr(original, carved):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    if original.shape != carved.shape:
        return None
    mse = np.mean((original.astype(np.float64) - carved.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def calculate_ssim(original, carved):
    """Calculate Structural Similarity Index (SSIM)."""
    if original.shape != carved.shape:
        return None
    
    # Convert to grayscale for SSIM
    if len(original.shape) == 3:
        gray1 = cv.cvtColor(original, cv.COLOR_RGB2GRAY).astype(np.float64)
        gray2 = cv.cvtColor(carved, cv.COLOR_RGB2GRAY).astype(np.float64)
    else:
        gray1 = original.astype(np.float64)
        gray2 = carved.astype(np.float64)
    
    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu1 = cv.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(gray2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))

# Paths that seam_carver.py expects
INPUT_IMAGE_PATH = DATA_DIR / "human.png"
PARAMS_PATH = DATA_DIR / "specifications.json"
OUTPUT_IMAGE_PATH = BASE_DIR / "output_carved.png"

st.title("Seam Carving Interface")

# 1. Upload image + get target width/height, then save them

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    w, h = img.size
    st.write(f"**Original size:** {w} × {h}")
    
    target_w = st.number_input("Target width", min_value=1, max_value=w, value=w, step=1)
    target_h = st.number_input("Target height", min_value=1, max_value=h, value=h, step=1)

    if st.button("Save image and parameters"):
        # Ensure data directory exists
        DATA_DIR.mkdir(exist_ok=True)
        
        # Save the input image as data/human.png
        img.save(INPUT_IMAGE_PATH)

        # Save parameters as JSON with keys seam_carver.py expects
        params = {
            "width": int(target_w),
            "height": int(target_h),
        }
        with open(PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        st.success(f"Saved `{INPUT_IMAGE_PATH.name}` and `{PARAMS_PATH.name}` in `{DATA_DIR}`")
        st.code("python seam_carver.py", language="bash")
        st.info("Run the command above to perform seam carving, then click 'Refresh output'")

st.write("---")

# 2. Show output image after seam_carver.py saves it
st.subheader("Output image")

st.write(f"Looking for: `{OUTPUT_IMAGE_PATH.name}` in `{BASE_DIR}`")

if st.button("Refresh output"):
    st.rerun()

if OUTPUT_IMAGE_PATH.exists():
    out_img = Image.open(OUTPUT_IMAGE_PATH).convert("RGB")
    ow, oh = out_img.size
    st.image(out_img, caption=f"Output image ({ow} × {oh})", use_column_width=True)
    
    # Calculate and display quality metrics if original image exists
    if INPUT_IMAGE_PATH.exists():
        original_pil = Image.open(INPUT_IMAGE_PATH).convert("RGB")
        original_np = np.array(original_pil)
        carved_np = np.array(out_img)
        
        # Resize original to match carved dimensions for comparison
        carved_h, carved_w = carved_np.shape[:2]
        resized_original = cv.resize(original_np, (carved_w, carved_h), interpolation=cv.INTER_AREA)
        
        # Calculate metrics
        psnr_value = calculate_psnr(resized_original, carved_np)
        ssim_value = calculate_ssim(resized_original, carved_np)
        
        st.write("---")
        st.subheader("Quality Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PSNR", f"{psnr_value:.2f} dB" if psnr_value else "N/A")
        with col2:
            st.metric("SSIM", f"{ssim_value:.4f}" if ssim_value else "N/A")
        
        st.caption("PSNR: Higher is better (typical: 20-40 dB). SSIM: Closer to 1 is better.")
else:
    st.info(f"No `{OUTPUT_IMAGE_PATH.name}` found yet. Run `python seam_carver.py` and click 'Refresh output'.")
