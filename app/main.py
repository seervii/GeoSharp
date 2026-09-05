"""
GeoSharp demo UI (Streamlit).

Upload a Sentinel-2 tile -> shows original vs sharpened vs uncertainty map
side-by-side, plus PSNR/SSIM vs a bicubic baseline.

Run with:
    streamlit run app/main.py
"""

import sys
import os

import numpy as np
import streamlit as st
from PIL import Image

# Allow importing from ../pipeline when run as `streamlit run app/main.py`
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pipeline.inference import run_sr
from pipeline.metrics import compare_to_baseline, bicubic_baseline


st.set_page_config(page_title="GeoSharp", layout="wide")

st.title("GeoSharp — Sentinel-2 Super-Resolution")
st.caption(
    "Sharpens Sentinel-2 imagery (10m) using a pretrained model (LDSR-S2), "
    "with a per-pixel uncertainty map showing reconstructed vs. observed detail."
)

uploaded_file = st.file_uploader(
    "Upload a Sentinel-2 tile (GeoTIFF)", type=["tif", "tiff"]
)

if uploaded_file is not None:
    # Save uploaded file temporarily so pipeline functions can read it by path
    temp_path = os.path.join("data", "raw", uploaded_file.name)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running super-resolution..."):
        sr_output, uncertainty_map = run_sr(temp_path)

    st.success("Done!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original (10m)")
        # Placeholder original preview -- replace with actual low-res
        # preview once preprocess.py is wired in for real tiles.
        st.image(
            np.random.rand(128, 128, 3),
            clamp=True,
            use_container_width=True,
        )

    with col2:
        st.subheader("Sharpened output")
        st.image(sr_output, clamp=True, use_container_width=True)

    with col3:
        st.subheader("Uncertainty map")
        st.image(uncertainty_map, clamp=True, use_container_width=True)
        st.caption("Brighter = lower confidence (more reconstructed detail)")

    st.divider()
    st.subheader("Quality vs. bicubic baseline")

    fake_low_res = np.random.rand(128, 128, 3).astype(np.float32)
    metrics = compare_to_baseline(fake_low_res, sr_output)

    m1, m2 = st.columns(2)
    m1.metric("PSNR (vs bicubic)", f"{metrics['psnr_sr_vs_baseline']:.2f} dB")
    m2.metric("SSIM (vs bicubic)", f"{metrics['ssim_sr_vs_baseline']:.3f}")

else:
    st.info("Upload a Sentinel-2 tile to see the sharpened result.")
