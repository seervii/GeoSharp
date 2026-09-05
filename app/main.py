"""
GeoSharp — Sentinel-2 Super-Resolution Demo

Pipeline:

    Sentinel-2 10m
        128 x 128 x 4
        B02 B03 B04 B08
              ↓
          LDSR-S2
            4x
              ↓
        512 x 512 x 4
              ↓
          2.5m output

Frontend shows:

    1. Original RGB
    2. GeoSharp SR RGB
    3. Sample-variance uncertainty
    4. LR-consistency residual
    5. Hallucination-risk overlay
    6. 4-band PSNR
    7. 4-band SSIM
    8. Uncertainty statistics
    9. LR-consistency statistics
    10. Hallucination-risk percentage

Bands:

    B02 = Blue
    B03 = Green
    B04 = Red
    B08 = NIR
"""

import sys
import os

import numpy as np
import streamlit as st
import rasterio


# =============================================================
# PATH SETUP
# =============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# =============================================================
# GEOSHARP IMPORTS
# =============================================================

from pipeline.inference import run_sr

from pipeline.metrics import (
    compare_to_baseline
)

from pipeline.hallucination_check import (
    lr_consistency_error,
    create_hallucination_risk_mask,
    percentile_normalize,
)


# =============================================================
# STREAMLIT CONFIG
# =============================================================

st.set_page_config(
    page_title="GeoSharp",
    page_icon="🛰️",
    layout="wide"
)


# =============================================================
# HEADER
# =============================================================

st.title(
    "GeoSharp — Sentinel-2 Super-Resolution"
)

st.caption(
    "AI-based super-resolution of Sentinel-2 imagery "
    "from 10m to 2.5m using LDSR-S2 with RGB-NIR "
    "reconstruction, uncertainty estimation and "
    "hallucination-risk diagnostics."
)


# =============================================================
# IMAGE UTILITIES
# =============================================================

def normalize_reflectance(data):
    """
    Convert Sentinel-2 data to float32 reflectance.

    Supports:

        Already normalized:
            0 - 1

        DN-scaled:
            approximately 0 - 10000
    """

    data = np.asarray(
        data,
        dtype=np.float32
    )

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    if np.nanmax(data) > 2.0:

        data = data / 10000.0

    return np.clip(
        data,
        0.0,
        1.0
    )


def stretch_rgb(rgb):
    """
    Display-only RGB enhancement.

    Does NOT modify the actual values used
    for model inference or metrics.
    """

    rgb = np.asarray(
        rgb,
        dtype=np.float32
    )

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Same scale applied to all RGB channels.
    # This preserves relative color balance.

    low = 0.02
    high = 0.35

    rgb = (
        rgb - low
    ) / (
        high - low
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    # Gamma correction
    rgb = np.power(
        rgb,
        1.0 / 2.2
    )

    return np.clip(
        rgb,
        0.0,
        1.0
    )


def bands_to_rgb(data):
    """
    Convert Sentinel-2 B02/B03/B04/B08
    into natural RGB.

    Input:

        (4, H, W)

    Output:

        (H, W, 3)

    RGB:

        R = B04
        G = B03
        B = B02
    """

    rgb = np.stack(
        [
            data[2],  # B04 -> Red
            data[1],  # B03 -> Green
            data[0],  # B02 -> Blue
        ],
        axis=-1
    )

    return rgb.astype(
        np.float32
    )


def uncertainty_display_map(
    uncertainty,
    low_percentile=2.0,
    high_percentile=98.0
):
    """
    Normalize uncertainty only for display.
    """

    uncertainty = np.asarray(
        uncertainty,
        dtype=np.float32
    )

    uncertainty = np.nan_to_num(
        uncertainty,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    low = np.percentile(
        uncertainty,
        low_percentile
    )

    high = np.percentile(
        uncertainty,
        high_percentile
    )

    if high <= low:

        return np.zeros_like(
            uncertainty,
            dtype=np.float32
        )

    normalized = (
        uncertainty - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


def make_risk_overlay(
    sr_rgb,
    risk_mask
):
    """
    Overlay hallucination-risk regions on RGB SR image.

    Risk areas are highlighted in red.

    IMPORTANT:
    The risk mask is calculated from the
    4-band RGB-NIR pipeline, but visualization
    is RGB only.
    """

    overlay = np.asarray(
        sr_rgb,
        dtype=np.float32
    ).copy()

    risk_mask = np.asarray(
        risk_mask,
        dtype=bool
    )

    # Red highlight
    overlay[risk_mask] = (
        0.5 * overlay[risk_mask]
        + 0.5 * np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )
    )

    return np.clip(
        overlay,
        0.0,
        1.0
    )


# =============================================================
# TIFF READER
# =============================================================

def read_tiff_4band(path):
    """
    Read:

        B02
        B03
        B04
        B08

    Returns:

        (4, H, W)
    """

    with rasterio.open(path) as src:

        if src.count < 4:

            raise ValueError(
                f"Expected at least 4 bands "
                f"(B02, B03, B04, B08), "
                f"but TIFF contains {src.count}."
            )

        data = src.read(
            indexes=[1, 2, 3, 4]
        )

    return normalize_reflectance(
        data
    )


# =============================================================
# FILE UPLOADER
# =============================================================

uploaded_file = st.file_uploader(
    "Upload a Sentinel-2 4-band GeoTIFF",
    type=["tif", "tiff"]
)


# =============================================================
# MAIN
# =============================================================

if uploaded_file is not None:

    # =========================================================
    # SAVE INPUT
    # =========================================================

    temp_path = os.path.join(
        "data",
        "raw",
        uploaded_file.name
    )

    os.makedirs(
        os.path.dirname(temp_path),
        exist_ok=True
    )

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # =========================================================
    # READ ORIGINAL
    # =========================================================

    try:

        data = read_tiff_4band(
            temp_path
        )

    except Exception as e:

        st.error(
            f"Could not read Sentinel-2 TIFF: {e}"
        )

        st.stop()

    # =========================================================
    # VALIDATE 4 BANDS
    # =========================================================

    if data.shape[0] != 4:

        st.error(
            f"Expected 4 bands, "
            f"got {data.shape[0]}"
        )

        st.stop()

    # =========================================================
    # ORIGINAL 4-CHANNEL ARRAY
    # =========================================================

    original_4ch_full = (
        np.transpose(
            data,
            (1, 2, 0)
        )
        .astype(np.float32)
    )

    # =========================================================
    # MODEL INPUT PATCH
    # =========================================================

    h, w = original_4ch_full.shape[:2]

    if h < 128 or w < 128:

        st.error(
            f"Input image is {w}x{h}. "
            "At least 128x128 pixels are required."
        )

        st.stop()

    row_start = (
        h - 128
    ) // 2

    col_start = (
        w - 128
    ) // 2

    original_4ch = (
        original_4ch_full[
            row_start:row_start + 128,
            col_start:col_start + 128,
            :
        ]
    )

    # Expected:
    #
    # (128, 128, 4)

    if original_4ch.shape != (
        128,
        128,
        4
    ):

        raise ValueError(
            f"Expected model input "
            f"(128,128,4), "
            f"got {original_4ch.shape}"
        )

    # =========================================================
    # ORIGINAL RGB
    # =========================================================

    original_rgb = bands_to_rgb(
        np.transpose(
            original_4ch,
            (2, 0, 1)
        )
    )

    original_rgb_display = stretch_rgb(
        original_rgb
    )

    # =========================================================
    # RUN LDSR-S2
    # =========================================================

    with st.spinner(
        "Running LDSR-S2 — generating 2.5m imagery..."
    ):

        sr_output, uncertainty_map = run_sr(
            temp_path,
            sampling_steps=100,
            patch_size=128
        )

    # =========================================================
    # VALIDATE SR OUTPUT
    # =========================================================

    if sr_output.shape != (
        512,
        512,
        4
    ):

        raise ValueError(
            f"Expected 512x512x4 output, "
            f"got {sr_output.shape}"
        )

    if uncertainty_map.shape != (
        512,
        512
    ):

        raise ValueError(
            f"Expected 512x512 uncertainty map, "
            f"got {uncertainty_map.shape}"
        )

    # =========================================================
    # NORMALIZE SR OUTPUT
    # =========================================================

    sr_output = np.asarray(
        sr_output,
        dtype=np.float32
    )

    sr_output = np.nan_to_num(
        sr_output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    sr_output = np.clip(
        sr_output,
        0.0,
        1.0
    )

    # =========================================================
    # SR RGB
    # =========================================================

    sr_rgb = np.stack(
        [
            sr_output[:, :, 2],  # B04 -> R
            sr_output[:, :, 1],  # B03 -> G
            sr_output[:, :, 0],  # B02 -> B
        ],
        axis=-1
    )

    sr_rgb_display = stretch_rgb(
        sr_rgb
    )

    # =========================================================
    # UNCERTAINTY VISUALIZATION
    # =========================================================

    uncertainty_visual = (
        uncertainty_display_map(
            uncertainty_map
        )
    )

    # =========================================================
    # HALLUCINATION / LR CONSISTENCY
    # =========================================================

    with st.spinner(
        "Running hallucination-risk diagnostics..."
    ):

        # -----------------------------------------------------
        # Degrade SR back to LR and calculate residual
        # -----------------------------------------------------

        error_lr, consistency_error = (
            lr_consistency_error(
                low_res_rgb=original_4ch,
                sr_output=sr_output,
                blur_sigma=1.0
            )
        )

        # -----------------------------------------------------
        # Normalize residual for display
        # -----------------------------------------------------

        consistency_visual = (
            percentile_normalize(
                consistency_error,
                low_percentile=1,
                high_percentile=99
            )
        )

        # -----------------------------------------------------
        # Create conservative risk mask
        #
        # BOTH must be high:
        #
        # high residual
        #       AND
        # high uncertainty
        # -----------------------------------------------------

        (
            risk_mask,
            error_threshold,
            uncertainty_threshold
        ) = create_hallucination_risk_mask(
            consistency_error,
            uncertainty_map,
            percentile=90.0
        )

        # -----------------------------------------------------
        # Percentage of high-risk pixels
        # -----------------------------------------------------

        flagged_percentage = (
            100.0
            * risk_mask.sum()
            / risk_mask.size
        )

        # -----------------------------------------------------
        # RGB risk overlay
        # -----------------------------------------------------

        risk_overlay = make_risk_overlay(
            sr_rgb_display,
            risk_mask
        )

    # =========================================================
    # SUCCESS
    # =========================================================

    st.success(
        "Super-resolution completed! "
        "10m → 2.5m"
    )

    # =========================================================
    # MAIN VISUALIZATION
    # =========================================================

    st.header(
        "Super-Resolution Result"
    )

    col1, col2, col3 = st.columns(
        3
    )

    # ---------------------------------------------------------
    # ORIGINAL
    # ---------------------------------------------------------

    with col1:

        st.subheader(
            "Original (10m)"
        )

        st.image(
            original_rgb_display,
            use_container_width=True
        )

        st.caption(
            "B04 / B03 / B02 natural-color RGB"
        )

    # ---------------------------------------------------------
    # SR
    # ---------------------------------------------------------

    with col2:

        st.subheader(
            "GeoSharp Output (2.5m)"
        )

        st.image(
            sr_rgb_display,
            use_container_width=True
        )

        st.caption(
            "LDSR-S2 4× reconstruction"
        )

    # ---------------------------------------------------------
    # UNCERTAINTY
    # ---------------------------------------------------------

    with col3:

        st.subheader(
            "Uncertainty Map"
        )

        st.image(
            uncertainty_visual,
            use_container_width=True
        )

        st.caption(
            "Brighter = higher model uncertainty"
        )

    # =========================================================
    # OUTPUT INFORMATION
    # =========================================================

    st.divider()

    st.header(
        "Output Information"
    )

    info1, info2, info3, info4 = st.columns(
        4
    )

    info1.metric(
        "Input",
        "128 × 128 × 4"
    )

    info2.metric(
        "Output",
        "512 × 512 × 4"
    )

    info3.metric(
        "Resolution",
        "10m → 2.5m"
    )

    info4.metric(
        "Scale",
        "4×"
    )

    st.caption(
        "Bands: B02 (Blue) • B03 (Green) • "
        "B04 (Red) • B08 (NIR)"
    )

    # =========================================================
    # 4-BAND QUALITY METRICS
    # =========================================================

    st.divider()

    st.header(
        "4-Band Reconstruction Quality"
    )

    # ---------------------------------------------------------
    # Bicubic baseline
    #
    # Original 128x128x4
    #       ↓
    # Bicubic 4x
    #       ↓
    # 512x512x4
    #
    # Compared against:
    #
    # GeoSharp 512x512x4
    # ---------------------------------------------------------

    metrics = compare_to_baseline(
        original_4ch.astype(
            np.float32
        ),
        sr_output.astype(
            np.float32
        ),
        scale_factor=4
    )

    metric1, metric2 = st.columns(
        2
    )

    metric1.metric(
        "PSNR — All 4 Bands",
        f"{metrics['psnr_sr_vs_baseline']:.2f} dB"
    )

    metric2.metric(
        "SSIM — All 4 Bands",
        f"{metrics['ssim_sr_vs_baseline']:.3f}"
    )

    st.caption(
        "Calculated across B02, B03, B04 and B08 "
        "against a 4× bicubic baseline."
    )

    # =========================================================
    # PER-BAND BASIC STATISTICS
    # =========================================================

    st.subheader(
        "Spectral Channels"
    )

    band1, band2, band3, band4 = st.columns(
        4
    )

    band1.metric(
        "B02",
        "Blue"
    )

    band2.metric(
        "B03",
        "Green"
    )

    band3.metric(
        "B04",
        "Red"
    )

    band4.metric(
        "B08",
        "NIR"
    )

    # =========================================================
    # UNCERTAINTY ANALYSIS
    # =========================================================

    st.divider()

    st.header(
        "Uncertainty Analysis"
    )

    uncertainty_mean = float(
        np.mean(
            uncertainty_map
        )
    )

    uncertainty_max = float(
        np.max(
            uncertainty_map
        )
    )

    uncertainty_p95 = float(
        np.percentile(
            uncertainty_map,
            95
        )
    )

    u1, u2, u3 = st.columns(
        3
    )

    u1.metric(
        "Mean Uncertainty",
        f"{uncertainty_mean:.6f}"
    )

    u2.metric(
        "Maximum Uncertainty",
        f"{uncertainty_max:.6f}"
    )

    u3.metric(
        "P95 Uncertainty",
        f"{uncertainty_p95:.6f}"
    )

    st.caption(
        "Uncertainty is estimated from stochastic "
        "diffusion samples. Higher values indicate "
        "greater disagreement between reconstructions."
    )

    # =========================================================
    # LR CONSISTENCY
    # =========================================================

    st.divider()

    st.header(
        "LR-Consistency Analysis"
    )

    st.caption(
        "The SR output is degraded back toward the "
        "original 10m observation. A larger residual "
        "means the generated fine-scale structure is "
        "less strongly supported by the observed input."
    )

    residual_mean = float(
        np.mean(
            consistency_error
        )
    )

    residual_max = float(
        np.max(
            consistency_error
        )
    )

    residual_p95 = float(
        np.percentile(
            consistency_error,
            95
        )
    )

    r1, r2, r3 = st.columns(
        3
    )

    r1.metric(
        "Mean Residual",
        f"{residual_mean:.6f}"
    )

    r2.metric(
        "Maximum Residual",
        f"{residual_max:.6f}"
    )

    r3.metric(
        "P95 Residual",
        f"{residual_p95:.6f}"
    )

    # =========================================================
    # LR CONSISTENCY MAP
    # =========================================================

    st.subheader(
        "LR-Consistency Residual Map"
    )

    st.image(
        consistency_visual,
        use_container_width=True
    )

    st.caption(
        "Brighter = greater difference between the "
        "observed low-resolution signal and the "
        "SR output after degradation."
    )

    # =========================================================
    # HALLUCINATION-RISK ANALYSIS
    # =========================================================

    st.divider()

    st.header(
        "Hallucination-Risk Analysis"
    )

    st.info(
        "Hallucination risk is flagged only where "
        "both LR-consistency residual and stochastic "
        "uncertainty are high. This is a risk indicator, "
        "not absolute proof of hallucination."
    )

    # ---------------------------------------------------------
    # Risk percentage
    # ---------------------------------------------------------

    if flagged_percentage < 5.0:

        risk_status = "Low"

    elif flagged_percentage < 15.0:

        risk_status = "Moderate"

    else:

        risk_status = "High"

    h1, h2, h3 = st.columns(
        3
    )

    h1.metric(
        "High-Risk Pixels",
        f"{flagged_percentage:.2f}%"
    )

    h2.metric(
        "Risk Status",
        risk_status
    )

    h3.metric(
        "Risk Threshold",
        "90th percentile"
    )

    # =========================================================
    # RISK THRESHOLDS
    # =========================================================

    st.subheader(
        "Risk Thresholds"
    )

    t1, t2 = st.columns(
        2
    )

    t1.metric(
        "LR Residual Threshold",
        f"{error_threshold:.6f}"
    )

    t2.metric(
        "Uncertainty Threshold",
        f"{uncertainty_threshold:.6f}"
    )

    # =========================================================
    # HALLUCINATION OVERLAY
    # =========================================================

    st.subheader(
        "Hallucination-Risk Overlay"
    )

    st.image(
        risk_overlay,
        use_container_width=True
    )

    st.caption(
        "Red regions = high LR-consistency residual "
        "AND high stochastic uncertainty."
    )

    # =========================================================
    # RISK MASK
    # =========================================================

    st.subheader(
        "Binary Risk Mask"
    )

    st.image(
        risk_mask.astype(
            np.float32
        ),
        use_container_width=True
    )

    st.caption(
        "White = high hallucination-risk region. "
        "Black = lower-risk region."
    )

    # =========================================================
    # DIAGNOSTIC SUMMARY
    # =========================================================

    st.divider()

    st.header(
        "GeoSharp Diagnostic Summary"
    )

    summary_col1, summary_col2 = st.columns(
        2
    )

    with summary_col1:

        st.markdown(
            f"""
**Reconstruction**

- Input: **128 × 128 × 4**
- Output: **512 × 512 × 4**
- Resolution: **10m → 2.5m**
- Super-resolution factor: **4×**
- Channels: **B02 / B03 / B04 / B08**
- Sampling steps: **100**

**Quality**

- 4-band PSNR: **{metrics['psnr_sr_vs_baseline']:.2f} dB**
- 4-band SSIM: **{metrics['ssim_sr_vs_baseline']:.3f}**
"""
        )

    with summary_col2:

        st.markdown(
            f"""
**Uncertainty**

- Mean: **{uncertainty_mean:.6f}**
- P95: **{uncertainty_p95:.6f}**
- Maximum: **{uncertainty_max:.6f}**

**Hallucination Risk**

- Mean LR residual: **{residual_mean:.6f}**
- P95 LR residual: **{residual_p95:.6f}**
- High-risk pixels: **{flagged_percentage:.2f}%**
- Risk status: **{risk_status}**
"""
        )

    # =========================================================
    # SCIENTIFIC CAVEAT
    # =========================================================

    st.warning(
        "Evaluation note: PSNR/SSIM compare the "
        "4-channel GeoSharp output against a 4× bicubic "
        "baseline because native 2.5m ground truth is "
        "not available. Therefore these metrics indicate "
        "relative reconstruction behavior, not absolute "
        "2.5m accuracy. Hallucination diagnostics are "
        "risk indicators rather than proof of hallucination."
    )

else:

    st.info(
        "Upload a 4-band Sentinel-2 GeoTIFF "
        "to run GeoSharp."
    )