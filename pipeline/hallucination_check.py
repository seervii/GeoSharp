"""
GeoSharp - Hallucination Diagnostic for Super-Resolution Output

The diagnostic uses two complementary signals:

1. LR-consistency residual
   ------------------------
   The SR image is first blurred with an approximate sensor PSF and then
   downsampled to the original LR resolution.

   If the degraded SR image differs strongly from the actual LR observation,
   the generated fine-scale structure is weakly supported by the input.

   IMPORTANT:
   This is a hallucination-RISK signal, not absolute proof of hallucination.
   Residuals can also arise from registration, radiometric differences,
   preprocessing, or an imperfect degradation model.

2. Sample-variance uncertainty
   ----------------------------
   run_sr() produces an uncertainty map from repeated diffusion samples.
   Regions where the samples disagree more are regions where the model is
   less certain.

3. Combined hallucination-risk mask
   ---------------------------------
   A region is considered high-risk when BOTH:
       - LR-consistency residual is high
       - sample variance uncertainty is high

Usage:
    python pipeline/hallucination_check.py data/raw/tile1_bands/

    python pipeline/hallucination_check.py data/raw/tile1_bands/ \
        --steps 30 \
        --patch-size 128 \
        --blur-sigma 1.0

Outputs:
    outputs/hallucination_check.png
    outputs/hallucination_risk_mask.png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.transform import resize


# -------------------------------------------------------------------------
# Imports from GeoSharp
# -------------------------------------------------------------------------

try:
    from pipeline.inference import run_sr
    from pipeline.preprocess import read_center_patch_from_folder
except ImportError:
    from inference import run_sr
    from preprocess import read_center_patch_from_folder


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 and clip to [0, 1].
    """
    img = np.asarray(img, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


def percentile_normalize(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robustly normalize a diagnostic map using percentiles.

    This prevents a few extreme values from dominating the visualization.
    """
    image = np.asarray(image, dtype=np.float32)

    lo = np.percentile(image, low_percentile)
    hi = np.percentile(image, high_percentile)

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - lo) / (hi - lo)

    return np.clip(normalized, 0.0, 1.0)


def resize_to_shape(
    image: np.ndarray,
    target_shape,
    order: int = 1,
) -> np.ndarray:
    """
    Resize an image/map to target H,W.

    Handles both:
        H x W
        H x W x C
    """
    target_h, target_w = target_shape[:2]

    if image.shape[:2] == (target_h, target_w):
        return image

    if image.ndim == 2:
        return resize(
            image,
            (target_h, target_w),
            order=order,
            anti_aliasing=(order > 0),
            preserve_range=True,
        ).astype(np.float32)

    return resize(
        image,
        (target_h, target_w, image.shape[2]),
        order=order,
        anti_aliasing=(order > 0),
        preserve_range=True,
    ).astype(np.float32)


# -------------------------------------------------------------------------
# LR consistency
# -------------------------------------------------------------------------

def degrade_sr_to_lr(
    sr_output: np.ndarray,
    lr_shape,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """
    Approximate the satellite imaging/degradation process:

        SR
         |
         v
      Gaussian PSF
         |
         v
      Downsampling
         |
         v
        LR

    A real satellite sensor has a more complicated PSF/MTF response.
    Gaussian blur is used here as a practical approximation.

    Parameters
    ----------
    sr_output:
        SR image, H x W x C, expected range [0, 1].

    lr_shape:
        Target LR shape (H, W).

    blur_sigma:
        Gaussian blur sigma in SR pixels.

    Returns
    -------
    degraded_lr:
        Approximate LR reconstruction.
    """

    sr_output = normalize_image(sr_output)

    # -------------------------------------------------------------
    # Step 1: Apply approximate sensor PSF.
    # Blur spatial dimensions independently for every channel.
    # -------------------------------------------------------------

    if blur_sigma > 0:
        blurred = gaussian_filter(
            sr_output,
            sigma=(blur_sigma, blur_sigma, 0),
        )
    else:
        blurred = sr_output.copy()

    # -------------------------------------------------------------
    # Step 2: Downsample to original LR resolution.
    # Anti-aliasing is intentionally enabled.
    # -------------------------------------------------------------

    degraded_lr = resize(
        blurred,
        lr_shape[:2],
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )

    return normalize_image(degraded_lr)


def lr_consistency_error(
    low_res_rgb: np.ndarray,
    sr_output: np.ndarray,
    blur_sigma: float = 1.0,
):
    """
    Calculate LR-consistency residual.

    The SR image is degraded back to LR space and compared with the
    original LR observation.

    Returns
    -------
    error_lr:
        Per-pixel LR residual at LR resolution.

    error_sr:
        Same residual resized to SR resolution for visualization.
    """

    low_res_rgb = normalize_image(low_res_rgb)
    sr_output = normalize_image(sr_output)

    # -------------------------------------------------------------
    # Degrade SR -> approximate LR
    # -------------------------------------------------------------

    reconstructed_lr = degrade_sr_to_lr(
        sr_output,
        low_res_rgb.shape,
        blur_sigma=blur_sigma,
    )

    # -------------------------------------------------------------
    # Channel-wise absolute residual
    # -------------------------------------------------------------

    channel_error = np.abs(reconstructed_lr - low_res_rgb)

    # Mean RGB residual
    error_lr = channel_error.mean(axis=-1)

    # -------------------------------------------------------------
    # Resize residual to SR dimensions.
    #
    # Nearest-neighbour keeps each LR residual associated with the
    # corresponding LR observation instead of inventing smooth
    # intermediate residual values.
    # -------------------------------------------------------------

    error_sr = resize(
        error_lr,
        sr_output.shape[:2],
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )

    return error_lr.astype(np.float32), error_sr.astype(np.float32)


# -------------------------------------------------------------------------
# Uncertainty handling
# -------------------------------------------------------------------------

def prepare_uncertainty_map(
    uncertainty_map: np.ndarray,
    target_shape,
) -> np.ndarray:
    """
    Make uncertainty map compatible with SR resolution.
    """

    uncertainty_map = np.asarray(
        uncertainty_map,
        dtype=np.float32,
    )

    # Remove unnecessary channel dimension if present.
    if uncertainty_map.ndim == 3:
        if uncertainty_map.shape[-1] == 1:
            uncertainty_map = uncertainty_map[..., 0]
        else:
            # If uncertainty somehow has multiple channels,
            # average them.
            uncertainty_map = uncertainty_map.mean(axis=-1)

    # Resize to SR resolution if necessary.
    uncertainty_map = resize_to_shape(
        uncertainty_map,
        target_shape,
        order=1,
    )

    # Remove NaN / Inf values.
    uncertainty_map = np.nan_to_num(
        uncertainty_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Variance cannot physically be negative.
    uncertainty_map = np.maximum(
        uncertainty_map,
        0.0,
    )

    return uncertainty_map.astype(np.float32)


# -------------------------------------------------------------------------
# Risk mask
# -------------------------------------------------------------------------

def create_hallucination_risk_mask(
    consistency_error: np.ndarray,
    uncertainty_map: np.ndarray,
    percentile: float = 90.0,
):
    """
    Create a conservative hallucination-risk mask.

    A pixel is flagged only when BOTH signals are in their top
    percentile:

        high LR residual
              AND
        high uncertainty

    This is intentionally an intersection rather than a union,
    reducing false positives.
    """

    error_threshold = np.percentile(
        consistency_error,
        percentile,
    )

    uncertainty_threshold = np.percentile(
        uncertainty_map,
        percentile,
    )

    high_error = consistency_error >= error_threshold
    high_uncertainty = uncertainty_map >= uncertainty_threshold

    risk_mask = high_error & high_uncertainty

    return (
        risk_mask,
        error_threshold,
        uncertainty_threshold,
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose unsupported/high-risk detail in GeoSharp "
            "super-resolution output."
        )
    )

    parser.add_argument(
        "tile_path",
        help=(
            "Folder containing per-band SAFE-style files "
            "(same input format as inference.py)."
        ),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of diffusion sampling steps.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Low-resolution patch size.",
    )

    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.0,
        help=(
            "Gaussian sensor-PSF approximation in SR pixels. "
            "Use 0 to disable blur."
        ),
    )

    parser.add_argument(
        "--risk-percentile",
        type=float,
        default=90.0,
        help=(
            "Percentile threshold for both residual and uncertainty. "
            "Default: 90."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output directory.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------------------

    if not 50.0 <= args.risk_percentile < 100.0:
        raise ValueError(
            "--risk-percentile should be between 50 and 100."
        )

    if args.blur_sigma < 0:
        raise ValueError(
            "--blur-sigma cannot be negative."
        )

    # ---------------------------------------------------------------------
    # Load LR input
    # ---------------------------------------------------------------------

    print("\nLoading low-resolution satellite bands...")

    low_res_bands = read_center_patch_from_folder(
        args.tile_path,
        patch_size=args.patch_size,
    )

    # Existing GeoSharp band ordering:
    #
    # low_res_bands[0] -> Band 1
    # low_res_bands[1] -> Band 2
    # low_res_bands[2] -> Band 3
    #
    # Existing pipeline converts to RGB using:
    # [Band 3, Band 2, Band 1]

    low_res_rgb = np.stack(
        [
            low_res_bands[2],
            low_res_bands[1],
            low_res_bands[0],
        ],
        axis=-1,
    )

    low_res_rgb = normalize_image(low_res_rgb)

    print(
        f"LR image shape: {low_res_rgb.shape}"
    )

    # ---------------------------------------------------------------------
    # Run SR
    # ---------------------------------------------------------------------

    print(
        f"\nRunning GeoSharp SR "
        f"({args.steps} sampling steps)..."
    )

    sr_output, uncertainty_map = run_sr(
        args.tile_path,
        sampling_steps=args.steps,
        patch_size=args.patch_size,
    )

    sr_output = normalize_image(sr_output)

    print(
        f"SR image shape: {sr_output.shape}"
    )

    # ---------------------------------------------------------------------
    # Prepare uncertainty
    # ---------------------------------------------------------------------

    uncertainty_map = prepare_uncertainty_map(
        uncertainty_map,
        sr_output.shape[:2],
    )

    # ---------------------------------------------------------------------
    # LR consistency
    # ---------------------------------------------------------------------

    print(
        "\nCalculating LR-consistency residual..."
    )

    error_lr, consistency_error = lr_consistency_error(
        low_res_rgb=low_res_rgb,
        sr_output=sr_output,
        blur_sigma=args.blur_sigma,
    )

    # ---------------------------------------------------------------------
    # Print numerical diagnostics
    # ---------------------------------------------------------------------

    print("\n========== DIAGNOSTICS ==========")

    print(
        "LR-consistency residual:"
    )

    print(
        f"  mean : {consistency_error.mean():.6f}"
    )

    print(
        f"  max  : {consistency_error.max():.6f}"
    )

    print(
        f"  p95  : {np.percentile(consistency_error, 95):.6f}"
    )

    print(
        "\nSample-variance uncertainty:"
    )

    print(
        f"  mean : {uncertainty_map.mean():.6f}"
    )

    print(
        f"  max  : {uncertainty_map.max():.6f}"
    )

    print(
        f"  p95  : {np.percentile(uncertainty_map, 95):.6f}"
    )

    # ---------------------------------------------------------------------
    # Create hallucination-risk mask
    # ---------------------------------------------------------------------

    risk_mask, error_threshold, uncertainty_threshold = (
        create_hallucination_risk_mask(
            consistency_error,
            uncertainty_map,
            percentile=args.risk_percentile,
        )
    )

    flagged_percentage = (
        100.0
        * risk_mask.sum()
        / risk_mask.size
    )

    print(
        "\nHallucination-risk thresholds:"
    )

    print(
        f"  residual threshold   : {error_threshold:.6f}"
    )

    print(
        f"  uncertainty threshold: {uncertainty_threshold:.6f}"
    )

    print(
        f"\nHigh hallucination-risk pixels: "
        f"{flagged_percentage:.2f}%"
    )

    print(
        "=================================\n"
    )

    # ---------------------------------------------------------------------
    # Normalize maps for visualization
    # ---------------------------------------------------------------------

    consistency_visual = percentile_normalize(
        consistency_error,
        low_percentile=1,
        high_percentile=99,
    )

    uncertainty_visual = percentile_normalize(
        uncertainty_map,
        low_percentile=1,
        high_percentile=99,
    )

    # ---------------------------------------------------------------------
    # Create overlay
    # ---------------------------------------------------------------------

    risk_overlay = sr_output.copy()

    # Red highlight for high-risk regions.
    #
    # We intentionally don't completely replace the SR image so that
    # the underlying generated structures remain visible.
    risk_overlay[risk_mask] = (
        0.5 * risk_overlay[risk_mask]
        + 0.5 * np.array([1.0, 0.0, 0.0])
    )

    # ---------------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------------

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Save binary mask
    # ---------------------------------------------------------------------

    mask_path = os.path.join(
        args.out_dir,
        "hallucination_risk_mask.png",
    )

    plt.imsave(
        mask_path,
        risk_mask.astype(np.uint8),
        cmap="gray",
    )

    # ---------------------------------------------------------------------
    # Create diagnostic figure
    # ---------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 5),
    )

    # -------------------------------------------------------------
    # Panel 1: LR input
    # -------------------------------------------------------------

    axes[0].imshow(low_res_rgb)

    axes[0].set_title(
        "Input (low-res)"
    )

    # -------------------------------------------------------------
    # Panel 2: SR output
    # -------------------------------------------------------------

    axes[1].imshow(sr_output)

    axes[1].set_title(
        "SR output"
    )

    # -------------------------------------------------------------
    # Panel 3: LR consistency
    # -------------------------------------------------------------

    axes[2].imshow(
        consistency_visual,
        cmap="inferno",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "LR-consistency residual\n"
        "(bright = weakly supported detail)"
    )

    # -------------------------------------------------------------
    # Panel 4: uncertainty
    # -------------------------------------------------------------

    axes[3].imshow(
        uncertainty_visual,
        cmap="inferno",
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Sample-variance uncertainty\n"
        "(bright = model disagreement)"
    )

    # -------------------------------------------------------------
    # Panel 5: risk overlay
    # -------------------------------------------------------------

    axes[4].imshow(
        risk_overlay
    )

    axes[4].set_title(
        "Hallucination-risk overlay\n"
        "(high residual + high uncertainty)"
    )

    # -------------------------------------------------------------
    # Remove axes
    # -------------------------------------------------------------

    for ax in axes:
        ax.axis("off")

    # -------------------------------------------------------------
    # Overall title
    # -------------------------------------------------------------

    fig.suptitle(
        "GeoSharp — Hallucination Diagnostic",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()

    # ---------------------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------------------

    out_path = os.path.join(
        args.out_dir,
        "hallucination_check.png",
    )

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved diagnostic figure to:\n"
        f"  {out_path}"
    )

    print(
        f"Saved risk mask to:\n"
        f"  {mask_path}"
    )


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()