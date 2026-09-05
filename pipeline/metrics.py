"""
GeoSharp Evaluation Metrics

Compares the LDSR-S2 super-resolved output against
a bicubic 4x upsampling baseline.

Input:
    128 x 128 x 4  (10m RGB-NIR)

Output:
    512 x 512 x 4  (2.5m RGB-NIR)

Important:
    Without a real 2.5m ground-truth image, PSNR/SSIM
    here measure similarity between the AI output and
    the bicubic baseline. They are NOT absolute accuracy
    measurements.
"""

import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)

from skimage.transform import resize


# ============================================================
# BICUBIC BASELINE
# ============================================================

def bicubic_baseline(
    low_res: np.ndarray,
    scale_factor: int = 4
) -> np.ndarray:
    """
    Upsample the low-resolution image using bicubic interpolation.

    Example:
        128 x 128 x 4
            ↓ 4x
        512 x 512 x 4
    """

    if low_res.ndim != 3:
        raise ValueError(
            f"Expected image shape (H, W, C), "
            f"got {low_res.shape}"
        )

    h, w, channels = low_res.shape

    output = resize(
        low_res,
        (
            h * scale_factor,
            w * scale_factor,
            channels
        ),
        order=3,
        anti_aliasing=True,
        preserve_range=True,
    )

    return output.astype(np.float32)


# ============================================================
# PSNR
# ============================================================

def compute_psnr(
    reference: np.ndarray,
    comparison: np.ndarray
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Higher = more similar.
    """

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    comparison = np.asarray(
        comparison,
        dtype=np.float32
    )

    if reference.shape != comparison.shape:
        raise ValueError(
            f"PSNR shape mismatch: "
            f"{reference.shape} vs {comparison.shape}"
        )

    return float(
        psnr(
            reference,
            comparison,
            data_range=1.0
        )
    )


# ============================================================
# SSIM
# ============================================================

def compute_ssim(
    reference: np.ndarray,
    comparison: np.ndarray
) -> float:
    """
    Compute Structural Similarity Index.

    Higher = more similar.
    """

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    comparison = np.asarray(
        comparison,
        dtype=np.float32
    )

    if reference.shape != comparison.shape:
        raise ValueError(
            f"SSIM shape mismatch: "
            f"{reference.shape} vs {comparison.shape}"
        )

    return float(
        ssim(
            reference,
            comparison,
            channel_axis=-1,
            data_range=1.0
        )
    )


# ============================================================
# COMPLETE COMPARISON
# ============================================================

def compare_to_baseline(
    original_low_res: np.ndarray,
    sr_output: np.ndarray,
    scale_factor: int = 4
) -> dict:
    """
    Compare LDSR-S2 output against a bicubic baseline.

    Parameters
    ----------
    original_low_res:
        Original 10m image.
        Expected shape: (128, 128, 4)

    sr_output:
        GeoSharp/LDSR-S2 output.
        Expected shape: (512, 512, 4)

    scale_factor:
        Super-resolution scale.
        GeoSharp uses 4x -> 2.5m.

    Returns
    -------
    dict:
        PSNR and SSIM values.
    """

    original_low_res = np.asarray(
        original_low_res,
        dtype=np.float32
    )

    sr_output = np.asarray(
        sr_output,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if original_low_res.ndim != 3:

        raise ValueError(
            "original_low_res must have shape "
            "(H, W, C). "
            f"Got {original_low_res.shape}"
        )


    if sr_output.ndim != 3:

        raise ValueError(
            "sr_output must have shape "
            "(H, W, C). "
            f"Got {sr_output.shape}"
        )


    # --------------------------------------------------------
    # Validate channels
    # --------------------------------------------------------

    if original_low_res.shape[2] != 4:

        raise ValueError(
            "Expected 4-channel Sentinel-2 input "
            "(B02, B03, B04, B08), "
            f"got {original_low_res.shape[2]} channels."
        )


    if sr_output.shape[2] != 4:

        raise ValueError(
            "Expected 4-channel SR output "
            "(B02, B03, B04, B08), "
            f"got {sr_output.shape[2]} channels."
        )


    # --------------------------------------------------------
    # Create bicubic baseline
    # --------------------------------------------------------

    baseline = bicubic_baseline(
        original_low_res,
        scale_factor=scale_factor
    )


    # --------------------------------------------------------
    # Match exact SR dimensions
    # --------------------------------------------------------

    target_h = sr_output.shape[0]
    target_w = sr_output.shape[1]

    if baseline.shape[:2] != (
        target_h,
        target_w
    ):

        baseline = resize(
            baseline,
            (
                target_h,
                target_w,
                4
            ),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)


    # --------------------------------------------------------
    # Safety clipping
    # --------------------------------------------------------

    baseline = np.clip(
        baseline,
        0.0,
        1.0
    )

    sr_output = np.clip(
        sr_output,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    psnr_value = compute_psnr(
        baseline,
        sr_output
    )

    ssim_value = compute_ssim(
        baseline,
        sr_output
    )


    return {
        "psnr_sr_vs_baseline": psnr_value,
        "ssim_sr_vs_baseline": ssim_value,
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Testing GeoSharp metrics..."
    )

    # Fake 4-channel Sentinel-2 input
    fake_low_res = np.random.rand(
        128,
        128,
        4
    ).astype(np.float32)

    # Fake 4x SR output
    fake_sr = np.random.rand(
        512,
        512,
        4
    ).astype(np.float32)


    results = compare_to_baseline(
        fake_low_res,
        fake_sr,
        scale_factor=4
    )


    print(
        "\nInput:",
        fake_low_res.shape
    )

    print(
        "SR output:",
        fake_sr.shape
    )

    print(
        "\nPSNR:",
        f"{results['psnr_sr_vs_baseline']:.2f} dB"
    )

    print(
        "SSIM:",
        f"{results['ssim_sr_vs_baseline']:.3f}"
    )

    print(
        "\n✓ Metrics test completed."
    )