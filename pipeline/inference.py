"""
GeoSharp inference module.

Pipeline:
    Sentinel-2 10m input
        128 x 128 x 4
              ↓
        LDSR-S2 (4x)
              ↓
        512 x 512 x 4
              ↓
        GeoSharp 2.5m output

Channels:
    B02 = Blue
    B03 = Green
    B04 = Red
    B08 = NIR

Final output:
    (512, 512, 4)

Uncertainty:
    (512, 512)
"""

import os
import numpy as np

USE_PLACEHOLDER = False

_MODEL = None
_DEVICE = None


def _placeholder_run_sr(tile_path: str):
    """Fake output for UI testing."""

    fake_output = np.random.rand(
        512, 512, 4
    ).astype(np.float32)

    fake_uncertainty = np.random.rand(
        512, 512
    ).astype(np.float32)

    return fake_output, fake_uncertainty


def _get_model():
    """
    Load LDSR-S2 model and pretrained weights once.

    Input:
        (B, 4, H, W)

    LDSR-S2 internally performs:
        128x128x4 -> 512x512x4
    """

    global _MODEL, _DEVICE

    # Reuse already loaded model
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import requests
    from io import StringIO
    from omegaconf import OmegaConf
    import opensr_model
    import torch

    # Select GPU if available
    _DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {_DEVICE}")

    # Official ESA OpenSR configuration
    config_url = (
        "https://raw.githubusercontent.com/"
        "ESAOpenSR/opensr-model/"
        "refs/heads/main/"
        "opensr_model/configs/config_10m.yaml"
    )

    response = requests.get(
        config_url,
        timeout=30
    )

    response.raise_for_status()

    config = OmegaConf.load(
        StringIO(response.text)
    )

    # Create model
    model = opensr_model.SRLatentDiffusion(
        config,
        device=_DEVICE
    )

    # Load pretrained LDSR-S2 weights
    model.load_pretrained(
        config.ckpt_version
    )

    _MODEL = model

    print("LDSR-S2 model loaded successfully.")

    return _MODEL, _DEVICE


def _real_run_sr(
    tile_path: str,
    sampling_steps: int = 100,
    patch_size: int = 128
):
    """
    Real LDSR-S2 inference.

    Input:
        128 x 128 x 4

    LDSR-S2:
        4x super-resolution

    Final:
        512 x 512 x 4

    Resolution:
        10m -> 2.5m
    """

    import torch
    import rasterio

    try:
        from pipeline.preprocess import (
            read_center_patch_from_folder
        )
    except ImportError:
        from preprocess import (
            read_center_patch_from_folder
        )

    # =========================================================
    # LOAD MODEL
    # =========================================================

    model, device = _get_model()

    # =========================================================
    # 1. READ 128x128 PATCH
    # =========================================================

    if os.path.isdir(tile_path):

        tile = read_center_patch_from_folder(
            tile_path,
            patch_size=patch_size
        )

    else:

        with rasterio.open(tile_path) as src:

            h = src.height
            w = src.width

            if h < patch_size or w < patch_size:
                raise ValueError(
                    f"Tile is {w}x{h}, smaller than "
                    f"requested patch_size={patch_size}."
                )

            from rasterio.windows import Window

            # Center crop
            row_off = (
                h - patch_size
            ) // 2

            col_off = (
                w - patch_size
            ) // 2

            window = Window(
                col_off,
                row_off,
                patch_size,
                patch_size
            )

            tile = src.read(
                window=window
            )

    print(
        f"Patch shape: {tile.shape}"
    )

    # =========================================================
    # 2. VERIFY 4 BANDS
    # =========================================================

    if tile.shape[0] < 4:

        raise ValueError(
            f"Expected 4 bands "
            f"(B02, B03, B04, B08), "
            f"but got {tile.shape[0]}."
        )

    # EXACTLY FOUR CHANNELS:
    #
    # 0 = B02 Blue
    # 1 = B03 Green
    # 2 = B04 Red
    # 3 = B08 NIR

    tile = tile[:4]

    # =========================================================
    # 3. NORMALIZE REFLECTANCE
    # =========================================================

    tile = np.asarray(
        tile,
        dtype=np.float32
    )

    # If Sentinel-2 data is stored as
    # integer DN values such as 0-10000,
    # convert to reflectance.
    #
    # If already normalized 0-1,
    # leave unchanged.

    max_value = np.nanmax(tile)

    if max_value > 2.0:

        print(
            "Input appears to be DN-scaled. "
            "Converting to reflectance."
        )

        tile = tile / 10000.0

    tile = np.nan_to_num(
        tile,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    tile = np.clip(
        tile,
        0.0,
        1.0
    )

    # =========================================================
    # 4. CREATE MODEL TENSOR
    # =========================================================

    tensor = (
        torch.from_numpy(tile)
        .float()
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Input tensor shape: "
        f"{tuple(tensor.shape)}"
    )

    # Expected:
    #
    # (1, 4, 128, 128)

    expected_input = (
        1,
        4,
        patch_size,
        patch_size
    )

    if tuple(tensor.shape) != expected_input:

        raise RuntimeError(
            f"Expected input tensor "
            f"{expected_input}, "
            f"got {tuple(tensor.shape)}"
        )

    # =========================================================
    # 5. LDSR-S2 INFERENCE
    # =========================================================

    print(
        f"Sampling steps: {sampling_steps}"
    )

    with torch.no_grad():

        sr = model.forward(
            tensor,
            sampling_steps=sampling_steps
        )

    print(
        f"LDSR raw output: "
        f"{tuple(sr.shape)}"
    )

    # Expected:
    #
    # (1, 4, 512, 512)

    if sr.ndim != 4:

        raise RuntimeError(
            f"Unexpected model output "
            f"dimensions: {sr.shape}"
        )

    if sr.shape[1] < 4:

        raise RuntimeError(
            f"Expected 4-channel "
            f"LDSR-S2 output, "
            f"but received {sr.shape}."
        )

    # =========================================================
    # 6. KEEP ALL FOUR CHANNELS
    # =========================================================

    sr = sr[:, :4]

    # =========================================================
    # IMPORTANT:
    #
    # DO NOT DOWNsample 512 -> 256.
    #
    # LDSR-S2 is already producing
    # the required native 4x output.
    #
    # 128 pixels @ 10m
    #       ↓
    # 512 pixels @ 2.5m
    # =========================================================

    expected_sr = (
        1,
        4,
        patch_size * 4,
        patch_size * 4
    )

    if tuple(sr.shape) != expected_sr:

        raise RuntimeError(
            f"Expected LDSR output "
            f"{expected_sr}, "
            f"got {tuple(sr.shape)}"
        )

    print(
        "✓ Native 4x LDSR output confirmed"
    )

    # =========================================================
    # 7. CONVERT TO NUMPY
    # =========================================================

    output_image = (
        sr
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Expected:
    #
    # (512, 512, 4)

    expected_output = (
        patch_size * 4,
        patch_size * 4,
        4
    )

    if output_image.shape != expected_output:

        raise RuntimeError(
            f"Final SR shape is "
            f"{output_image.shape}, "
            f"but expected "
            f"{expected_output}."
        )

    # =========================================================
    # 8. UNCERTAINTY ESTIMATION
    # =========================================================

    n_samples = 4

    samples = []

    print(
        f"Generating {n_samples} "
        f"stochastic samples for uncertainty..."
    )

    with torch.no_grad():

        for i in range(n_samples):

            sample = model.forward(
                tensor,
                sampling_steps=sampling_steps
            )

            # Keep four channels
            sample = sample[:, :4]

            # IMPORTANT:
            # No 512 -> 256 downsampling.
            #
            # Keep native 512x512 output.

            sample_np = (
                sample
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            samples.append(
                sample_np
            )

            print(
                f"Uncertainty sample "
                f"{i + 1}/{n_samples}"
            )

    # Shape:
    #
    # (4 samples, 4 channels, 512, 512)

    samples = np.stack(
        samples,
        axis=0
    )

    expected_samples = (
        n_samples,
        4,
        patch_size * 4,
        patch_size * 4
    )

    if samples.shape != expected_samples:

        raise RuntimeError(
            f"Unexpected uncertainty "
            f"sample shape: {samples.shape}; "
            f"expected {expected_samples}"
        )

    # =========================================================
    # 9. CALCULATE UNCERTAINTY
    # =========================================================

    # Variance across stochastic samples
    #
    # Result before mean:
    #     (4 channels, 512, 512)
    #
    # Then average across four bands:
    #     (512, 512)

    uncertainty_map = (
        np.var(
            samples,
            axis=0
        )
        .mean(axis=0)
        .astype(np.float32)
    )

    expected_uncertainty = (
        patch_size * 4,
        patch_size * 4
    )

    if uncertainty_map.shape != expected_uncertainty:

        raise RuntimeError(
            f"Uncertainty shape is "
            f"{uncertainty_map.shape}, "
            f"but expected "
            f"{expected_uncertainty}."
        )

    # =========================================================
    # 10. FINAL RESULT
    # =========================================================

    print()
    print("=" * 60)
    print("GeoSharp Inference Complete")
    print("=" * 60)

    print(
        f"Input:       {tile.shape}"
    )

    print(
        f"SR output:   {output_image.shape}"
    )

    print(
        f"Uncertainty: {uncertainty_map.shape}"
    )

    print(
        "Resolution:  10m -> 2.5m"
    )

    print("=" * 60)

    return (
        output_image,
        uncertainty_map
    )


def run_sr(
    tile_path: str,
    sampling_steps: int = 100,
    patch_size: int = 128
):
    """
    Main GeoSharp inference function.

    Final resolution:
        10m -> 2.5m

    Final channels:
        B02, B03, B04, B08

    Returns:
        output_image:
            (512, 512, 4)

        uncertainty_map:
            (512, 512)
    """

    if USE_PLACEHOLDER:

        return _placeholder_run_sr(
            tile_path
        )

    return _real_run_sr(
        tile_path,
        sampling_steps=sampling_steps,
        patch_size=patch_size
    )


# =============================================================
# COMMAND LINE TEST
# =============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "GeoSharp 10m -> 2.5m "
            "super-resolution"
        )
    )

    parser.add_argument(
        "tile_path",
        nargs="?",
        default="data/raw/sample_tile.tif",
        help=(
            "Path to a tile file or folder "
            "containing B02/B03/B04/B08 files."
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help=(
            "Diffusion sampling steps. "
            "100 = final quality."
        )
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help=(
            "Input patch size. "
            "Default = 128."
        )
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print output shapes."
    )

    args = parser.parse_args()

    # =========================================================
    # RUN
    # =========================================================

    out, unc = run_sr(
        args.tile_path,
        sampling_steps=args.steps,
        patch_size=args.patch_size
    )

    print()
    print("=" * 60)
    print("GeoSharp Result")
    print("=" * 60)

    print(
        f"Output shape:      {out.shape}"
    )

    print(
        f"Uncertainty shape: {unc.shape}"
    )

    # =========================================================
    # HARD VALIDATION
    # =========================================================

    expected_output = (
        args.patch_size * 4,
        args.patch_size * 4,
        4
    )

    expected_uncertainty = (
        args.patch_size * 4,
        args.patch_size * 4
    )

    assert out.shape == expected_output, (
        f"Expected {expected_output}, "
        f"got {out.shape}"
    )

    assert unc.shape == expected_uncertainty, (
        f"Expected {expected_uncertainty}, "
        f"got {unc.shape}"
    )

    print()
    print(
        "✓ 10m -> 2.5m conversion confirmed"
    )

    print(
        "✓ 4-band RGB-NIR output confirmed"
    )

    print(
        "✓ Final SR shape:",
        out.shape
    )

    print(
        "✓ Final uncertainty:",
        unc.shape
    )