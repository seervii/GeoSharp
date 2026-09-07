"""
GeoSharp inference backend.

Loads ESA OpenSR's LDSR-S2 architecture and GeoSharp's fine-tuned
WorldStrat x4 checkpoint from Hugging Face, then exposes the same
run_sr(tile_path, sampling_steps, patch_size) -> (sr, uncertainty)
interface expected by the Streamlit app and diagnostic scripts.

Input:
    B02/B03/B04/B08
    128x128
    reflectance [0,1]

Output:
    B02/B03/B04/B08
    512x512
    reflectance [0,1]

Scale:
    4x (10m -> 2.5m)
"""

import os
import sys
from pathlib import Path

import numpy as np

# =============================================================
# SETTINGS
# =============================================================

USE_PLACEHOLDER = False

# Hugging Face repo containing the fine-tuned GeoSharp checkpoint.
HF_REPO_ID = os.getenv(
    "GEOSHARP_MODEL_REPO",
    "kapoorraaghav/geosharp-ldsrs2-worldstrat-x4",
)

HF_CHECKPOINT_FILENAME = os.getenv(
    "GEOSHARP_CHECKPOINT",
    "ldsrs2_worldstrat_x4_best.pt",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "config_10m.yaml"

_MODEL = None
_DEVICE = None
_MODEL_SOURCE = None

_MPS_PATCH_APPLIED = False


# =============================================================
# PLACEHOLDER
# =============================================================


def _placeholder_run_sr(tile_path: str):
    """Fake output for frontend-only testing."""
    return (
        np.random.rand(512, 512, 4).astype(np.float32),
        np.random.rand(512, 512).astype(np.float32),
    )


# =============================================================
# MPS COMPATIBILITY
# =============================================================


def _patch_opensr_mps_float64():
    """
    Patch OpenSR's sampler buffer registration for Apple MPS.

    OpenSR can create DDIM schedule tensors as float64. Apple MPS
    does not support float64 tensors, which causes errors such as:

        TypeError: Cannot convert a MPS Tensor to float64 dtype

    We intercept sampler register_buffer() methods and convert
    floating-point tensors/NumPy arrays to float32 before they are
    moved onto MPS.

    The patch is applied only when the selected device is MPS.
    """

    global _MPS_PATCH_APPLIED

    if _MPS_PATCH_APPLIED:
        return

    import inspect

    import torch
    from opensr_model.diffusion import utils as diffusion_utils

    patched_any = False

    for class_name, cls in inspect.getmembers(
        diffusion_utils,
        inspect.isclass,
    ):
        if not hasattr(cls, "register_buffer"):
            continue

        original = getattr(cls, "register_buffer")

        # Avoid patching the same class more than once during
        # Streamlit reruns/imports.
        if getattr(original, "_geosharp_mps_safe", False):
            patched_any = True
            continue

        def make_safe_register_buffer(original_method):
            def safe_register_buffer(self, name, attr):
                # Convert NumPy floating arrays before OpenSR moves them.
                if isinstance(attr, np.ndarray):
                    if np.issubdtype(attr.dtype, np.floating):
                        attr = torch.from_numpy(
                            np.asarray(
                                attr,
                                dtype=np.float32,
                            )
                        )

                # Convert floating tensors to float32.
                elif isinstance(attr, torch.Tensor):
                    if attr.is_floating_point() and attr.dtype != torch.float32:
                        attr = attr.to(dtype=torch.float32)

                return original_method(self, name, attr)

            safe_register_buffer._geosharp_mps_safe = True
            return safe_register_buffer

        setattr(
            cls,
            "register_buffer",
            make_safe_register_buffer(original),
        )

        patched_any = True

        print(
            f"Applied MPS float32 compatibility patch to {class_name}.register_buffer"
        )

    if not patched_any:
        raise RuntimeError(
            "Could not find an OpenSR sampler class with register_buffer(). "
            "The installed opensr_model package may have an incompatible "
            "diffusion implementation."
        )

    _MPS_PATCH_APPLIED = True


# =============================================================
# CONFIG
# =============================================================


def _load_config():
    """Load the pinned local OpenSR configuration."""
    from omegaconf import OmegaConf

    if not LOCAL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing LDSR-S2 config: {LOCAL_CONFIG_PATH}")

    return OmegaConf.load(LOCAL_CONFIG_PATH)


# =============================================================
# CHECKPOINT
# =============================================================


def _get_checkpoint_path() -> Path:
    """Download/cache the fine-tuned checkpoint from Hugging Face."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to load the fine-tuned "
            "GeoSharp model. Install it with:\n\n"
            "pip install huggingface_hub"
        ) from exc

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CHECKPOINT_FILENAME,
    )

    return Path(path)


# =============================================================
# DEVICE
# =============================================================


def _select_device(torch):
    """
    Select the best available accelerator.

    Priority:
        CUDA -> MPS -> CPU
    """

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# =============================================================
# MODEL LOADING
# =============================================================


def _get_model():
    """Load and cache the fine-tuned LDSR-S2 model once per process."""

    global _MODEL
    global _DEVICE
    global _MODEL_SOURCE

    if _MODEL is not None:
        return _MODEL, _DEVICE

    import opensr_model
    import torch

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    _DEVICE = _select_device(torch)

    print(f"Using device: {_DEVICE}")

    # ---------------------------------------------------------
    # MPS compatibility patch
    # ---------------------------------------------------------

    if _DEVICE == "mps":
        print("Apple Silicon MPS detected.")
        _patch_opensr_mps_float64()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = _load_config()

    # ---------------------------------------------------------
    # Instantiate official architecture
    # ---------------------------------------------------------

    print("Creating ESA OpenSR LDSR-S2 model...")

    model = opensr_model.SRLatentDiffusion(
        config,
        device=_DEVICE,
    )

    # ---------------------------------------------------------
    # Load ESA base checkpoint
    # ---------------------------------------------------------

    print("Loading ESA OpenSR base LDSR-S2 checkpoint...")

    model.load_pretrained(config.ckpt_version)

    # ---------------------------------------------------------
    # Load GeoSharp fine-tuned checkpoint
    # ---------------------------------------------------------

    checkpoint_path = _get_checkpoint_path()

    print(f"Loading GeoSharp fine-tuned checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=_DEVICE,
    )

    # The training notebook/test script saves a dictionary with
    # model_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Defensive fallback for a raw state_dict.
        state_dict = checkpoint

    # ---------------------------------------------------------
    # Load fine-tuned weights
    # ---------------------------------------------------------

    model.load_state_dict(state_dict)

    # Ensure model parameters are on the selected device.
    model = model.to(_DEVICE)

    # Inference mode.
    model.eval()

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    _MODEL = model

    _MODEL_SOURCE = f"hf://{HF_REPO_ID}/{HF_CHECKPOINT_FILENAME}"

    print("✓ GeoSharp fine-tuned LDSR-S2 loaded")

    if isinstance(checkpoint, dict):
        print(
            "  epoch:",
            checkpoint.get(
                "epoch",
                "N/A",
            ),
        )

        print(
            "  val_loss:",
            checkpoint.get(
                "val_loss",
                "N/A",
            ),
        )

    print(
        "  source:",
        _MODEL_SOURCE,
    )

    return _MODEL, _DEVICE


# =============================================================
# TIFF READER
# =============================================================


def _read_patch(
    tile_path: str,
    patch_size: int,
) -> np.ndarray:
    """
    Read a centered 4-band patch from GeoTIFF or SAFE-style
    band folder.
    """

    import rasterio
    from rasterio.windows import Window

    # ---------------------------------------------------------
    # SAFE-style folder
    # ---------------------------------------------------------

    if os.path.isdir(tile_path):
        from pipeline.preprocess import (
            read_center_patch_from_folder,
        )

        return read_center_patch_from_folder(
            tile_path,
            patch_size=patch_size,
        )

    # ---------------------------------------------------------
    # GeoTIFF
    # ---------------------------------------------------------

    with rasterio.open(tile_path) as src:
        if src.count < 4:
            raise ValueError(
                f"Expected at least 4 bands (B02, B03, B04, B08), got {src.count}."
            )

        if src.height < patch_size or src.width < patch_size:
            raise ValueError(
                f"Tile is {src.width}x{src.height}, "
                f"smaller than patch_size={patch_size}."
            )

        row_off = (src.height - patch_size) // 2

        col_off = (src.width - patch_size) // 2

        window = Window(
            col_off,
            row_off,
            patch_size,
            patch_size,
        )

        tile = src.read(
            indexes=[
                1,
                2,
                3,
                4,
            ],
            window=window,
        )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    tile = np.asarray(
        tile,
        dtype=np.float32,
    )

    tile = np.nan_to_num(
        tile,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # Handle Sentinel-2 DN-scaled imagery.
    if np.nanmax(tile) > 2.0:
        tile = tile / 10000.0

    tile = np.clip(
        tile,
        0.0,
        1.0,
    )

    return tile


# =============================================================
# TENSOR CONVERSION
# =============================================================


def _to_tensor(
    tile: np.ndarray,
    device: str,
):
    """Convert CHW NumPy input into float32 BCHW tensor."""

    import torch

    tile = np.asarray(
        tile,
        dtype=np.float32,
    )

    tensor = torch.from_numpy(tile).float()

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    return tensor


# =============================================================
# MODEL FORWARD
# =============================================================


def _forward(
    model,
    tensor,
    sampling_steps: int,
):
    """Run one SR inference pass."""

    import torch

    # Explicitly keep model input in float32.
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.float()

    with torch.no_grad():
        out = model.forward(
            tensor,
            sampling_steps=sampling_steps,
        )

    return out[:, :4]


# =============================================================
# MAIN SR PIPELINE
# =============================================================


def _real_run_sr(
    tile_path: str,
    sampling_steps: int = 100,
    patch_size: int = 128,
):
    """
    Run super-resolution plus stochastic uncertainty estimation.
    """

    import torch

    # ---------------------------------------------------------
    # Validate model-specific dimensions
    # ---------------------------------------------------------

    if patch_size != 128:
        raise ValueError(
            "GeoSharp's fine-tuned WorldStrat x4 checkpoint "
            "is trained for 128x128 LR input."
        )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model, device = _get_model()

    # ---------------------------------------------------------
    # Read input patch
    # ---------------------------------------------------------

    tile = _read_patch(
        tile_path,
        patch_size,
    )

    expected_input_shape = (
        4,
        patch_size,
        patch_size,
    )

    if tile.shape != expected_input_shape:
        raise RuntimeError(
            f"Expected input patch shape {expected_input_shape}, got {tile.shape}."
        )

    # ---------------------------------------------------------
    # Convert to tensor
    # ---------------------------------------------------------

    tensor = _to_tensor(
        tile,
        device,
    )

    # ---------------------------------------------------------
    # Primary SR result
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Running LDSR-S2 inference on {device}")
    print(f"Sampling steps: {sampling_steps}")
    print("=" * 60)

    sr = _forward(
        model,
        tensor,
        sampling_steps,
    )

    print("✓ Main SR generation complete")

    expected_output_shape = (
        1,
        4,
        patch_size * 4,
        patch_size * 4,
    )

    if tuple(sr.shape) != expected_output_shape:
        raise RuntimeError(
            f"Expected model output {expected_output_shape}, got {tuple(sr.shape)}."
        )

    # ---------------------------------------------------------
    # Convert primary SR to HWC NumPy
    # ---------------------------------------------------------

    output_image = (
        sr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    )

    output_image = np.nan_to_num(
        output_image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    output_image = np.clip(
        output_image,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Stochastic uncertainty
    # ---------------------------------------------------------
    #
    # Default is 2 samples for a much faster interactive demo.
    #
    # Set:
    #
    # GEOSHARP_UNCERTAINTY_SAMPLES=4
    #
    # or:
    #
    # GEOSHARP_UNCERTAINTY_SAMPLES=8
    #
    # for a more stable uncertainty estimate.
    # ---------------------------------------------------------

    n_samples = int(
        os.getenv(
            "GEOSHARP_UNCERTAINTY_SAMPLES",
            "2",
        )
    )

    n_samples = max(
        2,
        n_samples,
    )

    print(f"Generating {n_samples} stochastic samples for uncertainty...")

    samples = []

    print(f"Generating {n_samples} uncertainty samples:")

    for i in range(n_samples):
        # Terminal progress bar
        progress = (i + 1) / n_samples
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        sys.stdout.write(f"\r  [{bar}] {i + 1}/{n_samples} ({progress * 100:5.1f}%)")
        sys.stdout.flush()

        with torch.no_grad():
            sample = _forward(
                model,
                tensor,
                sampling_steps,
            )

        sample = sample.squeeze(0).detach().cpu().numpy().astype(np.float32)

        samples.append(sample)

    print("\n✓ Uncertainty sampling complete")

    # ---------------------------------------------------------
    # Stack stochastic samples
    # ---------------------------------------------------------

    samples = np.stack(
        samples,
        axis=0,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Per-pixel mean variance across 4 bands
    # ---------------------------------------------------------

    uncertainty_map = (
        np.var(
            samples,
            axis=0,
        )
        .mean(axis=0)
        .astype(np.float32)
    )

    uncertainty_map = np.nan_to_num(
        uncertainty_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    uncertainty_map = np.maximum(
        uncertainty_map,
        0.0,
    ).astype(np.float32)

    print("✓ SR + uncertainty complete")

    return (
        output_image,
        uncertainty_map,
    )


# =============================================================
# PUBLIC API
# =============================================================


def run_sr(
    tile_path: str,
    sampling_steps: int = 100,
    patch_size: int = 128,
):
    """
    Public interface used by the GeoSharp UI and diagnostics.
    """

    if USE_PLACEHOLDER:
        return _placeholder_run_sr(tile_path)

    return _real_run_sr(
        tile_path,
        sampling_steps=sampling_steps,
        patch_size=patch_size,
    )


# =============================================================
# CLI TEST
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=("GeoSharp 10m -> 2.5m inference test")
    )

    parser.add_argument(
        "tile_path",
        nargs="?",
        default="data/raw/sample_tile.tif",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    out, unc = run_sr(
        args.tile_path,
        sampling_steps=args.steps,
        patch_size=args.patch_size,
    )

    print(
        "Output shape:",
        out.shape,
    )

    print(
        "Uncertainty shape:",
        unc.shape,
    )
