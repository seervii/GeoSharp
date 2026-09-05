"""
GeoSharp inference module.

Exposes ONE shared function that the rest of the pipeline (and the UI)
depends on:

    run_sr(tile_path: str) -> (output_image: np.ndarray, uncertainty_map: np.ndarray)

While the real opensr-model integration is being wired up (GPU setup,
checkpoint download, etc.), leave USE_PLACEHOLDER = True so the UI person
can build and test against fake data. Flip it to False once opensr-model
is confirmed working end-to-end.
"""

import numpy as np

USE_PLACEHOLDER = False  # flip to False once opensr-model is installed & tested


def _placeholder_run_sr(tile_path: str):
    """Fake output so the UI can be built before the real model works."""
    fake_output = np.random.rand(512, 512, 3).astype(np.float32)
    fake_uncertainty = np.random.rand(512, 512).astype(np.float32)
    return fake_output, fake_uncertainty


_MODEL = None  # module-level cache so we don't reload weights on every call
_DEVICE = None


def _get_model():
    """
    Load the LDSR-S2 model + pretrained weights once, cache it for reuse.

    Confirmed API from https://github.com/ESAOpenSR/opensr-model (README,
    checked Sep 2026):

        config = OmegaConf.load(<config_10m.yaml>)
        model = opensr_model.SRLatentDiffusion(config, device=device)
        model.load_pretrained(config.ckpt_version)
        sr = model.forward(tensor, sampling_steps=100)

    Input tensor shape is (B, 4, H, W) -- 4 channels = RGB-NIR, NOT plain
    RGB. Default patch size is 128x128 -> upsampled to 512x512 (factor=4).
    """
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    import torch
    import requests
    from io import StringIO
    from omegaconf import OmegaConf
    import opensr_model

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Config is fetched from the opensr-model repo directly (per their README)
    config_url = (
        "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/"
        "refs/heads/main/opensr_model/configs/config_10m.yaml"
    )
    response = requests.get(config_url)
    config = OmegaConf.load(StringIO(response.text))

    model = opensr_model.SRLatentDiffusion(config, device=_DEVICE)
    model.load_pretrained(config.ckpt_version)  # auto-downloads checkpoint

    _MODEL = model
    return _MODEL, _DEVICE


def _real_run_sr(tile_path: str, sampling_steps: int = 100, patch_size: int = 128):
    """
    Real inference using the pretrained LDSR-S2 model via opensr-model.

    Requires:
        pip install opensr-model opensr-utils
        (see requirements.txt for the CUDA 12.8 torch note if you're on
        an RTX 50-series GPU)

    NOTE: opensr-model's raw model expects a 4-channel (RGB-NIR) tensor,
    NOT plain RGB -- if your tile only has RGB bands, you'll need a
    placeholder/duplicate 4th channel or use a NIR band from the tile.

    IMPORTANT (demo scope -- "Option A"): opensr-model's raw
    SRLatentDiffusion.forward() only handles small patches. Its internal
    no-data-mask step allocates a (target_size x target_size) float array
    where target_size = input_width * 4, so feeding it a full ~11000x11000
    Sentinel-2 tile tries to allocate tens of GB and crashes. For the SIH
    demo we read a single centered patch_size x patch_size patch directly
    off disk via a windowed rasterio read (see preprocess.py's
    read_center_patch_from_folder) -- the full-resolution raster is never
    loaded into memory. Full-tile tiling/stitching via opensr-utils
    (large_file_processing) is documented below as future work, since it
    is far slower and doesn't have a built-in uncertainty map.
    """
    import os

    import torch
    import rasterio

    try:
        from pipeline.preprocess import read_center_patch_from_folder
    except ImportError:
        from preprocess import read_center_patch_from_folder

    model, device = _get_model()

    # If given a folder of per-band files (e.g. a SAFE-style download with
    # separate B02/B03/B04/B08 JP2s), read one centered patch directly
    # off disk -- no full-tile array is ever created.
    if os.path.isdir(tile_path):
        tile = read_center_patch_from_folder(tile_path, patch_size=patch_size)
    else:
        # Single-file tile: read a centered window the same way, so a
        # large single GeoTIFF doesn't blow up memory either.
        with rasterio.open(tile_path) as src:
            h, w = src.height, src.width
            if h < patch_size or w < patch_size:
                raise ValueError(
                    f"Tile is {w}x{h}, smaller than requested "
                    f"patch_size={patch_size}."
                )
            from rasterio.windows import Window

            row_off = (h - patch_size) // 2
            col_off = (w - patch_size) // 2
            window = Window(col_off, row_off, patch_size, patch_size)
            tile = src.read(window=window)  # shape: (bands, patch_size, patch_size)

    if tile.shape[0] < 4:
        raise ValueError(
            f"Expected at least 4 bands (RGB-NIR) for opensr-model, got "
            f"{tile.shape[0]}. Check preprocess.py band selection."
        )

    tensor = torch.from_numpy(tile[:4]).float().unsqueeze(0).to(device)  # (1,4,H,W)

    # Normalize to match training distribution if not already done in
    # preprocess.py -- opensr-model expects reflectance-scale input.
    # tensor = tensor / 10000.0  # uncomment if tile isn't pre-normalized

    # 2. Run inference on a single 128x128 patch.
    with torch.no_grad():
        sr = model.forward(tensor, sampling_steps=sampling_steps)

    output_image = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H,W,C)
    output_image = output_image[:, :, :3]  # drop NIR for RGB display in UI

    # 3. Uncertainty map: LDSR-S2 estimates this via multi-sample variance
    #    (per the paper/demo.py in their repo) rather than a single
    #    deterministic forward pass. Run forward() multiple times and take
    #    the pixel-wise variance across samples as an uncertainty proxy.
    #    Check their demo.py for the exact recommended sample count --
    #    starting with 4 samples here as a reasonable default.
    n_samples = 4
    samples = [
        model.forward(tensor, sampling_steps=sampling_steps)
        .squeeze(0)[:3]
        .cpu()
        .numpy()
        for _ in range(n_samples)
    ]
    uncertainty_map = np.var(np.stack(samples), axis=0).mean(axis=0)  # (H,W)

    return output_image, uncertainty_map

    # For a FULL tile (not just one 128x128 patch), use opensr-utils
    # instead, which handles tiling/stitching/georeferencing:
    #
    #   import opensr_utils
    #   sr_job = opensr_utils.large_file_processing(
    #       root=tile_path,
    #       model=model,
    #       window_size=(128, 128),
    #       factor=4,
    #       overlap=12,
    #       eliminate_border_px=2,
    #       device=device,
    #       gpus=0,
    #   )
    #   # consult opensr-utils docs for how sr_job exposes the output
    #   # array and whether it provides uncertainty directly


def run_sr(tile_path: str, sampling_steps: int = 30, patch_size: int = 128):
    """
    Super-resolve a Sentinel-2 tile.

    Args:
        tile_path: path to a Sentinel-2 tile (GeoTIFF or similar).
        sampling_steps: diffusion sampling steps for LDSR-S2. The
            opensr-model default/paper setting is 100, which is slow on
            CPU (5 forward passes total: 1 output + 4 for the uncertainty
            variance estimate). Default here is lowered to 30 for faster
            demo iteration -- quality is somewhat softer but still fine
            for a live pitch. Bump back to 100 for final result screenshots.
        patch_size: side length of the square patch read from the tile
            (default 128, matching opensr-model's expected input size).

    Returns:
        output_image: np.ndarray, sharpened RGB image
        uncertainty_map: np.ndarray, per-pixel confidence/uncertainty map
    """
    if USE_PLACEHOLDER:
        return _placeholder_run_sr(tile_path)
    return _real_run_sr(tile_path, sampling_steps=sampling_steps, patch_size=patch_size)


if __name__ == "__main__":
    # Quick manual test:
    #   python pipeline/inference.py data/raw/tile1_bands/
    #   python pipeline/inference.py data/raw/tile1_bands/ --steps 10   (fast preview)
    #   python pipeline/inference.py data/raw/tile1_bands/ --steps 100 --patch-size 128  (final quality)
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run GeoSharp super-resolution on a tile.")
    parser.add_argument(
        "tile_path", nargs="?", default="data/raw/sample_tile.tif",
        help="Path to a tile file, or a folder of per-band SAFE-style files.",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Diffusion sampling steps (default 30, fast demo preview; use 100 for final quality).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128,
        help="Square patch size read from the tile (default 128).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs",
        help="Directory to save output PNGs into (default: outputs/).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving PNGs, just print array shapes (old behavior).",
    )
    args = parser.parse_args()

    out, unc = run_sr(args.tile_path, sampling_steps=args.steps, patch_size=args.patch_size)
    print(f"Output shape: {out.shape}, Uncertainty shape: {unc.shape}")

    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)

        # 1. Sharpened output -> PNG. Values should already be roughly in
        #    [0, 1] (reflectance-scale) but clip defensively before
        #    converting to uint8.
        from PIL import Image

        out_clipped = np.clip(out, 0.0, 1.0)
        out_uint8 = (out_clipped * 255).astype(np.uint8)
        sr_path = os.path.join(args.out_dir, "sr_output.png")
        Image.fromarray(out_uint8).save(sr_path)

        # 2. Uncertainty map -> colored heatmap (brighter/hotter = higher
        #    variance across the 4 diffusion samples = less confident).
        import matplotlib.pyplot as plt

        unc_path = os.path.join(args.out_dir, "uncertainty_map.png")
        plt.imsave(unc_path, unc, cmap="inferno")

        print(f"Saved sharpened output to: {sr_path}")
        print(f"Saved uncertainty heatmap to: {unc_path}")


