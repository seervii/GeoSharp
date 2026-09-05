"""
Basic preprocessing for Sentinel-2 L2A tiles before running super-resolution.

For the prototype (RGB only, single tiles), this stays simple:
  1. Read the tile
  2. Select RGB bands (B04, B03, B02 for Sentinel-2)
  3. Normalize reflectance values to [0, 1]
  4. Save as a processed array/GeoTIFF ready for inference.py

Full cloud-masking / co-registration (as in the proposed full architecture)
is NOT needed here since Copernicus L2A tiles are already atmospherically
corrected, and we're hand-picking clear tiles for the demo.
"""

import os
import re

import numpy as np
import rasterio

# Matches SAFE-style per-band filenames, e.g.
# T43RFM_20260627T052639_B02_10m.jp2
_BAND_FILENAME_RE = re.compile(r"_(B0[2348])_10m\.(jp2|tif|tiff)$", re.IGNORECASE)


def stack_bands_from_folder(folder_path: str, out_path: str = None) -> str:
    """
    Stack individual Sentinel-2 band files (B02, B03, B04, B08 -- Blue,
    Green, Red, NIR) from a folder of per-band JP2/TIFF files into a
    single 4-band GeoTIFF, in the exact channel order opensr-model expects
    (see inference.py's _real_run_sr docstring: B02, B03, B04, B08).

    Looks for files matching the standard SAFE-style naming pattern, e.g.:
        T43RFM_20260627T052639_B02_10m.jp2
        T43RFM_20260627T052639_B03_10m.jp2
        T43RFM_20260627T052639_B04_10m.jp2
        T43RFM_20260627T052639_B08_10m.jp2

    Args:
        folder_path: directory containing the four per-band files.
        out_path: where to write the stacked GeoTIFF. Defaults to
            <folder_path>/stacked_4band.tif.

    Returns:
        Path to the stacked 4-band GeoTIFF.
    """
    band_order = ["B02", "B03", "B04", "B08"]
    band_files = {}

    for fname in os.listdir(folder_path):
        match = _BAND_FILENAME_RE.search(fname)
        if match:
            band_files[match.group(1).upper()] = os.path.join(folder_path, fname)

    missing = [b for b in band_order if b not in band_files]
    if missing:
        raise FileNotFoundError(
            f"Could not find band file(s) {missing} in '{folder_path}'. "
            f"Expected filenames ending in '_B02_10m.jp2', '_B03_10m.jp2', "
            f"'_B04_10m.jp2', '_B08_10m.jp2' (or .tif)."
        )

    # Use the first band's georeferencing/profile as the base for the stack.
    with rasterio.open(band_files["B02"]) as src0:
        profile = src0.profile.copy()

    bands = []
    for b in band_order:
        with rasterio.open(band_files[b]) as src:
            bands.append(src.read(1))
    stacked = np.stack(bands, axis=0)  # (4, H, W) = B02, B03, B04, B08

    profile.update(driver="GTiff", count=4, dtype=stacked.dtype)

    if out_path is None:
        out_path = os.path.join(folder_path, "stacked_4band.tif")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    return out_path


def read_center_patch_from_folder(
    folder_path: str, patch_size: int = 128
) -> np.ndarray:
    """
    Read a single centered patch (default 128x128) directly from the
    per-band SAFE-style files, WITHOUT ever materializing a full-tile
    array in memory or on disk.

    opensr-model's raw SRLatentDiffusion.forward() only handles small
    patches (its no-data-mask upsampling allocates target_size**2 floats,
    which is fine at 128x128 but blows up to tens of GB on a full
    ~11000x11000 Sentinel-2 tile). This function is the "Option A" fix:
    grab one representative patch straight off disk via rasterio's
    windowed read, so the full-resolution raster is never loaded.

    Args:
        folder_path: directory containing the four per-band files
            (see stack_bands_from_folder for the expected naming pattern).
        patch_size: side length of the square patch to read (default 128,
            matching opensr-model's expected input size).

    Returns:
        np.ndarray of shape (4, patch_size, patch_size), band order
        B02, B03, B04, B08 (Blue, Green, Red, NIR).
    """
    from rasterio.windows import Window

    band_order = ["B02", "B03", "B04", "B08"]
    band_files = {}

    for fname in os.listdir(folder_path):
        match = _BAND_FILENAME_RE.search(fname)
        if match:
            band_files[match.group(1).upper()] = os.path.join(folder_path, fname)

    missing = [b for b in band_order if b not in band_files]
    if missing:
        raise FileNotFoundError(
            f"Could not find band file(s) {missing} in '{folder_path}'. "
            f"Expected filenames ending in '_B02_10m.jp2', '_B03_10m.jp2', "
            f"'_B04_10m.jp2', '_B08_10m.jp2' (or .tif)."
        )

    bands = []
    for b in band_order:
        with rasterio.open(band_files[b]) as src:
            h, w = src.height, src.width
            if h < patch_size or w < patch_size:
                raise ValueError(
                    f"Band {b} is {w}x{h}, smaller than requested "
                    f"patch_size={patch_size}."
                )
            row_off = (h - patch_size) // 2
            col_off = (w - patch_size) // 2
            window = Window(col_off, row_off, patch_size, patch_size)
            bands.append(src.read(1, window=window))

    stacked = np.stack(bands, axis=0)  # (4, patch_size, patch_size), raw reflectance
    # Sentinel-2 L2A reflectance is scaled by 10000 -- normalize to [0, 1]
    # before feeding the model, same as normalize_reflectance() does for
    # the RGB-only path. Without this, values like 3000-8000 blow past the
    # model's expected [0,1] range and the output saturates to white after
    # the np.clip(out, 0, 1) step in inference.py.
    return normalize_reflectance(stacked)


def load_rgb_bands(tile_path: str) -> np.ndarray:
    """
    Load and stack the RGB bands (B04=Red, B03=Green, B02=Blue) from a
    Sentinel-2 L2A GeoTIFF.

    Assumes a tile that already contains these bands in some order --
    adjust band indices if your tile layout differs.
    """
    with rasterio.open(tile_path) as src:
        # Adjust these indices to match your actual tile's band order
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)

    rgb = np.stack([red, green, blue], axis=-1)
    return rgb


def normalize_reflectance(rgb: np.ndarray, scale: float = 10000.0) -> np.ndarray:
    """
    Sentinel-2 L2A reflectance values are typically scaled by 10000.
    Normalize to [0, 1] and clip outliers.
    """
    normalized = rgb.astype(np.float32) / scale
    return np.clip(normalized, 0.0, 1.0)


def preprocess_tile(tile_path: str) -> np.ndarray:
    """Full preprocessing pipeline for one tile: load -> normalize."""
    rgb = load_rgb_bands(tile_path)
    return normalize_reflectance(rgb)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_tile.tif"
    processed = preprocess_tile(path)
    print(f"Preprocessed tile shape: {processed.shape}, "
          f"range: [{processed.min():.3f}, {processed.max():.3f}]")