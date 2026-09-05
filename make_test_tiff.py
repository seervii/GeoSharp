"""
Create a 4-band GeoTIFF test image from extracted Sentinel-2 bands.

Expected input bands:
    B02 = Blue
    B03 = Green
    B04 = Red
    B08 = NIR

The script reads a centered 128x128 patch and writes:
    data/test_input_4band.tif

Output shape:
    4 x 128 x 128

This is a real 4-band GeoTIFF made from the Indian Sentinel-2 data.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parent

# Extracted tile from the existing Indian Sentinel-2 test
TILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "temp_tiles"
    / "S2C_MSIL2A_20260715T053641_N0512_R005_T43REP_20260715T101509_SAFE"
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "test_input_4band.tif"

PATCH_SIZE = 128

BANDS = {
    "B02": None,
    "B03": None,
    "B04": None,
    "B08": None,
}


def find_band(band_name):
    """Find the extracted JP2 for a Sentinel-2 band."""

    matches = list(TILE_DIR.rglob(f"*_{band_name}_10m.jp2"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {band_name} JP2 inside:\n{TILE_DIR}"
        )

    return matches[0]


def main():

    print("=" * 70)
    print("CREATING 4-BAND TEST GEOTIFF")
    print("=" * 70)

    if not TILE_DIR.exists():
        raise FileNotFoundError(
            f"Extracted tile directory not found:\n{TILE_DIR}\n\n"
            "Run test_indian_tiles.py first so the bands are extracted."
        )

    band_paths = {}

    for band in BANDS:
        path = find_band(band)
        band_paths[band] = path
        print(f"{band}: {path.name}")

    # ---------------------------------------------------------
    # Read centered 128x128 patch from B02 to determine window
    # ---------------------------------------------------------

    with rasterio.open(band_paths["B02"]) as src:

        height = src.height
        width = src.width

        row_off = (height - PATCH_SIZE) // 2
        col_off = (width - PATCH_SIZE) // 2

        window = Window(
            col_off,
            row_off,
            PATCH_SIZE,
            PATCH_SIZE
        )

        profile = src.profile.copy()

        # Read all four bands using the SAME window.
        arrays = []

        for band in ["B02", "B03", "B04", "B08"]:

            with rasterio.open(band_paths[band]) as band_src:

                arr = band_src.read(
                    1,
                    window=window
                )

                arrays.append(arr)

        # -----------------------------------------------------
        # Stack:
        #   B02
        #   B03
        #   B04
        #   B08
        # -----------------------------------------------------

        data = np.stack(arrays, axis=0)

        # Convert to float32.
        data = data.astype(np.float32)

        # Sentinel-2 JP2 values are generally reflectance-scaled.
        # Your existing pipeline is already giving values around
        # 0.08-0.63, so only divide if raw DN values are detected.

        if data.max() > 2.0:
            print("Raw DN values detected -> dividing by 10000.")
            data /= 10000.0

        print()
        print(f"Created array shape: {data.shape}")
        print(f"Value range: {data.min()} -> {data.max()}")

        # -----------------------------------------------------
        # Create GeoTIFF profile
        # -----------------------------------------------------

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=4,
            height=PATCH_SIZE,
            width=PATCH_SIZE,
            compress="lzw"
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Write 4-band GeoTIFF
        # -----------------------------------------------------

        with rasterio.open(
            OUTPUT_PATH,
            "w",
            **profile
        ) as dst:

            dst.write(data)

            dst.set_band_description(1, "B02 - Blue")
            dst.set_band_description(2, "B03 - Green")
            dst.set_band_description(3, "B04 - Red")
            dst.set_band_description(4, "B08 - NIR")

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)
    print(f"TIFF created:")
    print(OUTPUT_PATH)
    print()
    print("Bands:")
    print("  1 = B02 Blue")
    print("  2 = B03 Green")
    print("  3 = B04 Red")
    print("  4 = B08 NIR")
    print()
    print("Input shape:  128 x 128 x 4")
    print("Resolution:   Sentinel-2 10m")
    print("=" * 70)


if __name__ == "__main__":
    main()
