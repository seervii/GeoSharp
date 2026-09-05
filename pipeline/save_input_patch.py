"""
Save the same center patch that inference.py feeds to the model, as a
viewable RGB PNG -- for before/after comparison against sr_output.png.

Usage:
    python pipeline/save_input_patch.py data/raw/tile1_bands/
    python pipeline/save_input_patch.py data/raw/tile1_bands/ --patch-size 128 --out-dir outputs
"""

import argparse
import os

import numpy as np
from PIL import Image

try:
    from pipeline.preprocess import read_center_patch_from_folder
except ImportError:
    from preprocess import read_center_patch_from_folder


def save_input_patch(folder_path: str, patch_size: int = 128, out_dir: str = "outputs"):
    # (4, H, W) = B02, B03, B04, B08 -- already normalized to [0, 1]
    # by the fixed read_center_patch_from_folder.
    patch = read_center_patch_from_folder(folder_path, patch_size=patch_size)

    # RGB display order is B04 (Red), B03 (Green), B02 (Blue).
    red = patch[2]
    green = patch[1]
    blue = patch[0]
    rgb = np.stack([red, green, blue], axis=-1)

    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "input_patch.png")
    Image.fromarray(rgb_uint8).save(out_path)
    print(f"Saved input patch to: {out_path}")
    print(f"Shape: {rgb_uint8.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the input patch as a PNG for comparison.")
    parser.add_argument("folder_path", help="Folder of per-band SAFE-style files (same as inference.py).")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    save_input_patch(args.folder_path, patch_size=args.patch_size, out_dir=args.out_dir)
