"""
Run inference on a tile AND compute PSNR/SSIM vs the bicubic baseline,
all from the command line (no notebook needed).

Usage:
    python pipeline/run_metrics.py data/raw/tile1_bands/
    python pipeline/run_metrics.py data/raw/tile1_bands/ --steps 30 --patch-size 128
"""

import argparse
import os

import numpy as np

try:
    from pipeline.inference import run_sr
    from pipeline.preprocess import read_center_patch_from_folder
    from pipeline.metrics import compare_to_baseline
except ImportError:
    from inference import run_sr
    from preprocess import read_center_patch_from_folder
    from metrics import compare_to_baseline


def main():
    parser = argparse.ArgumentParser(description="Run inference + compute PSNR/SSIM vs bicubic baseline.")
    parser.add_argument("tile_path", help="Folder of per-band SAFE-style files (same as inference.py).")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    # 1. Get the same normalized low-res input patch that was fed to the model.
    #    (B02, B03, B04, B08) -> reorder to RGB for the metrics functions.
    low_res_bands = read_center_patch_from_folder(args.tile_path, patch_size=args.patch_size)
    low_res_rgb = np.stack([low_res_bands[2], low_res_bands[1], low_res_bands[0]], axis=-1)

    # 2. Run the actual SR inference (uses USE_PLACEHOLDER flag in inference.py).
    sr_output, uncertainty_map = run_sr(args.tile_path, sampling_steps=args.steps, patch_size=args.patch_size)
    sr_output = np.clip(sr_output, 0.0, 1.0)

    # 3. Compute PSNR/SSIM of sr_output vs a bicubic upsample of the same low-res input.
    results = compare_to_baseline(low_res_rgb, sr_output, scale_factor=sr_output.shape[0] // low_res_rgb.shape[0])

    print("Metrics (SR output vs bicubic baseline):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
