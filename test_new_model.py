"""
GeoSharp — Test New Fine-Tuned Model

Run from project root:

    python test_new_model.py

Optional:

    python test_new_model.py --steps 30

    python test_new_model.py --steps 100

This evaluates the fine-tuned BEST checkpoint on 5 images
from the WorldStrat x4 test split and compares it against bicubic.
"""

import argparse
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import rasterio
import requests
import torch
import opensr_model
from omegaconf import OmegaConf
from skimage.transform import resize
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "datasets"
    / "worldstrat_x4"
)

INDEX_PATH = PROJECT_ROOT / "data" / "index.csv"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "finetuned"
    / "ldsrs2_worldstrat_x4_best.pt"
)

CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "ESAOpenSR/opensr-model/"
    "refs/heads/main/"
    "opensr_model/configs/config_10m.yaml"
)


# ------------------------------------------------------------
# TIFF READING
# ------------------------------------------------------------

def read_tiff(path):
    with rasterio.open(path) as src:
        return src.read()


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize(arr):
    arr = arr.astype(np.float32)

    if np.nanmax(arr) > 2.0:
        arr = arr / 10000.0

    return np.clip(arr, 0.0, 1.0)


# ------------------------------------------------------------
# BICUBIC BASELINE
# ------------------------------------------------------------

def bicubic(lr):

    lr = np.transpose(lr, (1, 2, 0))

    out = resize(
        lr,
        (512, 512, 4),
        order=3,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    )

    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main(steps):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("GEOSHARP — NEW MODEL TEST")
    print("=" * 70)

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint exists:", CHECKPOINT.exists())

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )


    # --------------------------------------------------------
    # TEST SPLIT — ONLY 5 IMAGES
    # --------------------------------------------------------

    index_df = pd.read_csv(
        INDEX_PATH,
        sep="\t"
    )

    test_df = (
        index_df[
            index_df["split"]
            .astype(str)
            .str.lower()
            == "test"
        ]
        .reset_index(drop=True)
        .head(5)
    )

    print("Test images:", len(test_df))


    # --------------------------------------------------------
    # LOAD BASE LDSR-S2
    # --------------------------------------------------------

    print("\nLoading LDSR-S2...")

    response = requests.get(
        CONFIG_URL,
        timeout=30
    )

    response.raise_for_status()

    config = OmegaConf.load(
        StringIO(response.text)
    )

    model = opensr_model.SRLatentDiffusion(
        config,
        device=device,
    )

    model.load_pretrained(
        config.ckpt_version
    )


    # --------------------------------------------------------
    # LOAD FINE-TUNED CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("✓ Fine-tuned checkpoint loaded")

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "N/A")
    )

    print(
        "Validation loss:",
        checkpoint.get("val_loss", "N/A")
    )

    print(
        "Sampling steps:",
        steps
    )


    # --------------------------------------------------------
    # METRIC STORAGE
    # --------------------------------------------------------

    geo_psnr = []
    geo_ssim = []

    bic_psnr = []
    bic_ssim = []


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    for i, row in test_df.iterrows():

        lr_path = (
            DATA_DIR
            / str(row["lr_img"]).strip()
        )

        hr_path = (
            DATA_DIR
            / str(row["hr_img"]).strip()
        )

        lr = normalize(
            read_tiff(lr_path)
        )

        hr = normalize(
            read_tiff(hr_path)
        )

        assert lr.shape == (
            4,
            128,
            128
        ), lr.shape

        assert hr.shape == (
            4,
            512,
            512
        ), hr.shape


        # ----------------------------------------------------
        # HWC FOR METRICS
        # ----------------------------------------------------

        hr_hwc = np.transpose(
            hr,
            (1, 2, 0)
        )


        # ----------------------------------------------------
        # GEOSHARP
        # ----------------------------------------------------

        tensor = (
            torch
            .from_numpy(lr)
            .float()
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            sr = model.forward(
                tensor,
                sampling_steps=steps,
            )

        sr = (
            sr
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        sr = np.transpose(
            sr,
            (1, 2, 0)
        )

        sr = np.clip(
            sr,
            0.0,
            1.0
        )

        assert sr.shape == (
            512,
            512,
            4
        ), sr.shape


        # ----------------------------------------------------
        # BICUBIC
        # ----------------------------------------------------

        bic = bicubic(lr)


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        p_geo = peak_signal_noise_ratio(
            hr_hwc,
            sr,
            data_range=1.0
        )

        s_geo = structural_similarity(
            hr_hwc,
            sr,
            channel_axis=-1,
            data_range=1.0,
        )

        p_bic = peak_signal_noise_ratio(
            hr_hwc,
            bic,
            data_range=1.0
        )

        s_bic = structural_similarity(
            hr_hwc,
            bic,
            channel_axis=-1,
            data_range=1.0,
        )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        geo_psnr.append(p_geo)
        geo_ssim.append(s_geo)

        bic_psnr.append(p_bic)
        bic_ssim.append(s_bic)


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"{i + 1:3d}/{len(test_df)} | "
            f"GeoSharp {p_geo:.3f} dB / {s_geo:.4f} | "
            f"Bicubic {p_bic:.3f} dB / {s_bic:.4f}"
        )


    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    geo_p = np.mean(geo_psnr)
    geo_s = np.mean(geo_ssim)

    bic_p = np.mean(bic_psnr)
    bic_s = np.mean(bic_ssim)


    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Test images   : {len(test_df)}"
    )

    print(
        f"GeoSharp PSNR : {geo_p:.3f} dB"
    )

    print(
        f"GeoSharp SSIM : {geo_s:.4f}"
    )

    print(
        f"Bicubic PSNR  : {bic_p:.3f} dB"
    )

    print(
        f"Bicubic SSIM  : {bic_s:.4f}"
    )

    print(
        f"PSNR Gain     : "
        f"{geo_p - bic_p:+.3f} dB"
    )

    print(
        f"SSIM Gain     : "
        f"{geo_s - bic_s:+.4f}"
    )

    print("=" * 70)


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help=(
            "Diffusion sampling steps. "
            "Use 30 for testing, 100 for final."
        ),
    )

    args = parser.parse_args()

    main(args.steps)