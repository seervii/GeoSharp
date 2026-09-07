# pipeline/test_indian_tiles.py

import sys
import zipfile
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


# ============================================================
# PATHS
# ============================================================

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

# Your existing setup may have raw directly in project root
RAW_CANDIDATES = [
    PROJECT_ROOT / "raw",
    PROJECT_ROOT / "data" / "raw",
]

# Use whichever raw folder actually contains ZIPs
RAW_DIR = None

for folder in RAW_CANDIDATES:
    if folder.exists() and list(folder.glob("*.zip")):
        RAW_DIR = folder
        break

# If no ZIP found, prefer existing raw folder
if RAW_DIR is None:
    for folder in RAW_CANDIDATES:
        if folder.exists():
            RAW_DIR = folder
            break

# Final fallback
if RAW_DIR is None:
    RAW_DIR = PROJECT_ROOT / "raw"

TEMP_DIR = PROJECT_ROOT / "data" / "temp_tiles"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetuned_2"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPELINE_DIR))


# ============================================================
# IMPORT PIPELINE
# ============================================================

from preprocess import read_center_patch_from_folder
import inference as inference_module
from inference import run_sr

# ============================================================
# FINE-TUNED MODEL
# ============================================================

FINETUNED_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "finetuned"
    / "ldsrs2_worldstrat_x4_best.pt"
)

def load_finetuned_model():
    """
    Load the ESA LDSR-S2 architecture and replace its pretrained
    weights with our fine-tuned GeoSharp BEST checkpoint.

    inference.run_sr() reuses inference_module._MODEL, so the
    existing testing / hallucination pipeline automatically runs
    with the new fine-tuned model.
    """
    import requests
    from io import StringIO
    from omegaconf import OmegaConf
    import opensr_model

    print()
    print("=" * 70)
    print("LOADING FINETUNED GEOSHARP MODEL")
    print("=" * 70)

    if not FINETUNED_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found:\n{FINETUNED_CHECKPOINT}"
        )

    config_url = (
        "https://raw.githubusercontent.com/"
        "ESAOpenSR/opensr-model/"
        "refs/heads/main/"
        "opensr_model/configs/config_10m.yaml"
    )

    response = requests.get(config_url, timeout=30)
    response.raise_for_status()

    config = OmegaConf.load(StringIO(response.text))

    model = opensr_model.SRLatentDiffusion(
        config,
        device=DEVICE,
    )

    # Load original ESA architecture/weights first.
    model.load_pretrained(config.ckpt_version)

    checkpoint = torch.load(
        FINETUNED_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    # Put the fine-tuned model into the existing inference cache.
    inference_module._MODEL = model
    inference_module._DEVICE = DEVICE

    print("✓ Fine-tuned model loaded successfully")
    print("Checkpoint:", FINETUNED_CHECKPOINT)
    print("Epoch:", checkpoint.get("epoch", "N/A"))
    print("Validation loss:", checkpoint.get("val_loss", "N/A"))
    print("=" * 70)

from hallucination_check import (
    lr_consistency_error,
    prepare_uncertainty_map,
    create_hallucination_risk_mask,
    percentile_normalize,
)


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 128
SAMPLING_STEPS = 30
BLUR_SIGMA = 1.0
RISK_PERCENTILE = 90.0

REQUIRED_BANDS = ["B02", "B03", "B04", "B08"]


# ============================================================
# DEVICE
# ============================================================

print()
print("=" * 70)
print("DEVICE CHECK")
print("=" * 70)

if torch.cuda.is_available():
    print("CUDA available : YES")
    print("GPU            :", torch.cuda.get_device_name(0))
    DEVICE = "cuda"
else:
    print("CUDA available : NO")
    print("Using CPU.")
    DEVICE = "cpu"

print("=" * 70)


# ============================================================
# FIND ZIP FILES
# ============================================================

def find_tiles():

    print()
    print("=" * 70)
    print("SEARCHING FOR SENTINEL-2 ZIP FILES")
    print("=" * 70)

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Checking:")

    for folder in RAW_CANDIDATES:
        print(" ", folder)

    zip_files = []

    # Search both expected locations
    for folder in RAW_CANDIDATES:

        if not folder.exists():
            continue

        found = list(folder.glob("*.zip"))

        if found:
            print()
            print("Found ZIP files in:")
            print(folder)

            for z in found:
                print(" ", z.name)

            zip_files.extend(found)

    # Remove duplicates
    zip_files = list(dict.fromkeys(zip_files))

    # --------------------------------------------------------
    # If still nothing, recursively search project
    # --------------------------------------------------------

    if not zip_files:

        print()
        print("No ZIPs in standard folders.")
        print("Searching project recursively...")

        for z in PROJECT_ROOT.rglob("*.zip"):

            # Don't search inside temp/output folders
            if "temp_tiles" in z.parts:
                continue

            if "outputs" in z.parts:
                continue

            zip_files.append(z)

    # Remove duplicates
    zip_files = list(dict.fromkeys(zip_files))

    if not zip_files:

        print()
        print("ERROR: No Sentinel-2 ZIP files found anywhere")
        print("inside the project.")

        return []

    print()
    print("ZIP FILES TO PROCESS:")

    for i, z in enumerate(zip_files, 1):
        print(f"{i}. {z}")

    return zip_files


# ============================================================
# EXTRACT ONLY REQUIRED 10m BANDS
# ============================================================

def extract_required_bands(zip_path):

    tile_name = zip_path.stem

    safe_name = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in tile_name
    )

    output_folder = TEMP_DIR / safe_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print()
    print("-" * 70)
    print("EXTRACTING")
    print("-" * 70)
    print("ZIP :", zip_path)
    print("OUT :", output_folder)

    extracted = []

    try:

        with zipfile.ZipFile(zip_path, "r") as z:

            for member in z.namelist():

                member_lower = member.lower()

                # We only need 10m data
                if "/img_data/r10m/" not in member_lower:
                    continue

                matched_band = None

                for band in REQUIRED_BANDS:

                    if (
                        f"_{band.lower()}_10m.jp2"
                        in member_lower
                    ):
                        matched_band = band
                        break

                    if (
                        f"_{band.lower()}_10m.tif"
                        in member_lower
                    ):
                        matched_band = band
                        break

                    if (
                        f"_{band.lower()}_10m.tiff"
                        in member_lower
                    ):
                        matched_band = band
                        break

                if matched_band is None:
                    continue

                filename = Path(member).name
                destination = output_folder / filename

                if destination.exists():

                    print(
                        f"{matched_band}: already extracted"
                    )

                    extracted.append(destination)
                    continue

                print(
                    f"{matched_band}: extracting..."
                )

                with z.open(member) as src:
                    with open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                extracted.append(destination)

        # ----------------------------------------------------
        # Check bands
        # ----------------------------------------------------

        print()
        print("BAND CHECK")

        valid = True

        for band in REQUIRED_BANDS:

            matches = [
                f for f in extracted
                if f"_{band}_10m".lower()
                in f.name.lower()
            ]

            if matches:
                print(
                    f"  {band}: OK -> {matches[0].name}"
                )
            else:
                print(
                    f"  {band}: MISSING"
                )
                valid = False

        if not valid:

            print()
            print(
                "ERROR: Required bands are missing."
            )

            return None

        print()
        print("Extraction successful.")

        return output_folder

    except Exception as e:

        print()
        print("EXTRACTION ERROR:")
        print(type(e).__name__, e)

        return None


# ============================================================
# SAVE RGB
# ============================================================

def save_rgb(image, path):

    image = np.asarray(image)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    image = np.clip(image, 0, 1)

    image = (
        image * 255
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


# ============================================================
# SAVE MAP
# ============================================================

def save_map(image, path):

    image = np.asarray(image)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    image = percentile_normalize(
        image,
        low_percentile=1,
        high_percentile=99,
    )

    image = np.clip(image, 0, 1)

    image = (
        image * 255
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(mask, path):

    mask = np.asarray(mask).astype(bool)

    image = (
        mask * 255
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


# ============================================================
# SAVE OVERLAY
# ============================================================

def save_overlay(rgb, mask, path):

    rgb = np.clip(
        np.asarray(rgb),
        0,
        1,
    ).copy()

    mask = np.asarray(mask).astype(bool)

    # Red = high hallucination risk
    rgb[mask] = [1.0, 0.0, 0.0]

    image = (
        rgb * 255
    ).astype(np.uint8)

    Image.fromarray(image).save(path)


# ============================================================
# PROCESS ONE TILE
# ============================================================

def process_tile(zip_path):

    tile_name = zip_path.stem

    print()
    print()
    print("#" * 70)
    print("PROCESSING:", tile_name)
    print("#" * 70)

    # --------------------------------------------------------
    # 1. EXTRACT
    # --------------------------------------------------------

    tile_folder = extract_required_bands(zip_path)

    if tile_folder is None:
        return False

    # --------------------------------------------------------
    # 2. READ LR PATCH
    # --------------------------------------------------------

    print()
    print("Reading Sentinel-2 patch...")

    try:

        lr_patch = read_center_patch_from_folder(
            str(tile_folder),
            patch_size=PATCH_SIZE,
        )

    except Exception as e:

        print()
        print("PATCH READING ERROR:")
        print(type(e).__name__, e)

        return False

    print(
        "Patch shape:",
        lr_patch.shape,
    )

    print(
        "Patch range:",
        float(lr_patch.min()),
        "to",
        float(lr_patch.max()),
    )

    # CHW -> HWC
    if lr_patch.shape[0] == 4:

        lr_rgb = np.transpose(
            lr_patch[:3],
            (1, 2, 0),
        )

    else:

        print(
            "ERROR: Expected 4-band input."
        )

        return False

    # --------------------------------------------------------
    # 3. SUPER RESOLUTION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING LDSR-S2")
    print("=" * 70)

    print(
        f"Sampling steps: {SAMPLING_STEPS}"
    )

    try:

        # IMPORTANT:
        # run_sr returns TWO values
        sr_output, uncertainty_raw = run_sr(
            str(tile_folder),
            sampling_steps=SAMPLING_STEPS,
            patch_size=PATCH_SIZE,
        )

    except Exception as e:

        print()
        print("SR ERROR:")
        print(type(e).__name__, e)

        return False

    print()
    print("SR shape:", sr_output.shape)

    print(
        "SR range:",
        float(sr_output.min()),
        "to",
        float(sr_output.max()),
    )

    print(
        "Uncertainty shape:",
        uncertainty_raw.shape,
    )

    sr_rgb = np.asarray(
        sr_output[:, :, :3]
    )

    sr_rgb = np.clip(
        sr_rgb,
        0,
        1,
    )

    # --------------------------------------------------------
    # 4. LR CONSISTENCY
    # --------------------------------------------------------

    print()
    print("Calculating LR consistency...")

    try:

        # IMPORTANT:
        # function returns TWO maps
        error_lr, error_sr = lr_consistency_error(
            lr_rgb,
            sr_rgb,
            blur_sigma=BLUR_SIGMA,
        )

    except Exception as e:

        print()
        print("CONSISTENCY ERROR:")
        print(type(e).__name__, e)

        return False

    print(
        "LR error shape:",
        error_lr.shape,
    )

    print(
        "SR error shape:",
        error_sr.shape,
    )

    # --------------------------------------------------------
    # 5. UNCERTAINTY
    # --------------------------------------------------------

    print()
    print("Preparing uncertainty...")

    try:

        uncertainty = prepare_uncertainty_map(
            uncertainty_raw,
            sr_rgb.shape[:2],
        )

    except Exception as e:

        print()
        print("UNCERTAINTY ERROR:")
        print(type(e).__name__, e)

        return False

    print(
        "Prepared uncertainty:",
        uncertainty.shape,
    )

    # --------------------------------------------------------
    # 6. RISK MASK
    # --------------------------------------------------------

    print()
    print("Creating hallucination-risk mask...")

    try:

        (
            risk_mask,
            error_threshold,
            uncertainty_threshold,
        ) = create_hallucination_risk_mask(
            error_sr,
            uncertainty,
            percentile=RISK_PERCENTILE,
        )

    except Exception as e:

        print()
        print("RISK MASK ERROR:")
        print(type(e).__name__, e)

        return False

    # --------------------------------------------------------
    # 7. STATISTICS
    # --------------------------------------------------------

    residual_mean = float(
        np.mean(error_sr)
    )

    residual_max = float(
        np.max(error_sr)
    )

    residual_p95 = float(
        np.percentile(error_sr, 95)
    )

    uncertainty_mean = float(
        np.mean(uncertainty)
    )

    uncertainty_max = float(
        np.max(uncertainty)
    )

    uncertainty_p95 = float(
        np.percentile(
            uncertainty,
            95,
        )
    )

    risk_pixels = int(
        np.sum(risk_mask)
    )

    total_pixels = int(
        risk_mask.size
    )

    risk_percentage = (
        risk_pixels /
        total_pixels *
        100
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"SR shape              : {sr_rgb.shape}"
    )

    print(
        f"LR consistency mean   : {residual_mean:.6f}"
    )

    print(
        f"LR consistency max    : {residual_max:.6f}"
    )

    print(
        f"LR consistency p95    : {residual_p95:.6f}"
    )

    print(
        f"Uncertainty mean      : {uncertainty_mean:.6f}"
    )

    print(
        f"Uncertainty max       : {uncertainty_max:.6f}"
    )

    print(
        f"Uncertainty p95      : {uncertainty_p95:.6f}"
    )

    print(
        f"Residual threshold    : {error_threshold:.6f}"
    )

    print(
        f"Uncertainty threshold : "
        f"{uncertainty_threshold:.6f}"
    )

    print(
        f"High-risk pixels      : "
        f"{risk_pixels:,}"
    )

    print(
        f"Total pixels          : "
        f"{total_pixels:,}"
    )

    print(
        f"High-risk percentage  : "
        f"{risk_percentage:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_DIR / tile_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving results to:")
    print(output_dir)

    save_rgb(
        lr_rgb,
        output_dir / "lr_rgb.png",
    )

    save_rgb(
        sr_rgb,
        output_dir / "sr_rgb.png",
    )

    save_map(
        error_lr,
        output_dir / "consistency_error_lr.png",
    )

    save_map(
        error_sr,
        output_dir / "consistency_error_sr.png",
    )

    save_map(
        uncertainty,
        output_dir / "uncertainty.png",
    )

    save_mask(
        risk_mask,
        output_dir / "hallucination_risk_mask.png",
    )

    save_overlay(
        sr_rgb,
        risk_mask,
        output_dir / "hallucination_risk_overlay.png",
    )

    # Raw arrays
    np.save(
        output_dir / "consistency_error_lr.npy",
        error_lr,
    )

    np.save(
        output_dir / "consistency_error_sr.npy",
        error_sr,
    )

    np.save(
        output_dir / "uncertainty.npy",
        uncertainty,
    )

    np.save(
        output_dir / "risk_mask.npy",
        risk_mask,
    )

    # Statistics
    with open(
        output_dir / "statistics.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"Tile: {tile_name}\n"
        )

        f.write(
            f"SR shape: {sr_rgb.shape}\n"
        )

        f.write(
            f"LR consistency mean: "
            f"{residual_mean:.6f}\n"
        )

        f.write(
            f"LR consistency max: "
            f"{residual_max:.6f}\n"
        )

        f.write(
            f"LR consistency p95: "
            f"{residual_p95:.6f}\n"
        )

        f.write(
            f"Uncertainty mean: "
            f"{uncertainty_mean:.6f}\n"
        )

        f.write(
            f"Uncertainty max: "
            f"{uncertainty_max:.6f}\n"
        )

        f.write(
            f"Uncertainty p95: "
            f"{uncertainty_p95:.6f}\n"
        )

        f.write(
            f"Residual threshold: "
            f"{error_threshold:.6f}\n"
        )

        f.write(
            f"Uncertainty threshold: "
            f"{uncertainty_threshold:.6f}\n"
        )

        f.write(
            f"High-risk pixels: "
            f"{risk_pixels}\n"
        )

        f.write(
            f"Total pixels: "
            f"{total_pixels}\n"
        )

        f.write(
            f"High-risk percentage: "
            f"{risk_percentage:.2f}%\n"
        )

    # --------------------------------------------------------
    # 9. DIAGNOSTIC FIGURE
    # --------------------------------------------------------

    print("Creating diagnostic figure...")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )

    axes[0, 0].imshow(
        np.clip(lr_rgb, 0, 1)
    )

    axes[0, 0].set_title(
        "Sentinel-2 LR RGB"
    )

    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        np.clip(sr_rgb, 0, 1)
    )

    axes[0, 1].set_title(
        "GeoSharp Fine-Tuned 2.5m"
    )

    axes[0, 1].axis("off")

    axes[0, 2].imshow(
        percentile_normalize(
            error_sr,
            1,
            99,
        ),
        cmap="gray",
    )

    axes[0, 2].set_title(
        "LR Consistency Error"
    )

    axes[0, 2].axis("off")

    axes[1, 0].imshow(
        percentile_normalize(
            uncertainty,
            1,
            99,
        ),
        cmap="gray",
    )

    axes[1, 0].set_title(
        "Diffusion Uncertainty"
    )

    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        risk_mask,
        cmap="gray",
    )

    axes[1, 1].set_title(
        f"Risk Mask ({risk_percentage:.2f}%)"
    )

    axes[1, 1].axis("off")

    axes[1, 2].imshow(
        np.clip(sr_rgb, 0, 1)
    )

    axes[1, 2].imshow(
        risk_mask,
        alpha=0.45,
        cmap="Reds",
    )

    axes[1, 2].set_title(
        "Risk Overlay"
    )

    axes[1, 2].axis("off")

    plt.suptitle(
        f"GeoSharp Diagnostic — {tile_name}",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "diagnostic_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("DONE:", tile_name)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GEOSHARP — INDIAN SENTINEL-2 TEST (FINETUNED MODEL)")
    print("=" * 70)

    # Load our fine-tuned checkpoint BEFORE calling run_sr().
    load_finetuned_model()

    zip_files = find_tiles()

    if not zip_files:

        print()
        print("Put your Sentinel-2 ZIP files anywhere inside:")
        print(PROJECT_ROOT)

        return

    successful = 0
    failed = 0

    for zip_file in zip_files:

        try:

            result = process_tile(
                zip_file
            )

            if result:
                successful += 1
            else:
                failed += 1

        except KeyboardInterrupt:

            print()
            print("Stopped by user.")
            break

        except Exception as e:

            failed += 1

            print()
            print("=" * 70)
            print("UNEXPECTED ERROR")
            print("=" * 70)
            print(
                zip_file
            )
            print(
                type(e).__name__,
                e,
            )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        "ZIP files found :",
        len(zip_files),
    )

    print(
        "Successful      :",
        successful,
    )

    print(
        "Failed          :",
        failed,
    )

    print()
    print(
        "Outputs:",
        OUTPUT_DIR,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()  