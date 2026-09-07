# GeoSharp — SIH 2026 Prototype

AI super-resolution of Sentinel-2 satellite imagery (10m -> sub-4m), with a
per-pixel confidence/uncertainty map showing reconstructed vs. observed detail.

**This prototype uses a pretrained model (LDSR-S2) for inference only — no
training.** The full custom hybrid CNN-Transformer architecture is the
proposed Phase 2 build if selected.

## Setup

```bash
git clone <this repo>
cd GeoSharp_prototype
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**If you have an RTX 50-series GPU (Blackwell, e.g. RTX 5060/5060 Ti/5070/5080/5090):**
install torch from the CUDA 12.8 wheel index *first*, before the rest of the
requirements, or you'll hit a `"CUDA capability sm_120 is not compatible"`
error:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.randn(1).cuda())"  # should NOT error
```

Then install everything else:

```bash
pip install -r requirements.txt
```

No local GPU? Use the official `opensr-model` Google Colab notebooks
instead (free GPU, no local install needed).

## Model / checkpoint

The app uses the GeoSharp fine-tuned LDSR-S2 checkpoint hosted on Hugging Face:

`kapoorraaghav/geosharp-ldsrs2-worldstrat-x4`

The first run downloads the approximately 2 GB checkpoint into the local Hugging Face cache. No Hugging Face token is required for this public repository.

The LDSR-S2 architecture configuration is pinned locally at `configs/config_10m.yaml`, matching the OpenSR 10m configuration supplied for this build.

The project keeps the same preprocessing and diagnostic stages: B02/B03/B04/B08 input, centered 128×128 patch, 4× reconstruction to 512×512, stochastic uncertainty, LR-consistency residual, and the conservative hallucination-risk intersection.

## API keys / credentials

opensr-model and the Hugging Face model do not require an application API key for this public-inference path. Copernicus credentials are only needed by the optional tile-download scripts.

## How to run

**1. Test your inference setup alone** (once opensr-model is installed):
```bash
python pipeline/inference.py data/raw/your_tile.tif
```
This prints the output/uncertainty shapes so you can confirm the model
runs before touching the UI. With `USE_PLACEHOLDER = True` (default in
`inference.py`), this works even without opensr-model installed, using
fake data.

**2. Run the full UI:**
```bash
streamlit run app/main.py
```
This opens a browser tab (usually `http://localhost:8501`) where you can
upload a tile and see original / sharpened / uncertainty side-by-side.

**3. Fine-tuned model loading:**
The default `pipeline/inference.py` loads the ESA LDSR-S2 architecture, then overlays the GeoSharp fine-tuned `ldsrs2_worldstrat_x4_best.pt` checkpoint from Hugging Face. The model is cached for the lifetime of the Streamlit process.

## Repo structure

```
pipeline/
  download.py     - pull sample Sentinel-2 tiles from Copernicus Data Space
  preprocess.py   - band selection + normalization
  inference.py    - run_sr(tile_path) -> (output_image, uncertainty_map)
  metrics.py      - PSNR/SSIM vs bicubic baseline
app/
  main.py         - Streamlit UI (upload -> original/sharpened/uncertainty)
data/raw/         - downloaded sample tiles
data/processed/   - preprocessed tiles
outputs/          - saved before/after results for demo & PPT
```

## Shared interface contract

Everything hangs off one function in `pipeline/inference.py`:

```python
def run_sr(tile_path: str) -> (output_image, uncertainty_map):
    ...
```

`USE_PLACEHOLDER = True` at the top of `inference.py` returns fake data so
the UI can be built and tested independently of the real model integration.
Flip it to `False` once `opensr-model` is confirmed working.

## Team split

- **GPU / model integration**: `pipeline/inference.py` (real opensr-model wiring)
- **UI + data**: `app/main.py`, `pipeline/download.py`, `pipeline/metrics.py`

Work on your own branch (e.g. `friend-inference`, `your-ui`), merge into
`main` once your part works against the real (non-placeholder) function.

## Running the UI

```bash
streamlit run app/main.py
```

## Judges' Q&A prep

**"Did you train this yourselves?"**
No — the prototype runs on a pretrained model (LDSR-S2, IEEE JSTARS 2025)
for inference. Our contribution is the end-to-end applied pipeline:
Sentinel-2 ingestion, evaluation harness, and a usable interface for
disaster-response/agricultural use cases. The full custom hybrid
CNN-Transformer architecture is the proposed Phase 2 build.

**"Where's your ground truth?"**
No native <4m Sentinel-2 ground truth exists — that's the core problem
this whole space addresses. For the prototype we compare against a bicubic
upsampling baseline to show the model adds real information beyond naive
interpolation. For Phase 2 training, Cartosat-3 (ISRO/NRSC Bhuvan) is the
planned high-res reference for building training pairs.

## Important resolution note

This checkpoint/configuration is an x4 super-resolution model: 128×128 LR → 512×512 SR, corresponding to 10m → 2.5m imagery. The project code and checkpoint metadata support 4×, not 5×.
