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

## API keys / credentials

**opensr-model needs NO API key.** It's a local pretrained model —
`model.load_pretrained()` auto-downloads the weights from HuggingFace the
first time you run it. Just need internet access once.

**Copernicus Data Space (only if scripting tile downloads) needs a
username/password**, not an API key:

1. Create a free account at https://dataspace.copernicus.eu/
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Fill in your real credentials in `.env`:
   ```
   COPERNICUS_USERNAME=your_email@example.com
   COPERNICUS_PASSWORD=your_password
   ```
4. Never commit `.env` — it's already in `.gitignore`.

If you'd rather skip scripting entirely, just download 1-3 tiles manually
from https://browser.dataspace.copernicus.eu/ and drop them in
`data/raw/` — no credentials needed in code at all for that path.

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

**3. Switch from placeholder to real model:**
Open `pipeline/inference.py`, change:
```python
USE_PLACEHOLDER = True
```
to
```python
USE_PLACEHOLDER = False
```
Nothing else needs to change — the UI already calls `run_sr()` and
doesn't care which implementation runs underneath.

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
