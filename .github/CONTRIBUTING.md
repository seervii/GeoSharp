# Contributing to GeoSharp

Thank you for considering a contribution to GeoSharp. These guidelines keep pull requests moving quickly and respect everyone's time.

## Project scope

GeoSharp is a 4× super-resolution tool for Sentinel-2 imagery (B02/B03/B04/B08) with per-pixel uncertainty output. The current code is **Phase 1**. The following are **Phase 2** and are proposals, not current features:

- the full 12-band pipeline,
- the hybrid CNN-Transformer architecture,
- training from scratch or changes to the fine-tuning setup.

Please open an issue and wait for a reply before starting on Phase 2 work.

## Code of conduct

By participating, you agree to keep code reviews, issues and pull request discussions respectful, constructive and harassment-free.

## How can I contribute?

### Report a bug

Check the existing [issues](https://github.com/seervii/GeoSharp/issues) first. If it hasn't been reported, open a new issue with:

- clear steps to reproduce it,
- expected versus actual behaviour,
- input details (tile name or size, bands, format) and any parameters used,
- your environment: Python version, PyTorch version, OS, and GPU/VRAM (or "CPU only" / "Colab").

### Suggest an enhancement

Open an issue before you start building. Explain the motivation (for example memory use, faster inference, UI improvements) and show an example of the expected result.

### Find something to work on

Look for issues labelled `good first issue` or `help wanted`. Larger ideas are labelled `phase-2`.

## Development workflow

### Prerequisites

- Python 3.9 or newer
- Git
- A CUDA-capable GPU is recommended for running LDSR inference locally. If you don't have one, use the official `opensr-model` Google Colab notebooks (free GPU). Users with an RTX 50-series GPU should follow the extra torch install step in the [README](README.md#quick-start).

### 1. Fork and clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/GeoSharp.git
cd GeoSharp
```

### 2. Add the upstream remote

This lets you pull new changes from the main repository:

```bash
git remote add upstream https://github.com/seervii/GeoSharp.git
```

### 3. Create a feature branch

Always branch from an up-to-date `main`:

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

### 4. Set up a virtual environment

```bash
python3 -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate
# Activate on Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 5. Run the app locally

```bash
streamlit run app/main.py
```

On first run the app downloads the fine-tuned checkpoint from Hugging Face (about 2 GB) into your local cache. This is expected. Do not add it to the repository.

### 6. Make and test your changes

- Implement your change or fix.
- Run an inference pass on a sample tile to confirm the pipeline still works, for example `python pipeline/inference.py data/raw/your_tile.tif`.
- Run the tests in the `test/` folder (see below) and add a test or a small reproducible example for new behaviour.
- Format your code to PEP 8.

### 7. Commit and push

Write short, descriptive commit messages:

```bash
git add <files you changed>
git commit -m "feat: add image tiling option to prevent CUDA OOM"
git push -u origin feature/your-feature-name
```

Use `git add <files>` rather than `git add .` so you don't commit data or credentials by accident.

### 8. Open a pull request

- Open a pull request targeting `main`.
- Reference the issue it addresses (for example `Closes #2`).
- Describe what you changed and how you tested it. Screenshots help for UI changes.
- If changes are requested, push new commits to the same branch and the PR updates automatically.

## Tests

The `test/` folder contains the project's tests. Run them from the repository root before opening a PR:

```bash
python -m pytest test/
```

`make_test_tiff.py` creates a small test GeoTIFF, and `test_new_model.py` checks that the model loads and runs. New pipeline code (preprocessing, metrics, inference helpers) should come with a test or a short example showing how to try it.

## Code style and standards

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Add clear docstrings and comments for tensor operations, image preprocessing and diffusion sampling parameters.
- Keep the shared interface stable: `run_sr(tile_path) -> (output_image, uncertainty_map)` in `pipeline/inference.py`. If you need to change it, open an issue first.

## Data, models and credentials

- **Do not commit** model weights (`.ckpt`, `.pth`, `.pt`, `.safetensors`), Sentinel-2 tiles or large output datasets. Make sure they are listed in `.gitignore`.
- **Never commit credentials.** Copernicus Data Space logins and any tokens belong in environment variables or an untracked `.env` file, not in the code.
- Keep `data/raw/` and `data/processed/` out of version control.

## Licensing and attribution

- By contributing, you agree that your contributions are released under the project's [MIT License](LICENSE).
- GeoSharp builds on ESA OpenSR (LDSR-S2) and was fine-tuned on a subset of the WorldStrat dataset. Keep those credits intact, and don't add code or data you don't have the right to share.
- The fine-tuned checkpoint follows the license terms of its training data and upstream model, which are not all MIT. See the License section of the [README](README.md#license).

## Questions

Open an issue with your question, or mention [@seervii](https://github.com/seervii) in a discussion on a relevant issue or pull request.
