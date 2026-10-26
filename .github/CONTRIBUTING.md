# Contributing to GeoSharp

First off, thank you for considering contributing to GeoSharp! Following these guidelines helps maintain project quality, keeps pull requests moving swiftly, and respects everyone's time.

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful, welcoming, and harassment-free environment for everyone. Please keep code reviews, issue discussions, and pull request conversations constructive and professional.

---

## How Can I Contribute?

### 1. Reporting Bugs
Before filing a bug report:
- Check existing Issues to ensure the bug has not already been reported.
- If not, open a new issue describing:
  - Clear steps to reproduce the issue.
  - Expected vs. actual behavior.
  - Input image details (resolution, format) and parameters used.
  - Your environment (Python version, PyTorch version, OS, and GPU/VRAM specs).

### 2. Suggesting Enhancements
Feature requests and model optimization ideas are welcome! Please open an issue to propose your idea before implementing it, explaining:
- The motivation behind the feature (e.g., memory optimization, faster inference, UI/CLI improvements).
- Example usage or expected output improvements.

### 3. Finding Things to Work On
Looking for where to start? Check issues tagged with `good first issue` or `help wanted`.

---

## Development Workflow

### Prerequisites
- [Python 3.9+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- A CUDA-capable GPU (recommended for running LDSR inference locally)

### Step-by-Step Guide for Contributors

#### 1. Fork & Clone
Fork the repository on GitHub, then clone your fork locally:
```bash
git clone [https://github.com/](https://github.com/)<your-username>/GeoSharp.git
cd GeoSharp
```

#### 2. Keep Your Fork Synced (Upstream)
Add the main repository as an upstream remote to stay up to date with new changes:
```bash
git remote add upstream [https://github.com/seervii/GeoSharp.git](https://github.com/seervii/GeoSharp.git)
```

#### 3. Create a Feature Branch
Always branch off an updated `main`:
```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

#### 4. Set Up a Virtual Environment & Install Dependencies
Create an isolated Python environment and install the required packages:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 5. Run the Application Locally
Launch the Streamlit interface to test your changes:
```bash
streamlit run app.py
```
- Implement your changes or bug fixes.
- Test your changes by running an inference pass on a sample low-resolution image to ensure the LDSR pipeline executes without errors.
- Format your Python code cleanly (following PEP 8 conventions).

#### 6. Commit & Push
Write concise, descriptive commit messages:
```bash
git add .
git commit -m "feat: add image tiling option to prevent CUDA OOM"
git push -u origin feature/your-feature-name
```

#### 7. Submit a Pull Request
- Go to GitHub and open a Pull Request targeting the `main` branch.
- Reference the issue being addressed in the PR description (e.g., `Closes #2`).
- If changes are requested during review, push new commits to the same branch—the PR updates automatically.

---

## Code Style & Standards

- Follow standard [PEP 8 Style Guidelines for Python Code](https://peps.python.org/pep-0008/).
- Add clear docstrings and comments explaining complex tensor operations, image pre-processing, or diffusion sampling parameters.
- Do not commit large model weight files (`.ckpt`, `.pth`, `.safetensors`) or large output datasets directly to the repository—ensure they are listed in `.gitignore`.
