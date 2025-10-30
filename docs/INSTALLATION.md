# Installation Guide (Headless Build)

Wan2GP now targets a direct Python workflow. The legacy Docker images, Pinokio bundles, and Gradio launch scripts are no longer maintained. Follow the steps below to prepare a local environment that can execute the CLI entry point.

## Prerequisites
- **Python**: 3.10.x (preferred) or 3.11.x with a matching toolchain (venv, pip).  
- **GPU**: CUDA- or ROCm-capable device with drivers configured by the operating system.  
- **Git**: for cloning and keeping the repository up to date.  
- **PyTorch**: install a build that matches your GPU stack (CUDA/ROCm). Refer to the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the correct command.

> The project assumes the operator provisions a virtual environment and installs dependencies before running any CLI command. If you encounter missing packages, resolve them in the environment rather than modifying project files.

## Step-by-Step Setup
```bash
# 1. Clone the repository
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP

# 2. Create and activate a virtual environment (example using venv)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install PyTorch that matches your GPU/driver combination
# (See https://pytorch.org/get-started/locally/ for the correct wheel)
pip install torch torchvision torchaudio --index-url <appropriate-wheel-index>

# 4. Install Wan2GP Python dependencies
pip install -r requirements.txt
```

## Verify the Installation
Run a dry-run generation to confirm that `cli.generate` resolves presets, discovers media paths, and writes logs:

```bash
python -m cli.generate --prompt "installation test" --dry-run
```

You should see a resolved configuration printed to the terminal without errors. If the command fails because of missing libraries, install them in the active virtual environment and rerun the dry-run.

## Next Steps
- Review `docs/CLI.md` for the full flag surface.  
- Consult `docs/MODELS.md` to understand available `--model-type` values and VRAM requirements.  
- For AMD/ROCm-specific notes, see `docs/AMD-INSTALLATION.md`.

Keep the environment reproducible: document driver versions, PyTorch wheels, and any custom patches in your own operation notes so future runs remain deterministic.
