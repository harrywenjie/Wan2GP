# AMD / ROCm Notes

Wan2GP does not ship a turnkey AMD build. If you plan to run the headless CLI on RDNA3-class GPUs or recent Ryzen APUs, you must provision the toolchain manually and keep it pinned for reproducibility.

## Recommended Environment
- **Python**: 3.11.x has the widest third-party wheel coverage for ROCm on Windows; Linux users can choose between 3.10 and 3.11 depending on their PyTorch build.
- **PyTorch**: Install a ROCm-enabled wheel that matches your GPU architecture (`gfx110x`, `gfx1151`, `gfx1201`, etc.). Community builds such as [scottt/rocm-TheRock](https://github.com/scottt/rocm-TheRock/releases) provide precompiled binaries for Windows.
- **Drivers**: Keep AMD Adrenalin / ROCm drivers aligned with the PyTorch wheel requirements. Mixing major versions usually breaks kernel loading.

## Setup Outline (Windows Example)
```cmd
:: Clone the repository
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP

:: Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

:: Install ROCm-enabled PyTorch wheels (adjust URLs for your Python version)
pip install ^
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch-gfx110x/torch-2.7.0a0+rocm_git3f903c3-cp311-cp311-win_amd64.whl ^
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch-gfx110x/torchaudio-2.7.0a0+52638ef-cp311-cp311-win_amd64.whl ^
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch-gfx110x/torchvision-0.22.0+9eb57cd-cp311-cp311-win_amd64.whl

:: Install core dependencies
pip install -r requirements.txt
```

Linux users should follow the ROCm installation guide from AMD, ensure `/opt/rocm` is on `PATH`, and then install the corresponding PyTorch wheels (or build from source).

## Validating the Toolchain
Before attempting a full render, run a dry-run to confirm that dependencies resolve and the GPU kernels initialise:

```cmd
python -m cli.generate --prompt "amd validation" --model-type t2v --dry-run --log-level DEBUG
```

If kernels fail to compile, recheck the PyTorch/driver pairing. Keep a record of the exact wheel filenames and driver versions so you can rebuild the environment deterministically.

## Operational Tips
- Prefer reproducible virtual environments per project; avoid mixing ROCm wheels across Python versions inside the same env.
- Some advanced accelerators (FlashAttention, SageAttention) do not have ROCm builds yet. Stick with the default attention kernels exposed by your PyTorch install.
- When you need to gather VRAM usage or performance metrics, run the CLI under `--dry-run` first to validate inputs, then launch the real generation while logging `nvidia-smi`/`rocm-smi` in parallel.

Future AMD contributions should document additional working wheel sources or patches in `PROJECT_PLAN_LIVE.md` so the headless workflow remains traceable.
