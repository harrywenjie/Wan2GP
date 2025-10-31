# Troubleshooting (CLI)

This guide captures the most common issues encountered while running `python -m cli.generate` and offers practical fixes aligned with the headless workflow.

## General Checklist
1. Activate the project virtual environment (`.venv`, `conda`, etc.).
2. Ensure PyTorch detects your GPU (`python -c "import torch; print(torch.cuda.is_available())"` or ROCm equivalent).
3. Run `python -m cli.generate --prompt "probe" --dry-run` to confirm presets load and media paths resolve.

## CLI Errors
| Symptom | Cause | Resolution |
| --- | --- | --- |
| `Invalid file inputs` | One of the supplied paths does not exist or is a directory | Verify every `--image-*`, `--video-*`, and `--audio-*` flag points to a real file. |
| `Unknown model type` | Value passed to `--model-type` not recognised by Wan2GP presets | Check the valid identifiers in `docs/MODELS.md` or list `defaults/*.json`. |
| `ModuleNotFoundError` | Dependency missing from the virtual environment | Install the missing package within the env (`pip install <package>`), then rerun the CLI. |
| `torch.cuda.OutOfMemoryError` | GPU VRAM exhausted mid-run | Switch to a lighter model (`--model-type t2v_1_3b`), lower `--frames`, or reduce resolution. |
| `RuntimeError: device not found` | PyTorch cannot access the GPU | Confirm drivers are loaded and that the correct PyTorch build (CUDA/ROCm) is installed. |

## Managing VRAM
- Use smaller model variants (`--model-type t2v_1_3b`, `i2v_1_3b`) when experimenting.  
- Shorten clips: lower `--frames` or `--resolution`.  
- When resuming a pipeline with large assets, restart the Python process between runs to release memory fragmentation.

## Performance Tips
- Keep `--dry-run` in your workflow to validate arguments without consuming GPU time.  
- Use dedicated output directories via `--output-dir` to avoid scanning large folders on subsequent runs.  
- Monitor GPU utilisation with `nvidia-smi` or `rocm-smi` in a separate terminal; capture snapshots for the project log.

## Reproducibility & Logging
- Always record the full command line (including `--seed`) alongside the resulting artifact path.  
- For deeper diagnostics, run with `--log-level DEBUG` to expose queue and progress callbacks.  
- Store observations (timings, VRAM load, anomalies) under `# Work History` in `docs/WORK_HISTORY.md`.

## When Things Still Fail
- Compare your environment (Python, PyTorch, driver versions) against a known-good setup.  
- Reinstall PyTorch with the matching CUDA/ROCm wheel if kernels fail to launch.  
- File unresolved questions under `## Immediate Next Actions` in `PROJECT_PLAN_LIVE.md` so future contributors can investigate.
