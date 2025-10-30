# Wan2GP (Headless CLI)

Wan2GP is being refactored into a lean, scriptable video generator. The legacy Gradio UI, plugin framework, and Docker workflows have been removed. The supported path today is a straightforward command line interface that wraps the existing generation pipelines while keeping reproducibility and deterministic outputs front and center.

## Current Status
- **UI Removed:** `wgp.py` no longer launches a browser interface. All interaction happens through the CLI.
- **Plugin System Retired:** plugin hooks, JSON manifests, and documentation were deleted. Extend Wan2GP by editing the CLI pipelines or writing standalone scripts.
- **Docker Workflow Dropped:** run Wan2GP directly inside your Python environment; container helpers are no longer maintained.

## Environment Expectations
- Use the project `venv` prepared by the operator. The CLI assumes dependencies are already installed.
- GPU support (CUDA, ROCm) depends on the underlying environment exactly as before; the CLI does not perform driver checks for you.

## Basic CLI Usage
```bash
python -m cli.generate \
  --prompt "sunset over the coast, cinematic drone shot" \
  --model-type t2v \
  --dry-run
```

Running without `--dry-run` will execute the generation pipeline and write the result under Wan2GP's configured output directory (or a path supplied via `--output-dir`).

All available options are documented in `docs/CLI.md`. Highlights include:
- `--negative-prompt`, `--guidance-scale`, `--steps`, `--frames` to tune sampling.
- Path-based inputs (`--image-start`, `--video-source`, `--audio-guide`, …) for image/video/audio-assisted workflows.
- `--log-level` to control verbosity and `--dry-run` to inspect the resolved configuration without generating frames.

## Recommended Workflow
1. Prepare prompts and any reference assets on disk.
2. Run `python -m cli.generate ... --dry-run` to confirm parsing and file discovery.
3. Launch the actual generation.
4. Capture timings, VRAM information, and output paths in `PROJECT_PLAN_LIVE.md` as part of the ongoing headless migration.

## Documentation
- `docs/CLI.md` — full argument reference and usage patterns.
- `PROJECT_PLAN_LIVE.md` — active migration log, open questions, and validation history.

Historical screenshots, plugin guides, and Docker instructions have been removed because they no longer match the headless build.
