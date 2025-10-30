# Getting Started (Headless CLI)

Wan2GP now runs entirely from the command line. Use this guide to execute your first generation and understand the basic workflow.

## 1. Prepare the Environment
Make sure the project virtual environment is active and dependencies are installed (see `docs/INSTALLATION.md`). You should be inside the repository root.

Confirm everything is wired correctly:
```bash
python -m cli.generate --prompt "sanity check" --dry-run
```
The dry-run prints the resolved configuration and exits without rendering frames. Fix any reported path or dependency errors before moving on.

## 2. Run a Minimal Generation
```bash
python -m cli.generate \
  --prompt "a sunrise over an alien coastline, volumetric lighting" \
  --model-type t2v \
  --frames 48 \
  --guidance-scale 6.5 \
  --output-dir outputs/demo
```

During execution the CLI will stream status and progress messages to the terminal. On success the final video path is printed at the end of the run.

## 3. Working With Assets
- Store input images, masks, and audio on disk. Pass them via the appropriate flags (`--image-start`, `--video-mask`, `--audio-guide`, etc.).  
- You can repeat `--image-ref` to supply multiple reference images.  
- The CLI validates every file path before launching generation; missing or non-file paths cause an immediate, descriptive error.

## 4. Common Adjustments
- **Reproducibility:** Always set `--seed` when you need deterministic outputs. Use `--dry-run` to snapshot the merged configuration for logging.  
- **Duration:** Increase `--frames` for longer clips.  
- **Resolution:** Provide `--resolution WIDTHxHEIGHT` to override presets (e.g. `832x480`).  
- **Performance:** Choose model variants with lower VRAM footprints using `--model-type` (see `docs/MODELS.md`).

## 5. Inspecting Results
Outputs are saved under the configured directory (default from Wan2GP settings, or the value you passed to `--output-dir`). Track each experiment in `PROJECT_PLAN_LIVE.md`—record seed, timing, VRAM usage, and output paths so runs remain auditable.

## Next Steps
- Review `docs/CLI.md` for every available flag.  
- Explore `docs/MODELS.md` to understand model families and requirements.  
- Consult `docs/TROUBLESHOOTING.md` if the CLI raises errors or you encounter GPU limitations.
