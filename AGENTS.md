# Repository Guidelines

## Project Structure & Module Organization
WanGP centers on `wgp.py`, the Gradio app that orchestrates GPU profiles, model loading, and queue control. Cross-cutting helpers live in `shared/`; extend those modules rather than reimplementing utilities inside features. Model defaults and VRAM presets belong in `defaults/` and `profiles/`, while preprocessing and postprocessing stages stay in their respective folders or `extract_source_images.py`. Keep downloadable weights in `loras*/`, `finetunes/`, and `models/`, and document user-facing changes in `docs/` so the UI, CLI, and Docker notes stay aligned.

## Build, Test, and Development Commands
Create a reproducible environment before editing:
```bash
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
python wgp.py
```
`python wgp.py` serves the local UI; append `--share`, `--debug-gen-form`, or `--betatest` when you need remote demos, UI timing, or experimental toggles. Debian-based owners can run `./run-docker-cuda-deb.sh` for a reproducible CUDA + SageAttention stack that auto-detects VRAM and launches the containerized app.

## Coding Style & Naming Conventions
Write Python 3.10 code with four-space indents and Black-compatible formatting. Use snake_case for functions and modules, PascalCase for classes, and keep side effects behind `if __name__ == "__main__":` in any new CLIs. Co-locate constants with the existing `WanGP_version` block, and keep config keys (`plugins.json`, `defaults/*.json`) lowercase with hyphenated identifiers.

## Testing Guidelines
There is no automated test suite yet, so rely on scenario-driven validation. Run `python wgp.py --save-masks` or `--save-speakers` to capture intermediates when touching video or audio stages, and queue multiple jobs to exercise `shared/utils/process_locks.py`. Document GPU model, VRAM profile, prompt, and outputs in your PR so reviewers can replay regressions quickly.

## Commit & Pull Request Guidelines
History favors short, present-tense subjects (`added loras accelerators for Ovi`, `fixed locator`). Squash noisy checkpoints locally, mention the main modules touched in the body, and update `docs/CHANGELOG.md` whenever you alter UX or model coverage. PRs should outline intent, list required assets/weights, link issues, and attach screenshots or clips that show the behavior change.

## Security & Configuration Tips
Do not commit model checkpoints, API tokens, or user-generated media. Respect user-configurable paths such as `--lora-dir` / `--lora-dir-i2v` instead of hard-coding folders, and record new environment variables in `README.md` or `docs/INSTALLATION.md`. Follow `docs/PLUGINS.md` when publishing plugins and remind testers to use `--lock-config` or `--lock-model` whenever workflows should avoid mutating operator settings.
