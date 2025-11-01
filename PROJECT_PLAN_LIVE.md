# Project Plan Live

## Objective
- Transition **Wan2GP** from a Gradio-centric application into a **lightweight, headless CLI tool** focused solely on core video-generation workflows.  
- All actions should move the codebase toward minimalism, reproducibility, and deterministic CLI operation. 
- Ensure all modules run cleanly **without** environment variables or assumptions from the legacy UI layer.
- When refactoring or removing code, preserve debuggability through logging.
- The headless build never exposes GUI-driven affordances — video/audio playback, galleries, or interactive widgets must be removed entirely; only on-disk artifacts remain supported.

---

## Project Roadmap

**Milestone 1 – Eliminate residual GUI surface (Active)**
- [In Progress] Audit and remove remaining UI artifacts (legacy screenshots, theme toggles, stray assets).
- [Pending] Decide whether any gallery/media normalisation helpers should be salvaged into CLI utilities or deleted. (Prior plugin-removal phases are archived in `docs/WORK_HISTORY.md`.)

**Milestone 2 – Finish Production Manager extraction (Active)**
- [Completed] Replace `wgp.generate_video` persistence paths with `MediaPersistenceContext` helpers supplied by `ProductionManager`.
- [Planned] Lift prompt-enhancer and LoRA orchestration into dedicated adapters so CLI flows depend on `ProductionManager` instead of `wgp`. Dependency details live in `docs/CONTEXT.md` (“ProductionManager Dependency Snapshot”).
- [Planned] Continue peeling residual runtime globals (model load/release, queue callbacks) listed in `docs/CONTEXT.md`.

**Milestone 3 – Harden CLI orchestration (Active)**
- [In Progress] Keep queue management under `cli/` while retiring legacy shims; move any remaining helpers out of `core/`.
- [Planned] Finish disk-first workflows for mask/voice pipelines and document validation expectations for each CLI entrypoint.
- [Planned] Emit a machine-readable artifact manifest from `cli.generate` capturing saved paths, metadata mode, and adapter payload hashes.

**Milestone 4 – Prune GUI-dependent tooling (Ongoing)**
- [In Progress] Audit preprocessing utilities for GUI assumptions and refactor or remove as needed.
- [Pending] Gate or retire models that still require interactive inputs.
- [Planned] Extract the prompt enhancer stack into a shared module and stand up a provider abstraction for local/cloud backends.
- [Completed] Thread `MediaPersistenceContext` through MatAnyOne so preprocessing mask/audio exports mirror the headless generation pipeline (2025-11-01).

**Milestone 5 – Retire ancillary runtimes (Pending)**
- [Pending] Remove Docker scripts and legacy launch docs once CLI parity is confirmed.
- [Pending] Audit Pinokio/one-click installers and drop any UI-only tooling.

**Milestone 6 – Documentation upkeep (Ongoing)**
- [In Progress] Keep this plan, `docs/CLI.md`, and `docs/APPENDIX_HEADLESS.md` aligned with the headless workflows.

**Deferred**
- After the headless milestones land, reorganise `models/`, `preprocessing/`, `postprocessing/`, and `shared/` under a Production Manager–centric layout (tracked in `docs/CONTEXT.md`).

---

## Validation Expectations
- For CLI regression checks, run `python -m cli.generate --prompt "smoke test prompt" --dry-run --log-level INFO` to confirm argument parsing, logging, and runtime overrides.
- Exercise preprocessing changes with `python -m cli.matanyone --input <video> --template-mask <mask> --dry-run` (or a short real run when GPU time permits).
- When MatAnyOne emits artifacts, inspect `<output_dir>/manifests/run_history.jsonl` (or the `--manifest-path` override) and confirm entries include `mask_foreground`, `mask_alpha`, and optional `rgba_archive` roles.
- Record timing, VRAM usage, and output artifact paths in `docs/WORK_HISTORY.md` whenever a command executes generation or preprocessing work.
- Keep any ad-hoc validation scripts lightweight and remove them once automated coverage exists.

---

## Immediate Next Actions
- Finish retiring `shared.utils.audio_video.save_*` adapters by migrating remaining consumers onto `core.io.media` helpers.
  - **Proposal (2025-11-02)**: Use `rg` to enumerate the outstanding adapters (audio/video/image) under `shared/utils` and refactor each caller to request a `MediaPersistenceContext` or call `write_*` directly.
  - **Rationale**: Completing the migration collapses persistence onto one implementation, simplifies logging/retry behaviour, and clears the final GUI-era shims.
  - **Design (2025-11-02)**: Stage conversions module-by-module, replacing adapter imports with context usage, add focused tests where coverage is missing, and delete the adapter functions once call sites are gone.
- Add an integration smoke test for `cli.matanyone` that exercises manifest emission end-to-end.
  - **Proposal (2025-11-02)**: Invoke the CLI against a tiny fixture (dry-run plus a short real run when possible) and assert the JSONL manifest contains `mask_foreground`, `mask_alpha`, and optional `rgba_archive` entries.
  - **Rationale**: Guards the new manifest pathway against regressions in CLI argument parsing, context wrapping, and recorder plumbing.
  - **Design (2025-11-02)**: Use a temporary output directory, run with `--dry-run` to confirm no file writes, then trigger a minimal propagation mocked via dependency injection and inspect the emitted JSONL before teardown.
  
