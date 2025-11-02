# Project Plan Live

## Date
2025-11-02

---

## Objective
- Transform **Wan2GP** from a GUI-based app into a **minimal, headless CLI framework** dedicated to core video-generation workflows.
- Prioritize **determinism, reproducibility**, and **modular clarity** over feature breadth.
- Eliminate all dependencies on legacy UI layers or environment-specific assumptions.
- Maintain transparency and debuggability through structured logging and clear CLI feedback.
- Support only **on-disk artifacts** as outputs — no interactive or visual interfaces.

---

## Project Roadmap

**Milestone 1 – Eliminate residual GUI surface (Active)**
- [In Progress] Audit and remove remaining UI artifacts (legacy screenshots, theme toggles, stray assets).
- [Pending] Decide whether any gallery/media normalisation helpers should be salvaged into CLI utilities or deleted. (Prior plugin-removal phases are archived in `SESSION_LOG.md`.)

**Milestone 2 – Finish Production Manager extraction (Active)**
- [Completed] Replace `wgp.generate_video` persistence paths with `MediaPersistenceContext` helpers supplied by `ProductionManager`.
- [Completed] Retired `shared.utils.audio_video.save_*` shims after migrating remaining callers onto `core.io.media` (2025-11-02).
- [Planned] Lift prompt-enhancer and LoRA orchestration into dedicated adapters so CLI flows depend on `ProductionManager` instead of `wgp`. Dependency details live in `LIVE_CONTEXT.md` (“ProductionManager Dependency Snapshot”).
- [Planned] Continue peeling residual runtime globals (model load/release, queue callbacks) listed in `LIVE_CONTEXT.md`.

**Milestone 3 – Harden CLI orchestration (Active)**
- [Completed] Wrapped `preprocessing/dwpose.save_one_video` around `core.io.media.write_video`, retiring its direct `imageio` dependency (2025-11-02).
- [In Progress] Keep queue management under `cli/` while retiring legacy shims; move any remaining helpers out of `core/`.
- [Completed] Added MatAnyOne CLI manifest integration smoke coverage verifying `mask_foreground`, `mask_alpha`, and `rgba_archive` roles (2025-11-02).
- [Completed] Persisted MatAnyOne audio tracks through `MediaPersistenceContext.save_audio`, wiring decoded artifacts into manifests and metadata sidecars (2025-11-02).
- [Completed] Surfaced MatAnyOne audio artifact metadata (sample rate, duration, language, channels) through CLI logging and manifest entries for downstream automation (2025-11-02).
- [Completed] Queue summaries and control-server status payloads now expose MatAnyOne audio metadata (`audio_tracks`) for orchestration clients (2025-11-02).
- [Completed] `cli.queue_controller_smoke` now seeds audio metadata and asserts the TCP `status` payload plus queue summary reflect `audio_tracks` fields end-to-end (2025-11-02).
- [Completed] Dwpose debug mask exports now route through `MediaPersistenceContext.save_video`, so CLI persistence honours shared codec/container defaults and manifest logging (2025-11-02).
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
- After the headless milestones land, reorganise `models/`, `preprocessing/`, `postprocessing/`, and `shared/` under a Production Manager–centric layout (tracked in `LIVE_CONTEXT.md`).

---

## Validation Expectations
- For CLI regression checks, run `python -m cli.generate --prompt "smoke test prompt" --dry-run --log-level INFO` to confirm argument parsing, logging, and runtime overrides.
- Run `python -m unittest tests.test_matanyone_cli_integration` to ensure MatAnyOne manifest emission stays in sync with the CLI surface.
- Exercise preprocessing changes with `python -m cli.matanyone --input <video> --template-mask <mask> --dry-run` (or a short real run when GPU time permits).
- When MatAnyOne emits artifacts, inspect `<output_dir>/manifests/run_history.jsonl` (or the `--manifest-path` override) and confirm entries include `mask_foreground`, `mask_alpha`, optional `rgba_archive`, and `audio` roles with the expected metadata sidecars, plus audio fields for `sample_rate`, `duration_s`, `language`, and `channels`.
- Record timing, VRAM usage, and output artifact paths in `SESSION_LOG.md` whenever a command executes generation or preprocessing work.
- Keep any ad-hoc validation scripts lightweight and remove them once automated coverage exists.

---

## Project Specific Docs

- `README.md` – Primary project overview. 
- `docs/APPENDIX_HEADLESS.md` - Architectural appendix for the headless CLI migration, queue orchestration, and persistence references.
- `docs/CLI.md` - CLI usage guide covering generation/preprocessing flags, runtime toggles, and the artifact manifest schema.
- `docs/AMD-INSTALLATION.md` – ROCm/AMD provisioning notes for RDNA-class deployments.
- `docs/CHANGELOG.md` – Headless-era changelog capturing milestone refactors and documentation updates.
- `docs/FINETUNES.md` – Guidance for creating, validating, and distributing finetune JSON profiles.
- `docs/GETTING_STARTED.md` – Quickstart walkthrough for running your first CLI generation and common adjustments.
- `docs/INSTALLATION.md` – Platform-neutral setup guide for the headless Python workflow.
- `docs/IO_MEDIA_MIGRATION.md` – Media persistence refactor plan tracking `core.io.media` progress.
- `docs/LORAS.md` – LoRA asset organisation and activation notes for configuration-driven runs.
- `docs/MODELS.md` – `--model-type` catalogue detailing hardware footprints and use cases.
- `docs/TROUBLESHOOTING.md` – Headless CLI troubleshooting checklist with common error resolutions.
- `docs/VACE.md` – Motion-transfer/VACE workflow reference covering required assets and CLI usage.

---

## Immediate Next Actions
- Stress-test queue-control audio summaries with multi-track payloads to make sure textual output scales beyond single-track MatAnyOne runs.
  - **Proposal (2025-11-02)**: Extend the smoke harness fixtures (or add a dedicated test) to seed multiple audio entries with mixed metadata, then assert both the JSON payload and queue summary enumerate each track cleanly.
  - **Rationale**: Bilingual or commentary-dual exports depend on multiple tracks; coverage here prevents regressions as persistence adapters evolve.
  - **Design (2025-11-02)**: Generalise the stub manager’s audio injection helper so future tests can reuse it, and verify summary formatting via targeted string assertions before promoting the helper into reusable test utilities.
- Add regression coverage for Dwpose debug persistence so context-driven saves stay locked to Production Manager defaults.
  - **Proposal (2025-11-02)**: Patch `preprocess_video_with_mask` via dependency injection in a unit test to assert `save_one_video` receives `MediaPersistenceContext` configs when debug exports are enabled, and confirm filenames align with the legacy `masked_frames{n}.mp4` scheme.
  - **Rationale**: The new context threading replaces `imageio` writers; coverage ensures future refactors keep the debug path wired through manifest-aware persistence.
  - **Design (2025-11-02)**: Stub a minimal context with counters for `save_video` calls, simulate mask/video tensors, and validate both the frame/mask exports and config overrides (FPS, quality) before promoting the helper into reusable fixtures.
 
