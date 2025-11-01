# Project Plan Live

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
- [Pending] Decide whether any gallery/media normalisation helpers should be salvaged into CLI utilities or deleted. (Prior plugin-removal phases are archived in `docs/WORK_HISTORY.md`.)

**Milestone 2 – Finish Production Manager extraction (Active)**
- [Completed] Replace `wgp.generate_video` persistence paths with `MediaPersistenceContext` helpers supplied by `ProductionManager`.
- [Completed] Retired `shared.utils.audio_video.save_*` shims after migrating remaining callers onto `core.io.media` (2025-11-03).
- [Planned] Lift prompt-enhancer and LoRA orchestration into dedicated adapters so CLI flows depend on `ProductionManager` instead of `wgp`. Dependency details live in `docs/CONTEXT.md` (“ProductionManager Dependency Snapshot”).
- [Planned] Continue peeling residual runtime globals (model load/release, queue callbacks) listed in `docs/CONTEXT.md`.

**Milestone 3 – Harden CLI orchestration (Active)**
- [In Progress] Keep queue management under `cli/` while retiring legacy shims; move any remaining helpers out of `core/`.
- [Completed] Added MatAnyOne CLI manifest integration smoke coverage verifying `mask_foreground`, `mask_alpha`, and `rgba_archive` roles (2025-11-03).
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
- Run `python -m unittest tests.test_matanyone_cli_integration` to ensure MatAnyOne manifest emission stays in sync with the CLI surface.
- Exercise preprocessing changes with `python -m cli.matanyone --input <video> --template-mask <mask> --dry-run` (or a short real run when GPU time permits).
- When MatAnyOne emits artifacts, inspect `<output_dir>/manifests/run_history.jsonl` (or the `--manifest-path` override) and confirm entries include `mask_foreground`, `mask_alpha`, and optional `rgba_archive` roles.
- Record timing, VRAM usage, and output artifact paths in `docs/WORK_HISTORY.md` whenever a command executes generation or preprocessing work.
- Keep any ad-hoc validation scripts lightweight and remove them once automated coverage exists.

---

## Immediate Next Actions
- Persist MatAnyOne audio tracks through `MediaPersistenceContext.save_audio` instead of relying on pipeline stubs.
  - **Proposal (2025-11-04)**: Decode or remux the temporary AAC tracks into context-managed outputs, invoke `save_audio` with per-track sample rates, and surface the saved paths in `MatAnyOneResult` metadata so manifests reflect the canonical audio artifacts produced by real runs.
  - **Rationale**: Recording concrete audio files keeps preprocessing parity with generation workflows, eliminates reliance on test-only stubs, and ensures downstream automation can hash and archive audio assets alongside mask outputs.
  - **Design (2025-11-04)**: Introduce an audio persistence helper in `preprocessing.matanyone.app` that shells out to `ffmpeg` (or streams via `soundfile`) to convert each extracted track, honour context overrides, and update the integration suite to exercise the end-to-end path without mocking `save_audio`.
- Sweep remaining preprocessing/generation modules for direct `imageio` writers and schedule migrations onto `core.io.media`.
  - **Proposal (2025-11-04)**: Use `rg` to catalogue outstanding `imageio.get_writer` usage (e.g. `preprocessing/dwpose`, `models/ltx_video`), prioritise high-traffic pipelines, and plan incremental refactors that reuse shared Video/Image save configs.
  - **Rationale**: Unifying persistence on the context layer keeps retry/logging semantics consistent and prevents regressions as codec overrides evolve.
  - **Design (2025-11-04)**: Track each call site in a checklist, extract small wrappers that forward to `write_video`/`write_image`, and add regression tests where the legacy writers had bespoke parameters (macro block sizes, CRF overrides) before retiring the direct `imageio` calls.
  
