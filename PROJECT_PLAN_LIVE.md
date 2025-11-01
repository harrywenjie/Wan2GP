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
- [In Progress] Keep queue management under `cli/` while retiring legacy shims; move any remaining helpers out of `core/`.
- [Completed] Added MatAnyOne CLI manifest integration smoke coverage verifying `mask_foreground`, `mask_alpha`, and `rgba_archive` roles (2025-11-02).
- [Completed] Persisted MatAnyOne audio tracks through `MediaPersistenceContext.save_audio`, wiring decoded artifacts into manifests and metadata sidecars (2025-11-02).
- [Completed] Surfaced MatAnyOne audio artifact metadata (sample rate, duration, language, channels) through CLI logging and manifest entries for downstream automation (2025-11-02).
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

## Project Specific Live Docs

- `docs/APPENDIX_HEADLESS.md` - Architectural appendix for the headless CLI migration, queue orchestration, and persistence references.
- `docs/CLI.md` - CLI usage guide covering generation/preprocessing flags, runtime toggles, and the artifact manifest schema.

---

## Immediate Next Actions
- Surface MatAnyOne audio metadata through queue summaries and schema references so automation can consume it without reading manifests.
  - **Proposal (2025-11-02)**: Extend `cli.queue_state.QueueStateTracker` and CLI status endpoints to emit `audio_tracks` entries (path, sample_rate, duration_s, language, channels) while updating `docs/CLI.md`/schema snippets to match.
  - **Rationale**: Runs now persist and log canonical audio metadata; exposing the same details via queue/status APIs lets orchestration layers preflight mux compatibility before artifacts land on disk.
  - **Design (2025-11-02)**: Add optional `audio_tracks` blocks to queue summaries, document the schema update, and add focused tests that stub MatAnyOne runs to assert the new fields propagate.
- Sweep remaining preprocessing/generation modules for direct `imageio` writers and schedule migrations onto `core.io.media`.
  - **Proposal (2025-11-02)**: Use `rg` to catalogue outstanding `imageio.get_writer` usage (e.g. `preprocessing/dwpose`, `models/ltx_video`), prioritise high-traffic pipelines, and plan incremental refactors that reuse shared Video/Image save configs.
  - **Rationale**: Unifying persistence on the context layer keeps retry/logging semantics consistent and prevents regressions as codec overrides evolve.
  - **Design (2025-11-02)**: Track each call site in a checklist, extract small wrappers that forward to `write_video`/`write_image`, and add regression tests where the legacy writers had bespoke parameters (macro block sizes, CRF overrides) before retiring the direct `imageio` calls.
  
