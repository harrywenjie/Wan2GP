# Project Plan Live

## Objective
Transition **Wan2GP** from a Gradio-centric application into a **lightweight, headless CLI tool** focused solely on core video-generation workflows.  
All removals or refactors must preserve reproducibility, deterministic generation paths, and stable GPU resource management.  
The headless build never exposes GUI-driven affordances — video/audio playback, galleries, or interactive widgets must be removed entirely; only on-disk artifacts remain supported.

---

## Project Roadmap
1. **Purge Gradio and plugin surface**
   - [Done] Delete `shared/gradio/*`, `shared/utils/plugins.py`, the entire `plugins/` tree, and plugin hooks from `wgp.py`.
   - [Done] Remove remaining `gradio` imports in `wgp.py` / `preprocessing/matanyone/app.py`, replacing event/progress helpers with CLI-native dataclasses.
   - [In Progress] Strip residual HTML/theme toggles and delete lingering UI assets (CLI theme flag removed; audit screenshots/theme knobs before final cleanup).

### Plugin Removal Plan (2025-02-14)
- **Phase 1 – Detach plugin manager**
  - [Done] Freeze dependencies on `server_config["enabled_plugins"]` and document the CLI-only configuration flow before excision.
  - [Done] Delete `shared/utils/plugins.py`, `plugins.json`, and the `plugins/` directory; strip `WAN2GPApplication` usage plus related imports from `wgp.py`.
  - [Done] Replace plugin tab wiring (`app.initialize_plugins`, `setup_ui_tabs`, `run_component_insertion`) with explicit CLI/queue stubs so the remaining code no longer assumes plugin hooks.
- **Phase 2 – Retire Gradio galleries**
  - [Done] Remove `shared/gradio/audio_gallery.py` and `shared/gradio/gallery.py`; inline minimal gallery stubs inside `wgp.py` to preserve state wiring.
  - [Pending] Salvage any reusable media normalisation helpers into CLI utilities or delete them if no longer needed.
- **Phase 3 – Clean trailing assets**
  - [In Progress] Remove residual plugin/UI assets (icons + favicon removed; legacy screenshots/theme knobs still pending review).
  - [Done] Update documentation to clarify that plugin-driven experiences are no longer supported and note CLI equivalents if they exist.
  - [Done] Implemented CLI preview/progress logging (queue summary media annotations + debug preview logs) and deleted the unused HTML helpers alongside the `ui.html` shim.

2. **Establish the Production Manager core**
   - [Planned] Introduce `core/production_manager.py` with a `ProductionManager` class that owns model orchestration, notifier wiring, and the generation lifecycle previously handled in `wgp.py`.
   - [Planned] Clarify that `ProductionManager` is the long-term replacement for `wgp.py`; treat any wrapper usage as a temporary bridge while we migrate the remaining helpers.
   - [Planned] Identify and inject dependencies (models, loaders, metadata helpers) so the class operates without mutating module-level globals.
   - [Planned] Provide a thin compatibility shim so current CLI entrypoints can invoke `ProductionManager` while the remaining `wgp` functionality is migrated, and track the follow-up work required to delete `wgp.py` entirely.

3. **Externalise queueing and CLI orchestration**
   - [In Progress] Keep queue management under `cli/` (and future worker integrations) while `ProductionManager` exposes stateless generation hooks.
   - [Planned] Relocate `QueueStateTracker` and related helpers out of `core/` into the CLI queue package so inference code remains the only surface under `core/`.
   - [In Progress] Adapt mask/voice workflows (MatAnyOne headless pipeline landed; audio editors still pending) to operate purely on disk-based inputs with CLI flags.
   - [Done] Replace `gr.*` notifications with structured logging/exception flows.

4. **Cull GUI-dependent features and models**
   - [In Progress] Drop or refactor preprocessing tools that still require canvases (`preprocessing/matanyone/` now headless; audit remaining tools).
   - [Pending] Audit models that depend on UI interactivity; remove or gate them until CLI workflows exist.
   - [In Progress] Document file-based inputs for retained models and update loaders (initial docs created; needs per-model validation).
   - [Planned] Extract the prompt enhancer stack (Florence2 + Llama-based rewrite helper) into a shared module and retire the LTX video diffusion pipeline once the enhancer has a neutral home.
   - [Planned] Introduce a prompt-enhancer provider abstraction that supports both the existing local models and future cloud LLM endpoints, including configuration for credentials, rate limits, and user-selectable provider flags.

5. **Retire ancillary runtimes**
   - [Pending] Remove Docker scripts (`run-docker-cuda-deb.sh`, `Dockerfile`, etc.) and legacy launch docs once CLI parity covers their use cases.
   - [Pending] Audit other deployment helpers (Pinokio, one-click installers) and drop UI-only tooling.

6. **Documentation sweep**
   - [Done] Rewrite `README.md`, `docs/CLI.md`, and related guides to describe the CLI-only workflow.
   - [In Progress] Keep this file (`PROJECT_PLAN_LIVE.md`) updated with each removal so future contributors know what was intentionally dropped.

### Deferred structural cleanup
- After the repo is fully headless, plan a follow-up migration that relocates `models/`, `preprocessing/`, `postprocessing/`, `shared/`, and other generation resources under `core/` to match the new Production Manager–centric layout.

---

## Validation Expectations
- For CLI regression checks, run `python -m cli.generate --prompt "smoke test prompt" --dry-run --log-level INFO` to confirm argument parsing, logging, and runtime overrides.
- Exercise preprocessing changes with `python -m cli.matanyone --input <video> --template-mask <mask> --dry-run` (or a short real run when GPU time permits).
- Record timing, VRAM usage, and output artifact paths in `docs/WORK_HISTORY.md` whenever a command executes generation or preprocessing work.
- Keep any ad-hoc validation scripts lightweight and remove them once automated coverage exists.

---

## Immediate Next Actions
- Outline the next ProductionManager extraction step: inventory remaining `wgp.py` dependencies and propose the next helper to migrate into `core/` or `cli/`.

---

## Handoff Protocol
- Start each session by reviewing `docs/CONTEXT.md` for architectural notes and `docs/WORK_HISTORY.md` for the latest work, then consult this plan's `## Immediate Next Actions`.
- Execute a coherent batch of related tasks from `## Immediate Next Actions` — enough for measurable progress but with a **focused** scope.
- After completion, append a detailed entry to `docs/WORK_HISTORY.md` (covering code changes, modules touched, assets removed, and validation runs). 
- Remove finished tasks from `## Immediate Next Actions`.
- Add follow-up tasks or new task derived from the roadmap/objective to `## Immediate Next Actions` in proper order; 
- Keep `## Immediate Next Actions` as a concise, timeless checklist.
- Audit `## Project Roadmap` and `## Validation Expectations`; update `docs/APPENDIX_HEADLESS.md` and `docs/CLI.md` as work lands.
- Record deeper notes, learnings, and observations in `docs/CONTEXT.md`.
- Keep `PROJECT_PLAN_LIVE.md` clear, consistent, and easy for future agents to follow.
