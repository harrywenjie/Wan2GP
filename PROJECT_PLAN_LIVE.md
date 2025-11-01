# Project Plan Live

## Objective
Transition **Wan2GP** from a Gradio-centric application into a **lightweight, headless CLI tool** focused solely on core video-generation workflows.  
All removals or refactors must preserve reproducibility, deterministic generation paths, and stable GPU resource management.  
The headless build never exposes GUI-driven affordances — video/audio playback, galleries, or interactive widgets must be removed entirely; only on-disk artifacts remain supported.

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
- [Planned] Thread `MediaPersistenceContext` through MatAnyOne so preprocessing mask/audio exports mirror the headless generation pipeline.

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
- Record timing, VRAM usage, and output artifact paths in `docs/WORK_HISTORY.md` whenever a command executes generation or preprocessing work.
- Keep any ad-hoc validation scripts lightweight and remove them once automated coverage exists.

---

## Immediate Next Actions
- Port MatAnyOne mask/audio writers to the shared `MediaPersistenceContext` so preprocessing artefacts respect codec overrides and the `save_masks` toggle.
  - **Proposal (2025-11-04)**: Thread the per-run media context through `cli.matanyone` and replace direct `save_image`/ZIP calls with `_save_*_artifact` helpers, mirroring the main generation path.
  - **Rationale**: Keeps preprocessing outputs aligned with the headless persistence pipeline, preventing drift between mask/audio artefacts produced by generation and MatAnyOne.
- Define a lightweight artifact manifest spec for CLI runs (JSONL or per-run summary) that captures saved paths, metadata mode, and adapter payload hashes.
  - **Proposal (2025-11-04)**: Draft a minimal schema, prototype emission in dry-run mode, and socialise with operators before wiring into `cli.generate`.
  - **Rationale**: Replaces ad-hoc logging with machine-readable manifests, improving reproducibility audits and integration with external schedulers.
- Plan the removal of legacy `save_video`/`save_image` wrappers once all call sites use `MediaPersistenceContext`.
  - **Proposal (2025-11-04)**: Audit remaining usages, enumerate blockers, and sketch a deprecation timeline so the wrappers can disappear when the runner extraction lands.
  - **Rationale**: Eliminating the wrappers reduces duplication, making the context the single persistence surface and simplifying future refactors.

---

## Handoff Protocol
- Start each session by reviewing `docs/CONTEXT.md` for architectural notes and `docs/WORK_HISTORY.md` for the latest work, then consult this plan's `## Immediate Next Actions`.
- Execute a coherent batch of related tasks from `## Immediate Next Actions` — enough for measurable progress but with a **focused** scope.
- After completion, replace the existing `docs/WORK_HISTORY.md` content with a concise (~800 token) summary of prior sessions, then append today’s detailed entry covering code changes.
- Remove finished tasks from `## Immediate Next Actions`.
- Add follow-up tasks or new task derived from the roadmap/objective to `## Immediate Next Actions` in proper order; 
- Keep `## Immediate Next Actions` as a concise, timeless checklist.
- Audit and prune `## Project Roadmap`, cleaning up stale milestones.
- Review `## Validation Expectations` and update as needed.
- Update `docs/APPENDIX_HEADLESS.md` and `docs/CLI.md` for changes.
- Record deeper notes, learnings, and observations in `docs/CONTEXT.md`; prune stale entries.
- Keep `PROJECT_PLAN_LIVE.md` clear, consistent, and easy for future agents to follow.
