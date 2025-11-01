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
- [In Progress] Replace `wgp.generate_video` persistence paths with `MediaPersistenceContext` helpers supplied by `ProductionManager`.
- [Planned] Lift prompt-enhancer and LoRA orchestration into dedicated adapters so CLI flows depend on `ProductionManager` instead of `wgp`. Dependency details live in `docs/CONTEXT.md` (“ProductionManager Dependency Snapshot”).
- [Planned] Continue peeling residual runtime globals (model load/release, queue callbacks) listed in `docs/CONTEXT.md`.

**Milestone 3 – Harden CLI orchestration (Active)**
- [In Progress] Keep queue management under `cli/` while retiring legacy shims; move any remaining helpers out of `core/`.
- [Planned] Finish disk-first workflows for mask/voice pipelines and document validation expectations for each CLI entrypoint.

**Milestone 4 – Prune GUI-dependent tooling (Ongoing)**
- [In Progress] Audit preprocessing utilities for GUI assumptions and refactor or remove as needed.
- [Pending] Gate or retire models that still require interactive inputs.
- [Planned] Extract the prompt enhancer stack into a shared module and stand up a provider abstraction for local/cloud backends.

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
- Replace the direct `server_config` lookups in `wgp.generate_video` with the run-scoped `MediaPersistenceContext`.
  - Proposal: thread the context handed in by `GenerationRuntime` through the video/image save paths, using `media_context.video_config(...)` / `media_context.image_config(...)` to assemble persistence settings and pivot debug-mask persistence to `media_context.should_save_masks()`.
  - Rationale: exercises the new context so codecs/containers/quality options stay centralised, minimises future override plumbing, and isolates the legacy module from `server_config` mutations.
  - Implementation sketch: update the `save_video` / `save_image` call sites to accept config objects (or kwargs derived from them) behind a thin compatibility layer, ensure temporary MMAudio flows still receive container overrides, and keep logging unchanged while context adoption is rolled out incrementally.
- Design prompt-enhancer and LoRA injection hooks for `ProductionManager` (loader interfaces, state caching, TaskInputManager touchpoints) ahead of the runtime refactor.
  - Proposal: split the prep work into two adapters under `core/`—a `LoRAInjectionManager` that wraps `setup_loras`, preset extraction, and multiplier resolution, plus a `PromptEnhancerBridge` that encapsulates enhancer provider setup / reset / processing. `ProductionManager` will vend cached instances tied to the current `server_config` so CLI callers stop reaching into `wgp`.
  - Rationale: reduces redundant module imports in the CLI, lets queue/worker flows share cached enhancer + LoRA metadata, and clears a path to delete the remaining `wgp` helpers once the adapters own all interactions.
  - Implementation sketch: introduce pure-Python interfaces in `core/lora/` and `core/prompt_enhancer/`, teach `TaskInputManager` to request them via `ProductionManager`, and update `cli.generate` to depend solely on the adapters while `wgp` keeps a temporary fallback shim for legacy callers.

---

## Handoff Protocol
- Start each session by reviewing `docs/CONTEXT.md` for architectural notes and `docs/WORK_HISTORY.md` for the latest work, then consult this plan's `## Immediate Next Actions`.
- Execute a coherent batch of related tasks from `## Immediate Next Actions` — enough for measurable progress but with a **focused** scope.
- After completion, append a detailed entry to `docs/WORK_HISTORY.md` (covering code changes, modules touched, assets removed, and validation runs; omit trivial edits). 
- Remove finished tasks from `## Immediate Next Actions`.
- Add follow-up tasks or new task derived from the roadmap/objective to `## Immediate Next Actions` in proper order; 
- Keep `## Immediate Next Actions` as a concise, timeless checklist.
- Audit and prune `## Project Roadmap`, cleaning up stale milestones.
- Review `## Validation Expectations` and update as needed.
- Update `docs/APPENDIX_HEADLESS.md` and `docs/CLI.md` for changes.
- Record deeper notes, learnings, and observations in `docs/CONTEXT.md`; prune stale entries.
- Keep `PROJECT_PLAN_LIVE.md` clear, consistent, and easy for future agents to follow.
