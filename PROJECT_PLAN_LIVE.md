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
- Stand up adapter shims for LoRA injection and prompt enhancement (`core/lora/manager.py`, `core/prompt_enhancer/bridge.py`) while legacy callers still delegate through `wgp`.
  - Proposal: implement `LoRAInjectionManager` with discovery/preset caching plus `PromptEnhancerBridge` wrapping the existing bootstrap logic, both backed by memoised caches keyed on `(model_type, server_config_hash)`. Initial versions may call into the current helpers but must expose clear interfaces (`hydrate`, `presets`, `prime`, `enhance`, `reset`) for the CLI to consume.
  - Rationale: creating the adapters now lets ProductionManager vend shared instances, eliminates repeated globbing/model loads across queue workers, and unlocks incremental retirement of `wgp.setup_loras` / `wgp.setup_prompt_enhancer`.
  - Implementation sketch: add the new modules with thin wrappers over `wgp`, include smoke-style tests covering cache reuse and reset behaviour, document the API in `docs/CONTEXT.md`, and leave TODO hooks where phase-two extraction will replace the shims with native implementations.
- Wire `ProductionManager` and `TaskInputManager` to the new adapters so CLI orchestration stops importing `wgp` for LoRA/prompt enhancer duties.
  - Proposal: cache adapter instances on `ProductionManager`, thread them through `GenerationRuntime`, and update queue serialization plus metadata preparation to consume adapter payloads instead of calling `wgp.setup_loras`/`setup_prompt_enhancer` directly.
  - Rationale: centralising adapter access keeps run-state deterministic, ensures queue workers share caches, and reduces the surface area that still depends on legacy globals.
  - Implementation sketch: add `production_manager.lora_manager()` / `prompt_enhancer()` accessors, adjust `TaskInputManager.prepare_inputs_dict` to request payloads from the adapters, and keep a temporary legacy shim for older entrypoints until CLI coverage is complete.

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
