# Project Plan Live

## Objective
Transition **Wan2GP** from a Gradio-centric application into a **lightweight, headless CLI tool** focused solely on core video-generation workflows.  
All removals or refactors must preserve reproducibility, deterministic generation paths, and stable GPU resource management.  
The headless build never exposes GUI-driven affordances — video/audio playback, galleries, or interactive widgets must be removed entirely; only on-disk artifacts remain supported.

---

## Context And Findings
- `wgp.py` remains the central orchestrator; queue management and generation logic are still co-located there, but the queue renderer now emits text-only summaries suitable for CLI logging.
- `preprocessing/matanyone/app.py` has been rewritten as a headless pipeline that requires on-disk template masks and GPU access; a dedicated CLI wrapper still needs to surface the new API.
- MatAnyOne CLI contract: accepts a video or still image, a grayscale mask aligned to the first processed frame, optional frame windowing, and produces foreground/alpha MP4 pairs (plus optional RGBA ZIP) under `mask_outputs/`, reattaching source audio when requested.
- Multiple model families (Wan, Flux, Qwen, Chatterbox, LTX) remain in-tree; CLI coverage is strongest for Wan, with other families pending validation.
- `cli/matanyone.py` now implements the MatAnyOne CLI wrapper, reusing `cli.telemetry.configure_logging`, validating media/mask paths, exposing frame range/dimension/mask-type/audio/codec flags (with dry-run support), and forwarding requests to `preprocessing.matanyone.app.generate_masks` using a fresh CLI GPU state plus log-backed notifier.
- Removed the legacy theme configuration surface in `wgp.py` (`--theme` flag plus `UI_theme`/`queue_color_scheme` defaults); confirmed no downstream callers rely on these settings and CLI logging remains unaffected.
- Headless flow remains CLI → `assemble_generation_params()` → `wgp.generate_video()`; the intermediary “core runner” extraction is still pending. A detailed diagram plus follow-ups live in `docs/APPENDIX_HEADLESS.md`.
- Model footprint: keep `models/wan` (and associated defaults/presets/settings) plus shared utilities; treat `models/flux`, `models/qwen`, `models/chatterbox`, and `models/ltx_video` as optional until full CLI parity exists; retain only stateless preprocessing helpers under `preprocessing/`.
- CLI flag reference migrated to `docs/CLI.md`; new runtime toggles should be documented there instead of embedding matrices in this file.

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

2. **Carve out a CLI-first core**
   - [In Progress] Relocate queue management, GPU scheduling, and generation routines from `wgp.py` into structured modules under `core/` / `cli/` while keeping CLI behaviour stable.
   - [Done] Replace `gr.*` notifications with structured logging/exception flows.
   - [In Progress] Adapt mask/voice workflows (MatAnyOne headless pipeline landed; audio editors still pending) to operate purely on disk-based inputs with CLI flags.

3. **Cull GUI-dependent features and models**
   - [In Progress] Drop or refactor preprocessing tools that still require canvases (`preprocessing/matanyone/` now headless; audit remaining tools).
   - [Pending] Audit models that depend on UI interactivity; remove or gate them until CLI workflows exist.
   - [In Progress] Document file-based inputs for retained models and update loaders (initial docs created; needs per-model validation).

4. **Retire ancillary runtimes**
   - [Pending] Remove Docker scripts (`run-docker-cuda-deb.sh`, `Dockerfile`, etc.) and legacy launch docs once CLI parity covers their use cases.
   - [Pending] Audit other deployment helpers (Pinokio, one-click installers) and drop UI-only tooling.

5. **Documentation sweep**
   - [Done] Rewrite `README.md`, `docs/CLI.md`, and related guides to describe the CLI-only workflow.
   - [In Progress] Keep this file (`PROJECT_PLAN_LIVE.md`) updated with each removal so future contributors know what was intentionally dropped.

---

## Validation Expectations
- For CLI regression checks, run `python -m cli.generate --prompt "smoke test prompt" --dry-run --log-level INFO` to confirm argument parsing, logging, and runtime overrides.
- Exercise preprocessing changes with `python -m cli.matanyone --input <video> --template-mask <mask> --dry-run` (or a short real run when GPU time permits).
- Record timing, VRAM usage, and output artifact paths in `## Previous Work Summary` whenever a command executes generation or preprocessing work.
- Keep any ad-hoc validation scripts lightweight and remove them once automated coverage exists.

---

## Previous Work Summary
- Retired the InsightFace-based `preprocessing/face_preprocessor.py`; Lynx and Stand-in code paths now halt with clear messages until the replacement cropper exists.
- Deleted the HyVideo/Hunyuan stack (code, defaults, lora scaffolding); removed all documentation and runtime references so the CLI focuses on Wan/Flux/LTX/Qwen/Chatterbox families.
- Created `cli/generate.py` to invoke `generate_video()` headlessly with CLI-friendly logging and a dry-run configuration test to confirm argument parsing.
- Drafted a CLI-oriented architecture map detailing the flow from argument parsing through model execution to output emission.
- Unified CLI and legacy code paths by introducing `assemble_generation_params()` with shared fallback defaults.
- Captured the proposed CLI argument surface and enumerated required file-based inputs for non-interactive workflows.
- Introduced `cli/arguments.py` to centralize the CLI flag surface, expanded `cli/generate.py` to forward path-based inputs, and updated `wgp._parse_args` to tolerate unknown flags; validated with `python -m cli.generate --prompt "test prompt" --dry-run`.
- Added CLI-side file path validation across all media inputs, exiting early with actionable errors; exercised via `python -m cli.generate --prompt "test" --dry-run --image-start missing.png`.
- Replaced ad-hoc `print` statements with structured CLI logging backed by a configurable `--log-level`; built send_cmd telemetry hooks and validated with `python -m cli.generate --prompt "test prompt" --dry-run --log-level DEBUG`.
- Audited `GENERATION_FALLBACKS` against per-model defaults; added `tea_cache_setting` and `tea_cache_start_step_perc` fallbacks to preserve parity for TEA cache workflows.
- Cataloged all remaining Gradio-dependent modules (core UI bootstrap, shared/gradio widgets, plugin system, stats panel, process locks, and Wan/Qwen/Chatterbox handlers) and tagged each for removal or CLI refactor follow-ups.
- Removed direct Gradio usage from Wan/Qwen/Chatterbox model handlers by wiring a shared CLI notifier to the generation logger; confirmed CLI dry-run still passes.
- Replaced the GPU lock wait notification with a logger/callable hook (`shared/utils/process_locks.py`), updated MatAnyOne to pass a Gradio callback locally, and removed the Gradio-only stats streamer (`shared/utils/stats.py` + related `wgp.py` wiring); validated with `python -m cli.generate --prompt "test" --dry-run`.
- Routed remaining notification paths in `wgp.py` through `shared.utils.notifications` by swapping `gr.Info`/`gr.Warning`/`gr.Error` for logger-backed helpers and a new `GenerationError`, preserving CLI error reporting; confirmed via `python -m cli.generate --prompt "test prompt" --dry-run`.
- Added `shared/ui_compat.py` to encode legacy UI responses as dataclasses, replaced `gr.*` return values across the queue helpers in `wgp.py` with the new compatibility layer, and verified CLI dry-run `python -m cli.generate --prompt "test prompt" --dry-run`.
- Removed the plugin manager infrastructure (deleted `shared/utils/plugins.py`, `plugins/`, and `plugins.json`; stripped plugin imports/hooks from `wgp.py`); CLI dry-run `python -m cli.generate --prompt "test" --dry-run` still passes.
- Removed `shared/gradio/*` and replaced `AudioGallery` / `AdvancedMediaGallery` with inline minimal stubs inside `wgp.py`; CLI dry-run `python -m cli.generate --prompt "test" --dry-run` still passes.
- Eliminated the remaining gallery/playback code paths in `wgp.py`, replacing them with hard errors to enforce the headless-only policy; `python -m cli.generate --prompt "test" --dry-run` still passes.
- Replaced GUI-era documentation with CLI-first guidance: deleted `docs/PLUGINS.md`, rewrote `README.md`, refreshed `docs/CLI.md`, and condensed `docs/CHANGELOG.md` around the headless workflow; no runtime validation was required.
- Completed the documentation sweep for the remaining guides (`docs/INSTALLATION.md`, `docs/AMD-INSTALLATION.md`, `docs/GETTING_STARTED.md`, `docs/TROUBLESHOOTING.md`, `docs/FINETUNES.md`, `docs/LORAS.md`, `docs/MODELS.md`, `docs/VACE.md`), replacing Gradio/Docker instructions with CLI scenarios and noting current gaps (e.g. lora activation still configured via presets); focused on content updates, no code changes to validate.
- Implemented the CLI LoRA discovery and selection flow (`--list-loras`, `--list-lora-presets`, `--loras`, `--lora-preset`, `--lora-multipliers`) with preset parsing and logging hooks; exercised via `python -m cli.generate --prompt "test" --dry-run`, `--list-loras`, `--list-lora-presets`, and expected error paths for missing LoRA weights/presets.
- Exposed headless runtime controls by extending `cli/arguments.py`/`cli.generate.py` with attention, compile, profile/preload, dtype/quant, and TeaCache flags; injected per-run overrides into `wgp` state, refreshed dry-run reporting/logging, and documented the new surface in `docs/CLI.md`. Validated with `python -m cli.generate --prompt "test prompt" --dry-run --attention sdpa --compile --profile 3 --preload 512 --tea-cache-level 1.75 --tea-cache-start-perc 25 --transformer-quantization none --text-encoder-quantization fp8 --fp16`.
- Swapped the queue renderer in `wgp.py` for a text-only summary, deleted the `icons/` directory and `favicon.png`, removed `gradio`-specific types in favour of `ui.UIEvent`, and rebuilt `preprocessing/matanyone/app.py` as a headless mask-propagation pipeline with explicit request/result dataclasses; sanity-checked with `python -m compileall wgp.py` and `python -m compileall preprocessing/matanyone/app.py`.
- Added `cli/matanyone.py` to expose MatAnyOne mask propagation as a CLI command with logging, path validation, frame/mask/audio controls, and dry-run support; documented the workflow in `docs/CLI.md` and verified the parser with `python -m cli.matanyone --help`.
- Reorganized `PROJECT_PLAN_LIVE.md` to keep context/responsive sections concise, moved the architecture diagram and follow-ups into `docs/APPENDIX_HEADLESS.md`, and pointed readers to `docs/CLI.md` for the canonical flag surface.
- Removed the obsolete CLI theme surface (`--theme`, `UI_theme`, `queue_color_scheme`) from `wgp.py`; confirmed the parser via `python -m cli.generate --prompt "smoke test prompt" --dry-run --log-level INFO`.

---

## Immediate Next Actions
- Audit docs and assets for any remaining references to GUI themes or queue color presets; prune or restate them for the CLI-only workflow.

---

## Handoff Protocol
- Execute a coherent batch of related tasks from `## Immediate Next Actions` — enough to make measurable progress but keep the scope **focused**.
- After finishing a task, append a concise summary to the end of `## Previous Work Summary`.  
  Include changes made, modules investigated, assets removed, and validation runs performed.  
  Then remove the completed task from `## Immediate Next Actions`.
- Insert any new or follow-up tasks into `## Immediate Next Actions` at the appropriate position in the sorted list.
- Keep `## Immediate Next Actions` as a concise, timeless checklist; move session-specific notes, timestamps, or narrative logs to `## Previous Work Summary` or other appropriate sections.
- Audit `## Context And Findings`, `## Project Roadmap`, and `## Validation Expectations`; refresh `docs/APPENDIX_HEADLESS.md` (architecture notes) plus `docs/CLI.md` (flag surface) when work lands.
- Ensure `PROJECT_PLAN_LIVE.md` remains clear and well-structured for seamless handover to future agents.
