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
- Documentation audit confirms GUI theme and queue color references have been removed from maintained guides; no textual cleanup pending for this surface.
- Queue summaries now annotate attached media (start/end/guide/mask cues) and preview refreshes emit debug logs, eliminating the need for legacy modal HTML helpers in `wgp.py`.
- Retired the unused queue editing/modal/progress HTML functions (`handle_queue_action`, `refresh_preview`, `show_modal_image`, `create_html_progress_bar`, `update_generation_status`) and removed the `ui.html` shim; CLI telemetry already covers progress reporting.
- `wgp.initialize_runtime()` now builds a headless default namespace without touching `_parse_args()`, so importing the module no longer consumes CLI flags; callers can re-run the idempotent bootstrap via `ensure_runtime_initialized` while the CLI continues to rely on runtime overrides for per-run tweaks.
- Removed the orphaned Gradio scaffolding in `wgp.py` (`set_gallery_tab`, `generate_video_tab`, `get_js`, `create_ui`) along with the embedded JS payloads; no GUI-only helpers remain in the headless build.
- Remaining `ui.*` compatibility helpers (147x `ui.update`, 61x `ui.button`, etc.) are now confined to legacy preset/model management surfaces that the CLI never touches; they remain candidates for deletion or conversion to structured logging.
- Hoisted `DEFAULT_BOOTSTRAP_VALUES` into `shared/bootstrap_defaults.py`, dropped the defunct `betatest` / `multiple_images` flags, and now import the canonical defaults in `wgp`.
- Shared the bootstrap module with the CLI and extended `_load_server_config` to seed `check_loras`, `save_masks`, `save_quantized`, `save_speakers`, and `verbose_level` so persistent toggles no longer rely on the runtime namespace.
- Exposed the persisted runtime toggles (`save_masks`, `save_quantized`, `save_speakers`, `check_loras`) through headless CLI flags that keep `wgp.args` and `server_config` in sync during runs.
- Normalised the bootstrap defaults for `preload`, `profile`, and `verbose` to integers and hardened `_build_default_args` coercion so downstream code no longer relies on string shims.
- Queue editing stubs (`move_task`, `remove_task`, `finalize_generation_with_state`) have been removed; `update_queue_data` now returns a plain CLI summary dict so downstream code can introspect without UI components.
- Remaining `ui.*` usage now lives inside preset/model selection and legacy settings panes that are unreachable from the CLI; queue operations run entirely via CLI-native returns.
- Removed the prompt wizard/LoRA preset helpers from `wgp.py`, collapsing the compatibility shims and pointing operators to the existing CLI LoRA flags instead of macro-based prompts.
- Retired the queue load/save/autosave handlers plus their filesystem side effects; the CLI documentation now directs batching workflows toward explicit shell scripting rather than zipped queue archives.
- Removed the queue editing helpers (`silent_cancel_edit`, `cancel_edit`, `edit_task_in_queue`, `process_prompt_and_add_tasks`, `init_process_queue_if_any`) and deleted the pause-for-edit loop; the CLI queue no longer exposes edit state or `queue_paused_for_edit`.
- Documentation sweep (`README.md`, `docs/APPENDIX_HEADLESS.md`, `docs/*.md`) turned up no leftover references to queue archives or the prompt wizard; `docs/CLI.md` already notes both removals.
- Remaining UI-bound helpers fall into four dead zones: (1) generation button/preview gates (`abort_generation`, `prepare_generate_video`, the `ui.text` branch in `process_tasks`) that only impacted the dashboard; (2) LoRA preset editors (`validate_delete_lset`, `validate_save_lset`, `cancel_lset`, `save_lset`, `delete_lset`, `refresh_lora_list`) superseded by CLI flags; (3) media picker/post-processing hooks (`video_to_control_video`, `video_to_source_video`, `image_to_ref_image_*`, `audio_to_source_set`, `apply_post_processing`, `remux_audio`, `use_video_settings`) that correspond to removed upload widgets; and (4) model/preset switches (`record_image_mode_tab`, `load_settings_from_file`, `preload_model_when_switching`, the refresh_* helpers, `detect_auto_save_form`, `generate_dropdown_model_list` + `change_model_*`) which orchestrated UI dropdowns. None have CLI call sites today.
- Excised `abort_generation`, `prepare_generate_video`, the LoRA preset editors, and the media picker/post-processing helpers; queue state transitions now rely purely on logging and dict returns instead of `ui.*`.
- Updated `process_tasks` to emit timestamp tuples for preview refreshes so the generator is fully headless while retaining cached previews for debugging.
- Implemented a CLI `--settings-file` flag that loads JSON or media metadata via `get_settings_from_file`, merges the defaults ahead of CLI overrides, and aborts when the file targets a different model type.
- Removed the legacy model/preset switching helpers (`record_image_mode_tab`, `switch_image_mode`, `change_model*`, `generate_dropdown_model_list`, `create_models_hierarchy`, etc.); `wgp.py` no longer imports `defaultdict` or returns `ui.dropdown` instances, and `python -m compileall wgp.py` still passes.
- Prompt enhancement now runs exclusively through `process_prompt_enhancer` inside `generate_video`; the manual `enhance_prompt` trigger along with the prompt-type, resolution/group, auto-save, and LoRA download UI wrappers have been deleted, so CLI configs (or future flags) are the sole control surface for that feature set.
- Proposed a CLI-facing control surface for the enhancer: add a `--prompt-enhancer` flag mapping to the legacy `"", "T", "I", "TI"` modes (off / text / image / text+image) and introduce a provider selector so the Florence2+LLaMA stack and future cloud LLMs can share the same entry point.
- Confirmed there are no remaining `ui.update` call sites; `shared.ui_compat` is now effectively idle and can be removed once the last compatibility hooks are collapsed.

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

2. **Carve out a CLI-first core**
   - [In Progress] Relocate queue management, GPU scheduling, and generation routines from `wgp.py` into structured modules under `core/` / `cli/` while keeping CLI behaviour stable.
   - [Done] Replace `gr.*` notifications with structured logging/exception flows.
   - [In Progress] Adapt mask/voice workflows (MatAnyOne headless pipeline landed; audio editors still pending) to operate purely on disk-based inputs with CLI flags.

3. **Cull GUI-dependent features and models**
   - [In Progress] Drop or refactor preprocessing tools that still require canvases (`preprocessing/matanyone/` now headless; audit remaining tools).
   - [Pending] Audit models that depend on UI interactivity; remove or gate them until CLI workflows exist.
   - [In Progress] Document file-based inputs for retained models and update loaders (initial docs created; needs per-model validation).
   - [Planned] Extract the prompt enhancer stack (Florence2 + Llama-based rewrite helper) into a shared module and retire the LTX video diffusion pipeline once the enhancer has a neutral home.
   - [Planned] Introduce a prompt-enhancer provider abstraction that supports both the existing local models and future cloud LLM endpoints, including configuration for credentials, rate limits, and user-selectable provider flags.

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
- Removed the dormant `wgp.py` UI hooks (`enhance_prompt`, `refresh_*prompt_type*`, resolution/guidance/auto-save toggles, and `download_lora`), keeping the generation-time `process_prompt_enhancer` flow intact; no runtime validation required for these deletions.
- Documented the upcoming CLI prompt-enhancer flags/provider abstraction and confirmed `ui.update` no longer appears anywhere, setting the stage to delete `shared/ui_compat.py` once remaining helpers are inlined.
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
- Audited documentation and tracked assets for GUI theme or queue color mentions; none remain, so no textual updates were required this session.
- Inspected `wgp.py` and `shared/ui_compat.py` for legacy HTML/theme hooks, catalogued the unused preview-modal/progress helpers, and staged follow-up tasks to replace them with CLI logging.
- Confirmed via `rg` that the preview/modal/progress helpers have no runtime callers, drafted the CLI logging replacement strategy, and updated the roadmap plus next steps accordingly.
- Updated queue summaries to flag attached media, log thumbnail refreshes at debug level, removed dormant modal/progress helpers (`handle_queue_action`, `refresh_preview`, `show_modal_image`, `create_html_progress_bar`, `update_generation_status`), and dropped `ui.html`; validated with `python -m compileall wgp.py shared/ui_compat.py`.
- Tallied remaining `ui.*` compatibility usage (e.g., 147 `ui.update`, 61 `ui.button`) within queue editing, wizard, and preset routines, and committed to removing those chains or providing explicit CLI error paths.
- Excised unused queue editing helpers (`move_task`, `remove_task`, `finalize_generation_with_state`) and converted `update_queue_data` to return a CLI-friendly summary dict rather than `ui.text`.
- Audited the surviving `ui.*` call sites (prompt wizard, LoRA presets, queue import/autosave) and confirmed they are unreachable via CLI execution; leaning toward removal instead of building equivalent CLI surfaces unless new requirements emerge.
- Drafted removal proposals covering the prompt wizard/LoRA preset helpers and the queue import/autosave routines so the next iteration can either land CLI replacements or delete the dead UI scaffolding.
- Removed the prompt wizard/LoRA preset helpers plus queue load/save/autosave handlers from `wgp.py`, pruned the associated constants/imports, refreshed `docs/CLI.md` with LoRA guidance and queue automation notes, and validated the module with `python -m compileall wgp.py`.
- Audited documentation for lingering queue archive or prompt wizard references; confirmed the updates are confined to `docs/CLI.md` and no additional cleanup is required.
- Excised the remaining queue editing helpers (`silent_cancel_edit`, `cancel_edit`, `edit_task_in_queue`, `process_prompt_and_add_tasks`, `init_process_queue_if_any`), removed the `queue_paused_for_edit` loop, dropped the now-unused `clean_image_list`, and recompiled `wgp.py` via `python -m compileall wgp.py` to confirm a clean headless build.
- Removed the lingering dashboard helpers (`abort_generation`, `prepare_generate_video`), LoRA preset editors, media picker/post-processing hooks, and `load_settings_from_file`; rewired `process_tasks` preview yields to timestamp tuples and verified `python -m compileall wgp.py`.
- Restored the `video_guide_processes` constant in `wgp.py` to resolve the undefined reference surfaced by Pylance after the UI compatibility layer removal; validated with `python -m compileall wgp.py`.
- Landed CLI prompt enhancer controls by adding `--prompt-enhancer`/`--prompt-enhancer-provider`, wired the selections into `cli.generate.apply_runtime_overrides` and `build_params`, refreshed dry-run reporting, and documented the new surface in `docs/CLI.md`; validated with `python -m compileall cli/generate.py`.
- Deleted `shared/ui_compat.py`, removed its `wgp.py` import, and confirmed the headless runtime compiles cleanly with `python -m compileall wgp.py`.
- Added a `--settings-file` CLI flag that merges saved defaults ahead of explicit overrides, updated `docs/CLI.md`, validated parsing with `python -m compileall cli/generate.py`, and exercised the dry-run path via `python -m cli.generate --prompt "test" --dry-run`.
- Deleted the model/preset switching shims (`record_image_mode_tab`, `switch_image_mode`, `change_model*`, `generate_dropdown_model_list`, `create_models_hierarchy`) and orphaned imports, then recorded the outstanding `ui.update` hotspots (prompt enhancer + prompt-type toggles, resolution/guidance controls, auto-save/video-length hooks) for follow-up; re-ran `python -m compileall wgp.py`.
- Removed the dormant prompt enhancer UI helpers from `wgp.py`, documented the change in the roadmap (shared prompt enhancer + future cloud providers), and deferred runtime validation because no executable paths were touched.
- CLI prompt enhancer controls now ship as `--prompt-enhancer` (text/image/combined) with an optional `--prompt-enhancer-provider` flag that maps to `server_config["enhancer_enabled"]` (0=off, 1=Florence2+Llama3.2, 2=Florence2+JoyCaption) while preserving the disabled default when both flags are omitted.
- Deleted `shared/ui_compat.py`; the headless runtime no longer imports the compatibility layer and `wgp.py` operates purely with CLI returns.
- Reviewed `wgp.py` for lingering UI scaffolding, documented the `_parse_args` bootstrap side effects and orphaned JS helpers, and refreshed `## Immediate Next Actions` to prioritize splitting the runner from the legacy UI hooks.
- Replaced the import-time `_parse_args()` side-effects with `_build_default_args()` and `initialize_runtime()`, added `ensure_runtime_initialized()` for idempotent bootstrapping, updated the CLI wrapper to call it, and deleted the final Gradio helpers (`set_gallery_tab`, `generate_video_tab`, `get_js`, `create_ui`); validated via `python -m compileall wgp.py` and `python -m compileall cli/generate.py`.
- Centralised the bootstrap defaults around `DEFAULT_BOOTSTRAP_VALUES`, added `_load_server_config()` to hydrate config before runtime overrides, updated `cli.generate` to call `ensure_runtime_initialized()`, and refreshed `docs/CLI.md` plus `docs/APPENDIX_HEADLESS.md` with the runtime bootstrap guidance; revalidated via `python -m compileall wgp.py` and `python -m compileall cli/generate.py`.
- Relocated the bootstrap constants to `shared/bootstrap_defaults.py`, purged the unused `betatest`/`multiple_images` flags, extended `wgp.py` to read persistent toggles from `server_config`, wired `cli/arguments.py` to the shared defaults for help text, and recorded the changes in this plan; validated with `python -m compileall wgp.py cli/arguments.py cli/generate.py shared/bootstrap_defaults.py`.
- Surface the persisted runtime toggles in `cli.generate` (`--save-masks`, `--save-quantized`, `--save-speakers`, `--check-loras`), normalised `preload`/`profile`/`verbose` bootstrap values to integers, updated docs with the new flags, and verified the modules with `python -m compileall cli/generate.py cli/arguments.py shared/bootstrap_defaults.py wgp.py`.
- Ensured the CLI toggle overrides remain per-run by keeping them out of `server_config`, refreshed `docs/CLI.md` with the ephemeral contract, and confirmed dry-run/log visibility via `python -m cli.generate --prompt "test" --dry-run --save-masks --no-check-loras`.

---

## Immediate Next Actions
- Audit remaining CLI overrides (prompt enhancer, output directory, VRAM profile settings) to document which ones intentionally persist to `wgp_config.json` versus per-run state.
- Draft the refactor plan to extract the generation runner (`assemble_generation_params` + `generate_video` hand-off) into a CLI-first module without dragging UI globals.

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
