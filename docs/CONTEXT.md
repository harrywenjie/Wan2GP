# Detailed Context

## Orchestration & Queue

- `wgp.py` still executes the core generation path. The CLI reaches it through `core.production_manager.ProductionManager.run_generation()`, which handles notifier injection, metadata wiring, and model reload checks before delegating to `wgp.generate_video()`.
- `cli.queue_controller.QueueController` is the primary queue runner (AsyncStream-backed). It owns abort/pause handling, drains queued tasks sequentially, and surfaces pause/resume/status/abort over the optional TCP control server (`cli.queue_control_server` + `cli.queue_control`). `tests/test_queue_prompt_payloads.py` locks in the enhanced prompt payload propagation path so queue snapshots keep the metadata required by downstream automation.
- Queue state lives under `cli.queue_state.QueueStateTracker`. Helper functions for queue summaries, abort/clear toggles, and counter resets live in `cli.queue_utils` / `cli.queue_state`, with `wgp.clear_queue_action` maintained only for legacy compatibility.
- `cli.runner` provides the CLI notifier (`CLIGenerationNotifier`), progress callback builder, and `send_cmd` bridge so telemetry updates stay in sync with `state["gen"]`. `cli.generate` traps `KeyboardInterrupt` and clears the queue via these helpers.

## MatAnyOne Pipeline

- `preprocessing/matanyone/app.py` is a headless pipeline that assumes on-disk source media/masks and GPU availability. `_persist_audio_artifacts` now decodes extracted AAC tracks via `ffmpeg` into float32 arrays, persists them through `MediaPersistenceContext.save_audio`, and records per-track metadata (sample rate, derived duration, channels, language, source codec) back into `MatAnyOneResult.metadata["audio_tracks"]`. JSON sidecars are written for each persisted audio artifact when the request runs in `metadata_mode=json`, mirroring the CLI logs.
- `cli/matanyone.py` wraps the pipeline with logging, input validation, frame/mask/audio controls, optional dry-run mode, and forwards requests to `generate_masks` using the shared CLI notifier. When `wgp` is available the CLI clones both `ProductionManager.metadata_state()` and `media_context()` so MatAnyOne writes metadata with the same templates and persists media through the shared context. Outputs land under `mask_outputs/` with container/codec overrides pulled from `server_config`; RGBA ZIP bundles now respect the context `save_masks` toggle while audio tracks are reattached onto the resolved container when requested.
- `tests/test_matanyone_persistence.py` guards the context-driven persistence flow by asserting video saves honour codec/container overrides, audio artifacts persist through `MediaPersistenceContext.save_audio` with per-track sample rates/languages, and mask archives follow the `save_masks` gating.
- `tests/test_matanyone_cli_integration.py` exercises the CLI end-to-end, patching the heavy pipeline while asserting the manifest records `mask_foreground`, `mask_alpha`, `rgba_archive`, and `audio` artifacts plus expected metadata sidecars and audio metadata fields (`sample_rate`, `duration_s`, `language`, `channels`).

## Metadata & IO

- `core/io/media.py` houses media persistence (`write_video`, `write_image`) plus `write_metadata_bundle`, all with logger-aware retry handling; `shared.utils.audio_video` now only exposes audio track utilities and metadata readers after retiring the legacy `save_*` shims.
- `ProductionManager.metadata_state()` returns a per-run `MetadataState` snapshot (choice + cloned templates). `GenerationRuntime` forwards the snapshot to `wgp.generate_video`, replacing the old module-level `metadata_choice` / `metadata_configs`.
- `_resolve_metadata_config` accepts either the dataclass or a dict for backward compatibility. CLI runs still honour `--metadata-mode`; MatAnyOne now passes the cloned `MetadataState` snapshot directly into its writers so embedded metadata and JSON sidecars stay aligned with the core generation pipeline.
- Embedded source images for metadata remain gated by `server_config["embed_source_images"]`; JSON sidecars are emitted when `metadata_mode=json`.
- `core.io.media.MediaPersistenceContext` now captures the video/image/audio/mask persistence templates plus the legacy `save_masks` debug toggle. `ProductionManager.media_context()` builds a fresh instance from `server_config` for each run and passes it through `GenerationRuntime` to `wgp.generate_video`, which routes saves through the context helpers while retaining the legacy wrappers as a fallback.
- Artifact manifests will be emitted as JSONL (`manifests/run_history.jsonl` inside the resolved `output_dir`). Each entry records saved artifact paths, the effective metadata mode, a reproducibility snapshot of CLI inputs, and SHA-256 hashes of adapter payloads derived from their canonical JSON serialisations (see `docs/CLI.md` for the public schema). Writers flush entries only after persistence succeeds; dry runs skip emission.
- `cli.generate` now wraps `ProductionManager.media_context()` with a `ManifestRecorder`, capturing every `MediaPersistenceContext.save_*` invocation before the JSONL writer emits a row. Adapter payload hashes are derived from canonical JSON, and failures log an `"error"` field while omitting artifacts so partially persisted runs never leak into downstream automation.

### Persistence Surface Update (2025-11-03)

- `wgp.save_video/save_image` have been removed. All persistence now flows through `MediaPersistenceContext` or directly into `core.io.media.write_*`, keeping a single code path for retries and logging.
- MatAnyOne fallbacks now call `write_video` when a context is unavailable, preserving codec/container overrides without touching the legacy `shared.utils.audio_video` shims.
- The legacy `shared.utils.audio_video.save_*` adapters have been deleted; any lagging preprocessors must migrate to `MediaPersistenceContext` or call `core.io.media.write_*` directly to preserve logging and retry behaviour.
- `shared.utils.utils.save_image` has been removed; any tensor-to-image persistence flows must route through `MediaPersistenceContext.save_image` or `core.io.media.write_image` so retry/logging semantics stay consistent.
- Remaining `models/wan` video utilities (`fantasytalking`, `multitalk`) now call `core.io.media.write_video`, keeping preprocessing helpers aligned with the CLI persistence stack instead of hand-rolled `imageio` writers.
- The MatAnyOne manifest recorder captures audio artifacts when reattaching source tracks; the CLI integration suite asserts the JSONL rows include `audio` roles with codec/container metadata plus `sample_rate`, `duration_s`, `language`, and `channels`.

## Pending Extraction Work

- Runner extraction: peel the remaining dependencies (`load_models`, prompt enhancer bootstrap, LoRA wiring, filesystem helpers) out of `wgp.generate_video` so `cli.runner` can own execution without touching module globals.
- Residual legacy utilities in `wgp.py` are down to preset/model management and enhancer setup; they will migrate into dedicated CLI modules or be deleted once replacements exist.

## Prompt Enhancer & LoRA Adapter Design

- **Adapter surfaces**
  - `core/lora/manager.LoRAInjectionManager` now wraps `wgp.setup_loras` with memoised discovery keyed on `(model_type, server_config_hash)`, exposing `hydrate`, `presets`, `resolve_preset`, `reset`, and `snapshot_state` helpers.
  - `core/prompt_enhancer/bridge.PromptEnhancerBridge` guards `setup_prompt_enhancer`/`process_prompt_enhancer`, caching primed configs per server snapshot and providing `prime`, `enhance`, `reset`, and `snapshot_state` entrypoints.
- **ProductionManager integration**
  - `ProductionManager.lora_manager()` / `.prompt_enhancer()` vend cached adapter instances; callers may inject shared adapters when constructing managers so queue controllers reuse discovery caches.
  - CLI `generate` hydrates LoRA listings via the adapter and reuses the same instances when building the runtime `ProductionManager`.
  - `run_generation()` now threads adapter payloads into `GenerationRuntime`, which updates queue state and server_config overrides before delegating to the legacy module.
- **TaskInputManager touchpoints**
  - `prepare_inputs_dict` and settings loaders now hydrate LoRA inventories through `lora_inventory()` before resolving selections, keeping metadata prep aligned with adapter state.
  - `build_lora_payload` and `resolve_prompt_enhancer` capture deterministic adapter snapshots for metadata, queue entries, and runtime execution.
- **State management**
  - LoRA discovery is cached in-memory only; `snapshot_state()` exposes counts for debug flags, while `reset()` clears caches for future CLI hooks. Adapter payloads flow through queue metadata so workers never touch `wgp.update_loras_url_cache`.
  - Prompt enhancer priming is tracked per server hash; `reset()` releases the cached models, and the bridge installs a primer callback so `wgp.load_models` primes the enhancer before `offload.profile` runs.
- **Implementation phases**
  1. (done) Adapter shims plus smoke tests landed (`tests/test_lora_manager.py`, `tests/test_prompt_enhancer_bridge.py`).
  2. (done) `ProductionManager`, `TaskInputManager`, and the CLI now depend on the adapters; `wgp` remains the execution backend for activation.
  3. (done) Runtime execution now consumes adapter payloads inside `wgp.generate_video`, eliminating the `state["loras"]` mutation and routing prompt enhancer priming through the bridge primer instead of direct `setup_prompt_enhancer` calls. Prompt enhancement is computed by `GenerationRuntime` via the bridge and delivered through `adapter_payloads`, so `wgp.generate_video` no longer invokes `process_prompt_enhancer` directly.
  4. (planned) Replace the remaining persistence helpers and inline enhancer utilities in `wgp` so the CLI runner can assume full control without touching legacy globals.

## ProductionManager Dependency Snapshot

- Runtime prep: relies on `wgp` for `wan_model`, `transformer_type`, `reload_needed`, and `load_models` to hydrate models on demand. Prompt enhancement sequences are expected to arrive via `adapter_payloads["prompt_enhancer"]`, allowing the legacy module to avoid touching enhancer globals during execution.
- Task inputs: still call through `wgp` helpers for model metadata (`get_model_record`, `get_model_family`, compatibility testers), settings writers, notifications, and the shared `lock`.
- Output persistence: `MediaPersistenceContext.save_video/save_image/save_audio/save_mask_archive` cover the legacy save sites while still leaning on `wgp` for filename allocation and a few inline helpers pending full extraction. Audio-only runs now save through `_save_audio_artifact` (wrapping `soundfile`) and RGBA mask archives respect the `save_masks` toggle.
