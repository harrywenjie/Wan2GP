# Detailed Context

## Orchestration & Queue

- `wgp.py` still executes the core generation path. The CLI reaches it through `core.production_manager.ProductionManager.run_generation()`, which handles notifier injection, metadata wiring, and model reload checks before delegating to `wgp.generate_video()`.
- `cli.queue_controller.QueueController` is the primary queue runner (AsyncStream-backed). It owns abort/pause handling, drains queued tasks sequentially, and surfaces pause/resume/status/abort over the optional TCP control server (`cli.queue_control_server` + `cli.queue_control`).
- Queue state lives under `cli.queue_state.QueueStateTracker`. Helper functions for queue summaries, abort/clear toggles, and counter resets live in `cli.queue_utils` / `cli.queue_state`, with `wgp.clear_queue_action` maintained only for legacy compatibility.
- `cli.runner` provides the CLI notifier (`CLIGenerationNotifier`), progress callback builder, and `send_cmd` bridge so telemetry updates stay in sync with `state["gen"]`. `cli.generate` traps `KeyboardInterrupt` and clears the queue via these helpers.

## MatAnyOne Pipeline

- `preprocessing/matanyone/app.py` is a headless pipeline that assumes on-disk source media/masks and GPU availability.
- `cli/matanyone.py` wraps the pipeline with logging, input validation, frame/mask/audio controls, optional dry-run mode, and forwards requests to `generate_masks` using the shared CLI notifier. When `wgp` is available the CLI clones `ProductionManager.metadata_state()` so MatAnyOne writes metadata with the same templates as the main generation path. Outputs land under `mask_outputs/` with optional RGBA ZIP bundles and audio reattachment when requested.

## Metadata & IO

- `core/io/media.py` houses media persistence (`write_video`, `write_image`) plus `write_metadata_bundle`, all with logger-aware retry handling; `shared.utils.audio_video` is now a thin adapter that injects the CLI notifications logger.
- `ProductionManager.metadata_state()` returns a per-run `MetadataState` snapshot (choice + cloned templates). `GenerationRuntime` forwards the snapshot to `wgp.generate_video`, replacing the old module-level `metadata_choice` / `metadata_configs`.
- `_resolve_metadata_config` accepts either the dataclass or a dict for backward compatibility. CLI runs still honour `--metadata-mode`; MatAnyOne now passes the cloned `MetadataState` snapshot directly into its writers so embedded metadata and JSON sidecars stay aligned with the core generation pipeline.
- Embedded source images for metadata remain gated by `server_config["embed_source_images"]`; JSON sidecars are emitted when `metadata_mode=json`.
- `core.io.media.MediaPersistenceContext` now captures the video/image persistence templates plus the legacy `save_masks` debug toggle. `ProductionManager.media_context()` builds a fresh instance from `server_config` for each run and passes it through `GenerationRuntime` to `wgp.generate_video` (the legacy module ignores the payload for now, pending call-site refactors).

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
- **TaskInputManager touchpoints**
  - `prepare_inputs_dict` and settings loaders now hydrate LoRA inventories through `lora_inventory()` before resolving selections, keeping metadata prep aligned with adapter state.
  - Follow-up work will add `build_lora_payload` / `resolve_prompt_enhancer` helpers so queue serialization records adapter payloads instead of raw `wgp` lookups.
- **State management**
  - LoRA discovery is cached in-memory only; `snapshot_state()` exposes counts for debug flags, while `reset()` clears caches for future CLI hooks.
  - Prompt enhancer priming is tracked per server hash; `reset()` releases models via the legacy helper until the runtime extraction lands.
- **Implementation phases**
  1. (done) Adapter shims plus smoke tests landed (`tests/test_lora_manager.py`, `tests/test_prompt_enhancer_bridge.py`).
  2. (done) `ProductionManager`, `TaskInputManager`, and the CLI now depend on the adapters; `wgp` remains the execution backend for activation.
  3. (pending) Remove the fallback paths and delete the old helpers once CLI adoption is complete, documenting the transition in `docs/APPENDIX_HEADLESS.md`.

## ProductionManager Dependency Snapshot

- Runtime prep: relies on `wgp` for `wan_model`, `transformer_type`, `reload_needed`, and `load_models` to hydrate models on demand.
- Task inputs: still call through `wgp` helpers for model metadata (`get_model_record`, `get_model_family`, compatibility testers), settings writers, notifications, and the shared `lock`.
- Output persistence: now uses `core.io.get_available_filename` but still leans on `wgp` for some inline save helpers; further IO extraction should remove these touchpoints.
