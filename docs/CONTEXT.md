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
  - `core/lora/manager.py` will expose a `LoRAInjectionManager` class with two public entrypoints: `hydrate(transformer, *, requested: Sequence[str], multipliers: Mapping[str, float])` for run-scoped activation and `presets()` for preset discovery. The manager hides filesystem discovery, preset parsing, cache refresh, and the existing `setup_loras`/`extract_preset` helpers. Internally it keeps a `LoRALibrary` cache keyed by `(model_type, lora_dir)` so repeated CLI runs avoid redundant globbing and metadata reads.
  - `core/prompt_enhancer/bridge.py` introduces `PromptEnhancerBridge` with lifecycle hooks: `prime(request: PromptEnhancerSpec)`, `enhance(prompts: List[str], context: PromptEnhancerContext)` and `reset()`. The bridge chooses provider backends (local Florence, remote API, noop) based on `server_config`, encapsulating the enhancer bootstrap that currently lives in `wgp.setup_prompt_enhancer`.
- **ProductionManager integration**
  - `ProductionManager` will vend cached instances through `production_manager.lora_manager()` and `production_manager.prompt_enhancer()`. Instances are keyed by the effective `server_config` snapshot so CLI queue workers share caches but isolate per-run overrides (e.g. temporary LoRA directories).
  - `GenerationRuntime` will request adapters during initialisation and thread them to `TaskInputManager` so queue serialization no longer imports `wgp` for preset metadata or enhancer defaults.
- **TaskInputManager touchpoints**
  - The manager gains `build_lora_payload(request: TaskPayload)` to translate CLI arguments into adapter calls and `resolve_prompt_enhancer(request: TaskPayload)` to fetch enhancer settings. Both methods return serialisable payloads so queue summaries and metadata writers stay deterministic.
  - Existing helpers like `prepare_inputs_dict` will consume adapter-derived payloads rather than re-reading lora directories or enhancer configs.
- **State management**
  - LoRA caches persist in-memory only; a `LoRAState` dataclass tracks discovered files, preset manifests, and multiplier defaults for diagnostics. The adapter exposes `snapshot_state()` for logging/debug CLI flags.
  - Prompt enhancer bridge stores a `PromptEnhancerRuntime` object (models/tokenizers/processors) behind a lazy loader so GPU memory is allocated only when an enhancer-enabled run is requested. `reset()` frees resources and clears the cached runtime.
- **Implementation phases**
  1. Land the adapter modules with shim implementations that call into the existing `wgp` helpers, plus unit-style smoke tests that validate caching behaviour with temporary directories.
  2. Teach `ProductionManager` and `TaskInputManager` to depend on the new adapters, keeping `wgp.setup_loras`/`wgp.setup_prompt_enhancer` as temporary fallbacks for legacy callers.
  3. Remove the fallback paths and delete the old helpers once CLI adoption is complete, documenting the transition in `docs/APPENDIX_HEADLESS.md`.

## ProductionManager Dependency Snapshot

- Runtime prep: relies on `wgp` for `wan_model`, `transformer_type`, `reload_needed`, and `load_models` to hydrate models on demand.
- Task inputs: still call through `wgp` helpers for model metadata (`get_model_record`, `get_model_family`, compatibility testers), settings writers, notifications, and the shared `lock`.
- Output persistence: now uses `core.io.get_available_filename` but still leans on `wgp` for some inline save helpers; further IO extraction should remove these touchpoints.
