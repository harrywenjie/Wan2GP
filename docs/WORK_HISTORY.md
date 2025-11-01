# Work History

## Prior Sessions (through 2025-11-02)
- The project has been migrated from a Gradio-first application to a headless CLI: argument parsing now lives under `cli.arguments`, queue execution runs through `cli.queue_controller`, metadata and persistence flow through `core.io.media`, and MatAnyOne gained a headless wrapper. Gradio widgets, galleries, and plugins were deleted in favour of structured logging and deterministic CLI entry points. LoRA discovery and prompt enhancement were wrapped in adapters (`LoRAInjectionManager`, `PromptEnhancerBridge`), queue metadata now carries adapter payloads, and documentation/validation routines were kept in lock-step with each milestone.
- 2025-11-01 focused on threading adapter payloads through task preparation and queue execution so workers never touched legacy globals, with smoke coverage ensuring the queue harness handled the new metadata.
- 2025-11-02 taught `wgp.generate_video` to consume LoRA payloads directly, primed the prompt enhancer through the bridge, introduced CLI cache-reset flags, refreshed docs, and validated through `cli.generate`, `cli.queue_controller_smoke`, and targeted compile checks.

## 2025-11-03
- Added `MediaPersistenceContext.save_video/save_image` and updated `wgp.generate_video` to route all video/image persistence through `_save_video_artifact` / `_save_image_artifact`, keeping retry logic consistent while honouring per-run templates.
- Reworked `GenerationRuntime._apply_adapter_payloads` to precompute prompt enhancer output via `PromptEnhancerBridge.enhance`, thread enhanced prompts through `adapter_payloads`, and emit progress updates. `wgp.generate_video` now consumes the payload, logs bridge errors, updates metadata with enhanced prompts, and no longer references `prompt_enhancer_*` globals during execution.
- Introduced `cli.generate.reset_adapter_caches` to centralise cache resets and added `tests/test_cli_reset_flags.py` to assert the new helper exercises LoRA and prompt enhancer resets.
- Updated `core/io/media.py`, `core/production_manager.py`, `wgp.py`, and `cli/generate.py` to reflect the new persistence and adapter flow; refreshed `docs/CONTEXT.md` and `PROJECT_PLAN_LIVE.md` to capture the architecture changes and queued follow-up work.
- Validations: `python -m unittest tests.test_lora_manager tests.test_prompt_enhancer_bridge tests.test_cli_reset_flags`; `python -m cli.generate --prompt "smoke test prompt" --model-type t2v --dry-run --log-level INFO --reset-lora-cache --reset-prompt-enhancer`; `python -m cli.queue_controller_smoke`.
