# Work History

## Prior Summary
Wan2GP has been methodically reshaped from a Gradio-bound prototype into a deterministic CLI stack. Earlier sessions dissolved the UI surface, consolidated queue orchestration under `cli.queue_controller`, and anchored runtime configuration in `ProductionManager` so generation runs share adapter caches, metadata snapshots, and persistence templates. The prompt enhancer and LoRA subsystems now flow through memoised adapters with unit coverage, while MatAnyOne preprocessing clones the same `MediaPersistenceContext` and metadata state as the main generator to keep mask/audio exports aligned. Documentation in `docs/CLI.md`, `docs/APPENDIX_HEADLESS.md`, and `PROJECT_PLAN_LIVE.md` tracks these migrations, and regression suites cover queue payload propagation, adapter hydration, and MatAnyOne persistence semantics.

## 2025-11-01 (Session 4)
- Added a manifest pipeline for `cli.generate`: new `cli/manifest.py` records `MediaPersistenceContext` saves, computes canonical adapter hashes, and emits JSONL rows (defaulting to `<output_dir>/manifests/run_history.jsonl` with a `--manifest-path` override). `cli.generate` now wraps `ProductionManager.media_context()` with the recorder, hashes queued adapter payloads, and writes success/error entries without risking partial records.
- Refactored `cli.generate` execution flow to enqueue tasks explicitly, reuse the recorder output when assembling manifest artifacts, and persist metadata/adapter snapshots even on failures or user interrupts. Introduced `_build_manifest_artifacts` to normalise video/audio/mask captures and added `tests/test_cli_manifest_utils.py` to cover manifest assembly and adapter hashing.
- Tightened persistence guarantees by making `_save_video_artifact` and `_save_image_artifact` in `wgp.py` require a `MediaPersistenceContext`, raising `RuntimeError` when absent. Added `tests/test_generation_runtime_context.py` to pin the new contract via `GenerationRuntime`.
- Documented the CLI changes (`docs/CLI.md`) and updated architectural notes (`docs/CONTEXT.md`) to describe the recorder hook and the stricter persistence contract.
- Validation: `python -m unittest discover tests`
