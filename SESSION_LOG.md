# Session Log

## Prior Summary
Wan2GP has been progressively retooled into a headless, CLI-first workflow. `core.production_manager.ProductionManager.run_generation()` now fronts every generation path, clones metadata/persistence contexts per run, and hands work to `wgp.generate_video` only after adapters, manifests, and notifier wiring are prepared. Queue control lives entirely under `cli/`, with `QueueController` orchestrating execution, a TCP control server mirroring the old UX, and state rebuilding handled by `cli.queue_state.QueueStateTracker` so telemetry stays deterministic across runs.

Media persistence has been consolidated inside `core/io/media.py`. `MediaPersistenceContext` vends retry-aware `save_*` helpers that honour codec/container overrides, while `write_video`/`write_image` expose logger-aware primitives for callers outside the Production Manager. Legacy `shared.utils.audio_video.save_*` shims have been retired, and MatAnyOne now persists foreground/alpha/videos, RGBA archives, and audio tracks through the shared context, complete with JSON sidecars and manifest emission synchronized with CLI logging.

MatAnyOne preprocessing mirrors these headless patterns: it decodes audio via ffmpeg, persists through the shared context, records per-track metadata, and updates queue summaries plus TCP status payloads with `audio_tracks` entries. Smoke tests (`cli.queue_controller_smoke`, `tests/test_queue_audio_metadata.py`, `tests/test_matanyone_cli_integration.py`) lock in audio/manifest semantics, while queue prompt payload coverage guards downstream automation hooks. Prompt enhancer and LoRA flows now run through dedicated adapters so the CLI controls priming and payload collection without reviving legacy globals.

Documentation and planning remain in sync: `PROJECT_PLAN_LIVE.md`, `docs/CLI.md`, and `docs/APPENDIX_HEADLESS.md` track the headless milestones, while `LIVE_CONTEXT.md` captures orchestration, persistence, and adapter design notes. Active roadmap items focus on peeling remaining GUI assumptions, finalising disk-first workflows, and emitting machine-readable manifests for every CLI entrypoint.

## 2025-11-02 (Session 14)
- Replaced the direct `imageio.get_writer` usage in `preprocessing/dwpose.save_one_video` with a `VideoSaveConfig`-backed call into `core.io.media.write_video`, preserving macro-block and quality overrides while routing diagnostics through the shared logger stack.
- Updated `PROJECT_PLAN_LIVE.md`, `LIVE_CONTEXT.md`, and `docs/IO_MEDIA_MIGRATION.md` to record the dwpose persistence change and refreshed `## Immediate Next Actions` to focus on threading contexts through remaining call sites.
- Validation: `python -m py_compile preprocessing/dwpose/pose.py`.
