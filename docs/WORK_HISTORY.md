# Work History

## Prior Summary
Wan2GP continues its evolution into a deterministic, headless CLI stack. Gradio-era surfaces are gone, queue orchestration now lives entirely under `cli.queue_controller`, and `ProductionManager` provides per-run metadata and persistence contexts so generation and preprocessing share adapters, templates, and logging. The new artifact manifest infrastructure records every `MediaPersistenceContext` save from `cli.generate`, while regression suites cover queue payload propagation, adapter hydration, and MatAnyOne persistence semantics.

## 2025-11-02 (Session 5)
- Extended manifest coverage to MatAnyOne: `cli.matanyone` now accepts `--manifest-path`, wraps the resolved `MediaPersistenceContext` with `ManifestRecorder`, and emits JSONL entries describing `mask_foreground`, `mask_alpha`, and optional `rgba_archive` artifacts (including request metadata and reproducibility inputs). Added `cli.manifest.resolve_manifest_path` and `build_matanyone_artifacts` helpers to share path logic with `cli.generate`, plus unit coverage (`tests/test_matanyone_manifest.py`) to assert role mapping, duration maths, and temp-file fallbacks.
- Refined MatAnyOne CLI execution flow: manifest emission handles dry runs and error cases, inputs are canonicalised into the manifest entry, and documentation (`docs/CLI.md`, `docs/APPENDIX_HEADLESS.md`) now reflects the headless manifest behaviour and new flag.
- Retired the legacy `wgp.save_video/save_image` wrappers and updated MatAnyOne fallbacks to call `core.io.media.write_video` directly. Cleaned up redundant imports, refreshed architectural notes in `docs/CONTEXT.md`, and queued follow-up work to finish deleting the remaining `shared.utils.audio_video` adapters.
- Validation: `python -m unittest tests.test_matanyone_persistence tests.test_matanyone_manifest`
