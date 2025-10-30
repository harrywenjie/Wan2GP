# Changelog (Headless Era)

This changelog tracks the ongoing transition of Wan2GP from a Gradio-first application to a deterministic, command line video generator. Legacy release notes that referenced the web UI, plugin catalogue, or Hunyuan-derived features have been removed to avoid confusion.

## 2025-02-17
- Replaced the legacy documentation set with CLI-focused guidance (`README.md`, `docs/CLI.md`).
- Removed the plugin manual and associated references; the plugin subsystem has been fully excised from the codebase.
- Pruned historical release notes that referenced HyVideo/Hunyuan modules and other retired UI-exclusive features.
- Clarified the preferred entry point (`python -m cli.generate`) and highlighted `--dry-run` for reproducible configuration checks.

## 2025-02-14
- Introduced the standalone CLI wrapper (`cli/generate.py`) with argument validation and structured logging.
- Added a shared CLI argument builder (`cli/arguments.py`) to harmonize flag parsing between new tooling and the remaining internals.
- Audited model handlers to remove direct Gradio dependencies and route notifications through the CLI logger.
