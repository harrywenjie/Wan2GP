# Headless Flow Appendix

This appendix captures slower-changing architectural references that support the headless CLI migration. Keep it updated when the overall shape of the system changes, but leave day-to-day notes in `PROJECT_PLAN_LIVE.md`.

## Architecture Overview (In Progress)
```
┌──────────────────────┐
│   CLI Frontends      │
│ (cli/generate.py,    │
│  cli/matanyone.py)   │
└──────────┬───────────┘
           │ argparse input
           ▼
┌──────────────────────┐
│ Config Assembly      │  ← consolidates defaults, presets, overrides
│ (assemble_generation │
│  _params, CLI args)  │
└──────────┬───────────┘
           │ normalized request
           ▼
┌──────────────────────┐
│ Core Runner (planned)│  ← queue mgmt, capability checks, logging hooks
│  • model resolver    │
│  • device scheduler  │
└──────────┬───────────┘
           │ generation job
           ▼
┌──────────────────────┐
│ Generation Pipelines │  ← existing `generate_video` families
│  (wan/flux/ltx/…)    │
└──────────┬───────────┘
           │ artifacts + status events
           ▼
┌──────────────────────┐
│ Progress Channel     │  ← replaces `gr.*` with structured logging / JSONL
│ (stdout/hooks)       │
└──────────┬───────────┘
           │ files + metadata
           ▼
┌──────────────────────┐
│ Output Writers       │  ← video/audio muxing, metadata emitters
└──────────────────────┘
```

**Current status:** `cli.generate` feeds directly into `wgp.generate_video()`. Configuration assembly lives in `assemble_generation_params()`, and the dedicated core runner extraction remains to be implemented.

**Follow-ups**
- [Done] Extract reusable config assembly so CLI/UI paths share one code path (`assemble_generation_params()`).
- [In Progress] Implement a queue-friendly runner module that exposes events over the progress channel without importing Gradio.
- [Pending] Ensure per-model hooks surface deterministic metadata for downstream automation (structured output manifests, JSONL logging).

## Reference Notes
- CLI flag surface: the canonical list lives in `docs/CLI.md`. Update that document when adding or removing arguments.
- Model footprint guidance: see `PROJECT_PLAN_LIVE.md` → `## Context And Findings` for the current keep/drop recommendations.
