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
│ Runtime Bootstrap    │  ← `wgp.initialize_runtime` / `ensure_runtime_initialized`
│  • load wgp_config   │
│  • normalise dirs    │
│  • seed defaults     │
└──────────┬───────────┘
           │ hydrated config
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

Runtime bootstrap now lives in `wgp.initialize_runtime()`, which loads `wgp_config.json`, migrates legacy settings, and prepares the global defaults. CLI frontends call `wgp.ensure_runtime_initialized()` automatically before applying runtime overrides, and custom scripts should do the same when using lower-level helpers.

**Follow-ups**
- [Done] Extract reusable config assembly so CLI/UI paths share one code path (`assemble_generation_params()`).
- [In Progress] Implement a queue-friendly runner module that exposes events over the progress channel without importing Gradio.
- [Pending] Ensure per-model hooks surface deterministic metadata for downstream automation (structured output manifests, JSONL logging).
- [Done] `ProductionManager.metadata_state()` now clones per-run metadata templates so CLI generation and MatAnyOne toggle between embedded metadata and JSON sidecars without mutating `wgp`.

## Reference Notes
- CLI flag surface: the canonical list lives in `docs/CLI.md`. Update that document when adding or removing arguments.
- Model footprint guidance: see `docs/CONTEXT.md` → `# Detailed Context` for the current keep/drop recommendations.
- Adapter shims: `core/lora/manager.LoRAInjectionManager` and `core/prompt_enhancer/bridge.PromptEnhancerBridge` provide cached discovery/priming layers. Use their `snapshot_state()` helpers when debugging cache contents and prefer `reset()` over mutating the legacy globals directly.

## Queue Control Harness
- `cli.queue_controller.QueueController` is the default queue orchestrator; the legacy `wgp.process_tasks` loop has been removed along with the `--legacy-queue` escape hatch.
- Shared queue helpers (`clear_queue_action`, `generate_queue_summary`, `update_queue_data`) live in `cli.queue_utils`, keeping the CLI controller and `wgp` wrapper aligned.
- `cli.queue_controller_smoke.run_smoke()` now provisions a temporary `QueueControlServer`, drives the pause/resume/status commands over TCP, and asserts the queue controller returns to the active state; keep this harness in the CI smoke suite to guard the control channel.
- Default bindings intentionally stay on `127.0.0.1`; only expose the control port beyond loopback when you can wrap it with SSH tunnels or other authenticated transport.
- Use `cli.queue_control.send_command()` (or the CLI wrapper) to script operational checks or to integrate the control port into higher-level automation once Celery workers are in play.
