# Command Line Interface

Wan2GP now ships with dedicated headless entry points for generation and preprocessing workflows:

- `python -m cli.generate` – wraps the video-generation routines from `wgp.py` while exposing reproducible CLI flags.
- `python -m cli.matanyone` – runs the MatAnyOne mask propagation pipeline against on-disk media.

## Video Generation (`cli.generate`)

`python -m cli.generate` is the primary gateway into Wan2GP's diffusion pipelines. It wraps the existing generation routines from `wgp.py` but exposes them with explicit, reproducible CLI flags.

```
python -m cli.generate \
  --prompt "dawn breaking over a neon coastline, wide aerial shot" \
  --model-type t2v \
  --frames 120 \
  --guidance-scale 6.5 \
  --output-dir outputs/sunrise
```

### Required Argument
- `--prompt TEXT` – the primary text prompt. Blank prompts are rejected.

### Prompt & Sampling Controls
- `--negative-prompt TEXT` – optional negative prompt; falls back to preset defaults when omitted.
- `--model-type NAME` – pipeline selector (`t2v`, `i2v_2_2`, etc.). The current defaults match the legacy UI model dropdowns.
- `--frames INT` – output length in frames. Defaults come from model presets.
- `--steps INT` – denoising steps; leave unset to reuse preset values.
- `--guidance-scale FLOAT` – sets all CFG guidance scales uniformly.
- `--prompt-enhancer {off,text,image,text+image}` – toggle the built-in prompt enhancer. `text` runs the LLM rewrite only, `image` captions the first reference frame, and `text+image` combines both. Overrides apply to the active run only; omit the flag to reuse the persisted default (usually disabled).
- `--prompt-enhancer-provider {llama3_2,joycaption}` – pick which backend to load when the enhancer is active. Defaults to `llama3_2`; requires `--prompt-enhancer`. Like other runtime toggles this selection is per-run unless written into `wgp_config.json` manually.
- Prompt enhancer priming is mediated by `core.prompt_enhancer.bridge.PromptEnhancerBridge`; the bridge caches loaded models per server configuration so repeat runs avoid gratuitous reloads. Pass `--reset-prompt-enhancer` when you need to flush the cache (and unload the enhancer models) before a run.
- Metadata snapshots include `adapter_payloads["prompt_enhancer"]` describing the selected mode/provider so queue workers and post-processing tools can reconstruct enhancer state without re-reading `wgp` globals.
- Queue entries and saved task metadata now persist the computed `enhanced_prompt` alongside the enhancer payload; automated smoke coverage exercises the queue controller to guard this propagation path.
- `--seed INT` – fixed seed for deterministic runs. Use `-1` to request a random seed from the pipeline.
- `--force-fps {auto,control,source,INT}` – override the output frame rate or reuse preset behaviour.

### LoRA Configuration
- `--list-loras` / `--list-lora-presets` – inspect the available LoRA weights or preset bundles for the active model family and exit immediately.
- `--loras NAME` – activate a LoRA by file name. Repeat the flag to layer multiple weights (names follow the files under `models/<model>/loras/`).
- `--lora-multipliers STRING` – supply an explicit multiplier string (same syntax as legacy presets) to override preset defaults.
- `--lora-preset NAME` – load a `.lset`/`.json` preset from the model’s LoRA directory. The preset merges with any `--loras` and multiplier overrides supplied on the CLI.

LoRA discovery now flows through `core.lora.manager.LoRAInjectionManager`, so repeated listings reuse cached directory scans keyed on `(model_type, server_config_hash)`. Cache resets remain automatic when configuration changes, and `--reset-lora-cache` forces a fresh directory scan ahead of the next run.

Every queued task records a deterministic adapter payload (`metadata["adapter_payloads"]["lora"]`) that captures the discovery hash, available weights/presets, activated selections, and multiplier string. The queue controller forwards this payload to the worker so LoRA activation never re-queries `wgp` during execution.

Macro-based prompt assembly and the legacy “wizard” surface have been removed; author prompts directly or script your own templating before invoking the CLI.

### File-Based Inputs
Every path must reference an existing file; the CLI validates before execution.
- `--image-start PATH` – kick-off image for i2v workflows.
- `--image-end PATH` – ending reference frame.
- `--image-ref PATH` – repeatable flag for additional reference images.
- `--video-source PATH` – source video to extend or edit.
- `--video-guide PATH` – motion/style guide video.
- `--video-mask PATH` / `--image-mask PATH` – masks that constrain edits.
- `--image-guide PATH` – conditioning image.
- `--audio-guide PATH`, `--audio-guide2 PATH` – audio conditioning tracks.
- `--audio-source PATH` – original audio to preserve or remix.
- `--settings-file PATH` – preload defaults from a saved Wan2GP settings JSON or media file with embedded metadata. The CLI merges these values before applying explicit flags and requires the file to target the active `--model-type`.

### Output Control
- `--output-dir PATH` – directory where rendered assets are written. Overrides are per-run; each execution falls back to the configured default (`save_path` inside `wgp.py`) unless the flag is provided again.

### Metadata Persistence
- `--metadata-mode {metadata,json}` – per-run override for how metadata is emitted. `metadata` embeds structured info back into the rendered media, while `json` writes sidecar manifests next to each artifact. Omit the flag to reuse the persisted default.
- Generation runs continue to respect `wgp_config.json -> metadata_type` when `--metadata-mode` is not supplied.
- When `metadata_type` is left at `metadata`, enabling `embed_source_images` preserves reference frames inside the video metadata bundle (the CLI injects this automatically via `ProductionManager`).
- MatAnyOne exposes the same `--metadata-mode` toggle: the CLI clones `ProductionManager.metadata_state()` so the foreground/alpha writers reuse the generation metadata templates. If the manager cannot be initialised MatAnyOne falls back to the default template set before emitting embedded metadata or JSON sidecars.

### Runtime Controls
- Unless stated otherwise, runtime toggles act on the current execution only. Persistent defaults continue to originate from `wgp_config.json`; adjust that file directly if you need new baseline behaviour.
- `--attention {auto,sdpa,sage,sage2,flash,xformers}` – select the attention backend; falls back to the configured default when omitted.
- `--compile` – enable the torch.compile transformer path (equivalent to setting `server_config["compile"]` to `"transformer"`).
- `--profile INT` – override the VRAM/profile budget used during model initialisation (matches the legacy profile dropdown). The CLI applies this override for the current run only.
- `--preload INT` – preload diffusion weights into VRAM (in megabytes). Use `0` to rely on profile defaults.
- `--fp16` / `--bf16` – force the transformer weights to the requested dtype for this run. Only one may be active at a time.
- `--transformer-quantization TEXT` – override transformer quantisation (e.g. `int8`, `fp8`, `none` for full precision).
- `--text-encoder-quantization TEXT` – override text encoder quantisation.
- `--tea-cache-level FLOAT` – enable TeaCache skipping with the provided multiplier (>0).
- `--tea-cache-start-perc FLOAT` – start TeaCache skipping at the provided percentage of the denoising schedule. Requires `--tea-cache-level`.
- `--save-masks` / `--no-save-masks` – mirror the legacy “save masks” toggle so CLI runs can persist or skip intermediate mask exports for the current execution only (defaults fall back to stored preferences). The toggle now also governs RGBA ZIP archives emitted from BGRA mask frames via the shared media context.
- `--save-quantized` / `--no-save-quantized` – control whether freshly quantised transformer weights are written back to disk after generation (current run only).
- `--save-speakers` / `--no-save-speakers` – enable or disable persistence of extracted speaker tracks used by MMAudio/Chatterbox flows (current run only).
- `--check-loras` / `--no-check-loras` – force a pre-flight audit of LoRA files (exiting early on missing assets) or bypass the check even when stored defaults enable it; changes apply to the active run only.

### Logging & Utility Flags
- `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` – adjust CLI logging verbosity (default `INFO`).
- `--dry-run` – resolve configuration, print the derived parameters, and exit without generating frames. Use this to validate arguments, file discovery, and preset merges.
- `--control-port INT` – expose a lightweight TCP control server that accepts pause/resume/status commands while the queue controller is running.
- `--control-host TEXT` – host/interface bound by the TCP control server (defaults to `127.0.0.1`; change with care if remote access is required).

### Execution Flow
1. The CLI parser gathers arguments (see `cli/arguments.py`).
2. File inputs are expanded and validated. Missing or non-file paths terminate the run with clear errors.
3. `wgp.py` is imported, triggering `wgp.ensure_runtime_initialized()` to load `wgp_config.json`, normalise settings directories, and seed the legacy defaults before any CLI overrides are applied.
4. The command constructs a lightweight state object and forwards parameters into `wgp.generate_video`.
5. Progress, status, and output notifications are logged to the terminal. The final output path is echoed on success.

### Artifact Manifest Specification
`cli.generate` writes a structured manifest once a run completes so schedulers and audit scripts can consume results without scraping logs. Unless overridden with `--manifest-path`, each invocation appends a single JSON object to `<output_dir>/manifests/run_history.jsonl`. Each line is UTF-8 JSON with sorted keys and the following schema:

- `run_id` – UUIDv4 assigned when the run starts.
- `timestamp` – ISO-8601 timestamp (UTC) recorded at completion.
- `output_dir` – absolute path resolved before generation begins.
- `metadata_mode` – effective metadata mode (`metadata` or `json`) after CLI overrides are applied.
- `adapter_payload_hashes` – dictionary mapping adapter names to `{ "sha256": "<hex>", "source_bytes": <int> }`. Hashes are computed from the canonical JSON serialisation (`json.dumps(payload, sort_keys=True, separators=(",", ":"))`) encoded as UTF-8. `source_bytes` records the byte length of that serialisation for quick diagnostics.
- `artifacts` – list of objects describing every saved file. Each entry captures: `role` (`foreground`, `alpha`, `audio`, `mask_archive`, etc.), `path` (absolute string), `container` (file container/extension), `codec` (video or audio codec when known), `frames` (integer when applicable), `duration_s` (float, optional), and `metadata_sidecar` (path to JSON/embedded indicator, `null` when embedded metadata was written). Future writers may extend entries with additional keys; consumers must ignore unknown fields.
- `inputs` – object capturing resolved CLI arguments that materially affect reproducibility (prompt text, seeds, frame counts, model identifiers, and any runtime overrides that were applied). This section mirrors what `tests/test_queue_prompt_payloads.py` asserts today so queue snapshots and manifest entries stay aligned.
- `status` – `"success"` or `"error"`. Failures capture `error` (string message) and omit `artifacts`.

The manifest writer must flush the JSON line only after persistence succeeds. Dry runs skip manifest emission entirely. MatAnyOne will reuse the same format once its pipeline emits manifests; its `artifacts` list uses `mask_foreground`, `mask_alpha`, and `rgba_archive` roles.

### Tips
- Treat `--dry-run` as your first step for any new command to ensure paths and overrides resolve correctly.
- Use dedicated output directories (`--output-dir`) when batching experiments so results are easy to compare.
- Combine shell scripts or job schedulers with the CLI to queue repeatable generations—no plugin hooks or UI callbacks remain.

### Queue Automation
- The headless build no longer supports importing or autosaving Gradio queue archives (`queue.zip`). All queue load/save handlers have been removed from `wgp.py`.
- Script repeated generations by calling `python -m cli.generate` in loops or job schedulers; the in-memory queue is only managed for the active process.
- `clear_queue` now affects the live queue state only. The CLI never writes auxiliary queue artifacts to disk, keeping runs deterministic.
- `python -m cli.queue_controller_smoke` runs a lightweight pause/resume smoke test for the headless queue controller without invoking the full generation pipeline.
- `python -m cli.queue_control --port <PORT> pause|resume|status|abort` sends control commands to a generation process started with `--control-port <PORT>`. Commands are processed synchronously and responses are returned as plain text (JSON for `status`).

### Queue Control
- Start the controller with `--control-port` to expose a local TCP endpoint (default host `127.0.0.1`). The CLI logs the active port once the listener is ready.
- Use `python -m cli.queue_control --port <PORT> pause` to pause the active generation, `resume` to continue, and `status` to retrieve a JSON summary (paused state, in-progress flag, queue length, and last progress status). `abort` signals an abort request.
- The control channel is intentionally simple: single-line commands, single-line responses. Communication is unencrypted; keep the listener bound to loopback unless you front it with your own secure tunnel.

## MatAnyOne Mask Propagation (`cli.matanyone`)

`python -m cli.matanyone` runs the headless MatAnyOne mask propagation pipeline. It accepts a source video or image plus a template mask and emits foreground/alpha MP4s (with optional RGBA archives for compositing workflows).

```
python -m cli.matanyone \
  --input assets/demo.mp4 \
  --template-mask assets/demo_mask.png \
  --output-dir mask_outputs/demo \
  --mask-type greenscreen \
  --dry-run
```

### Required Arguments
- `--input PATH` – source media file (video or still image). The CLI validates existence and type before continuing.
- `--template-mask PATH` – grayscale template mask aligned to the first processed frame.

### Optional Controls
- `--output-dir PATH` – destination directory for generated assets (defaults to `mask_outputs/`).
- `--start-frame INT` / `--end-frame INT` – clamp the processed frame window (end is exclusive; omit to process the full clip).
- `--new-dim SPEC` – resize directive understood by the MatAnyOne pipeline (e.g. `1080p outer`).
- `--matting {foreground,background}` – choose whether the mask represents the foreground (default) or background region.
- `--mask-type {wangp,greenscreen,alpha}` – select the output format: raw mask pair, composited greenscreen, or an RGBA ZIP bundle (saved only when the active `server_config` enables `save_masks`).
- `--erode-kernel INT` / `--dilate-kernel INT` – morphology kernel sizes to refine the mask.
- `--warmup-frames INT` – warm-up frame count for stabilising propagation (default `10`).
- `--device TEXT` – execution device (e.g. `cuda`, `cuda:1`, `cpu`).
- `--codec TEXT` – override the video codec used by the persistence context (defaults to the configured `server_config.video_output_codec`, falling back to `libx264_8`).
- `--metadata-mode {metadata,json}` – choose whether the foreground/alpha MP4s embed metadata or emit JSON sidecars. Defaults to `metadata`.
- `--no-audio` – skip audio track reattachment when the source includes audio.

When `wgp` is available, the CLI requests `ProductionManager.metadata_state()` (optionally honouring `--metadata-mode`) and forwards the snapshot to the preprocessing pipeline. This keeps MatAnyOne aligned with the primary generation path when cloning metadata config templates or switching to JSON sidecars. If `wgp` fails to initialise the pipeline still runs using the stock metadata defaults.

### Logging & Dry Runs
- `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` – matches the generation CLI logger configuration.
- `--dry-run` – validate paths, construct the propagation request, log the resolved configuration, and exit without running the model.

### Outputs
Successful runs log the written file paths and echo them to STDOUT. Expect:
- Foreground video (`<prefix><mask-suffix>.<container>`) using the container configured in `server_config` (defaults to `mp4`).
- Alpha companion (`<prefix>_alpha.<container>`), persisted with the same codec/container defaults.
- Optional RGBA ZIP archive when `--mask-type alpha` is selected **and** `save_masks` is enabled in the active `MediaPersistenceContext`.

MatAnyOne now threads the per-run `MediaPersistenceContext` supplied by `ProductionManager`. Video writes honour container/codec overrides (with `--codec` acting as a per-run override), audio tracks are reattached onto the context-derived container, and mask archives respect the `save_masks` toggle so debug bundles only materialise when explicitly configured. Persistence helpers retain retry logging so automation can detect and react to IO errors deterministically.

For architectural notes and migration history consult `PROJECT_PLAN_LIVE.md`. External scripts may reuse the bootstrap directly via `import wgp; wgp.ensure_runtime_initialized()` before calling lower level helpers.
