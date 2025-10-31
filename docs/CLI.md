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
- `--prompt-enhancer {off,text,image,text+image}` – toggle the built-in prompt enhancer. `text` runs the LLM rewrite only, `image` captions the first reference frame, and `text+image` combines both. When omitted the CLI reuses stored defaults (usually disabled).
- `--prompt-enhancer-provider {llama3_2,joycaption}` – pick which backend to load when the enhancer is active. Defaults to `llama3_2`; requires `--prompt-enhancer`.
- `--seed INT` – fixed seed for deterministic runs. Use `-1` to request a random seed from the pipeline.
- `--force-fps {auto,control,source,INT}` – override the output frame rate or reuse preset behaviour.

### LoRA Configuration
- `--list-loras` / `--list-lora-presets` – inspect the available LoRA weights or preset bundles for the active model family and exit immediately.
- `--loras NAME` – activate a LoRA by file name. Repeat the flag to layer multiple weights (names follow the files under `models/<model>/loras/`).
- `--lora-multipliers STRING` – supply an explicit multiplier string (same syntax as legacy presets) to override preset defaults.
- `--lora-preset NAME` – load a `.lset`/`.json` preset from the model’s LoRA directory. The preset merges with any `--loras` and multiplier overrides supplied on the CLI.

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
- `--output-dir PATH` – directory where rendered assets are written. When omitted the generator uses the configured default (`save_path` inside `wgp.py`).

### Runtime Controls
- `--attention {auto,sdpa,sage,sage2,flash,xformers}` – select the attention backend; falls back to the configured default when omitted.
- `--compile` – enable the torch.compile transformer path (equivalent to setting `server_config["compile"]` to `"transformer"`).
- `--profile INT` – override the VRAM/profile budget used during model initialisation (matches the legacy profile dropdown).
- `--preload INT` – preload diffusion weights into VRAM (in megabytes). Use `0` to rely on profile defaults.
- `--fp16` / `--bf16` – force the transformer weights to the requested dtype for this run. Only one may be active at a time.
- `--transformer-quantization TEXT` – override transformer quantisation (e.g. `int8`, `fp8`, `none` for full precision).
- `--text-encoder-quantization TEXT` – override text encoder quantisation.
- `--tea-cache-level FLOAT` – enable TeaCache skipping with the provided multiplier (>0).
- `--tea-cache-start-perc FLOAT` – start TeaCache skipping at the provided percentage of the denoising schedule. Requires `--tea-cache-level`.
- `--save-masks` / `--no-save-masks` – mirror the legacy “save masks” toggle so CLI runs can persist or skip intermediate mask exports for the current execution only (defaults fall back to stored preferences).
- `--save-quantized` / `--no-save-quantized` – control whether freshly quantised transformer weights are written back to disk after generation (current run only).
- `--save-speakers` / `--no-save-speakers` – enable or disable persistence of extracted speaker tracks used by MMAudio/Chatterbox flows (current run only).
- `--check-loras` / `--no-check-loras` – force a pre-flight audit of LoRA files (exiting early on missing assets) or bypass the check even when stored defaults enable it; changes apply to the active run only.

### Logging & Utility Flags
- `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` – adjust CLI logging verbosity (default `INFO`).
- `--dry-run` – resolve configuration, print the derived parameters, and exit without generating frames. Use this to validate arguments, file discovery, and preset merges.

### Execution Flow
1. The CLI parser gathers arguments (see `cli/arguments.py`).
2. File inputs are expanded and validated. Missing or non-file paths terminate the run with clear errors.
3. `wgp.py` is imported, triggering `wgp.ensure_runtime_initialized()` to load `wgp_config.json`, normalise settings directories, and seed the legacy defaults before any CLI overrides are applied.
4. The command constructs a lightweight state object and forwards parameters into `wgp.generate_video`.
5. Progress, status, and output notifications are logged to the terminal. The final output path is echoed on success.

### Tips
- Treat `--dry-run` as your first step for any new command to ensure paths and overrides resolve correctly.
- Use dedicated output directories (`--output-dir`) when batching experiments so results are easy to compare.
- Combine shell scripts or job schedulers with the CLI to queue repeatable generations—no plugin hooks or UI callbacks remain.

### Queue Automation
- The headless build no longer supports importing or autosaving Gradio queue archives (`queue.zip`). All queue load/save handlers have been removed from `wgp.py`.
- Script repeated generations by calling `python -m cli.generate` in loops or job schedulers; the in-memory queue is only managed for the active process.
- `clear_queue` now affects the live queue state only. The CLI never writes auxiliary queue artifacts to disk, keeping runs deterministic.

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
- `--mask-type {wangp,greenscreen,alpha}` – select the output format: raw mask pair, composited greenscreen, or RGBA ZIP bundle.
- `--erode-kernel INT` / `--dilate-kernel INT` – morphology kernel sizes to refine the mask.
- `--warmup-frames INT` – warm-up frame count for stabilising propagation (default `10`).
- `--device TEXT` – execution device (e.g. `cuda`, `cuda:1`, `cpu`).
- `--codec TEXT` – FFmpeg codec string used when writing MP4 outputs (default `libx264_8`).
- `--no-audio` – skip audio track reattachment when the source includes audio.

### Logging & Dry Runs
- `--log-level {CRITICAL,ERROR,WARNING,INFO,DEBUG}` – matches the generation CLI logger configuration.
- `--dry-run` – validate paths, construct the propagation request, log the resolved configuration, and exit without running the model.

### Outputs
Successful runs log the written file paths and echo them to STDOUT. Expect:
- Foreground MP4 (`<prefix>.mp4`)
- Alpha MP4 (`<prefix>_alpha.mp4`)
- Optional RGBA ZIP archive when `--mask-type alpha` is selected.

For architectural notes and migration history consult `PROJECT_PLAN_LIVE.md`. External scripts may reuse the bootstrap directly via `import wgp; wgp.ensure_runtime_initialized()` before calling lower level helpers.
