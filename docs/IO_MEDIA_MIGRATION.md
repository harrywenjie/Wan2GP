# Media Persistence Refactor Plan

## Objective
Relocate the media output helpers (`save_video`, `save_image`, and the associated metadata writers) from `shared.utils.audio_video` into a dedicated `core/io/media.py` module so the CLI stack can control persistence through explicit dependencies instead of `wgp` globals.

## Target Module Layout
- `core/io/media.py`
  - `VideoSaveConfig` dataclass encapsulating FPS, codec, container, retry, pixel format hints, and temporary directory policy.
  - `ImageSaveConfig` dataclass covering grid layout, normalization range, quality/format mapping, and retry behaviour.
  - `MetadataSaveConfig` dataclass pointing to format-specific metadata writers plus toggleable validation.
  - `write_video_frames(frames, config, *, logger, output_path=None)` returning the resolved path.
  - `write_image_tensor(tensor, config, *, logger, output_path)` returning the resolved path.
  - `write_metadata_bundle(path, metadata, config, *, logger)` orchestrating image/video/audio metadata writes.
  - Thin adapters for tensor → numpy conversion so callers can pass torch tensors or numpy arrays.

## Dependency Injection
- Callers provide `logger: logging.Logger` (or a callback) to replace the current `print`-based diagnostics.
- `ProductionManager` constructs the config dataclasses from `server_config` (`video_output_codec`, `video_container`, `image_output_codec`) and injects them into queue/runner flows.
- MatAnyOne supplies request-specific codec overrides by cloning the shared config with `.with_codec(request.codec)`.
- CLI runners retain control over retry counts and temporary directory usage via optional CLI flags.

## Compatibility
- Preserve `save_video` return semantics (string path) and argument surface by providing shims inside `shared.utils.audio_video` that delegate to the new helpers during a transition period.
- Maintain RGBA-aware image saving and `.temp` extension handling by porting the existing logic into `write_image_tensor`.
- Ensure metadata writers still support PNG/JPEG/WebP (image), MP4/MKV (video), and WAV/FLAC (audio) without behavioural regressions.

## Logging & Error Handling
- Replace bare `print` statements with injected logger calls (`logger.error`, `logger.warning`) while keeping message contents for parity.
- Surface retry exhaustion via `MediaWriteError` exceptions so CLI callers can respond deterministically (abort task or continue without optional assets).

## Migration Steps
1. **Introduce module and dataclasses** *(Done)* – `core/io/media.py` now owns the config dataclasses and implements `write_video`/`write_image` with the legacy preprocessing, retry, and codec handling plus logger hooks.
2. **Implement compatibility shims** *(Done)* – `shared.utils.audio_video.save_video/save_image` are thin adapters that construct configs, default to the notifications logger, and delegate to the core helpers; `wgp` and MatAnyOne inject the logger explicitly.
3. **Migrate metadata writers** *(Done)* – `write_metadata_bundle` now dispatches to the legacy `save_*_metadata` helpers, `shared.utils.audio_video` delegates through logger-aware shims, and both `wgp` and MatAnyOne emit metadata via `MetadataSaveConfig`.
4. **Retire legacy metadata helpers** *(Done)* – removed the `shared.utils.audio_video.save_image_metadata` shim and tightened imports so the bundler routes through the dedicated media modules.
5. **Document the new surfaces** *(Pending)* – refresh `docs/CLI.md`, `docs/APPENDIX_HEADLESS.md`, and this plan after the metadata migration to capture the final CLI workflow.

## Dependency Map
- **`core.io.media.write_video`**
  - Accepts torch tensors or iterables of HWC `uint8` frames, performs grid assembly via `torchvision`, clamps/normalises values, and writes with `imageio` using merged codec parameters.
  - Retries are logged through the injected logger; returns `None` after exhausting attempts to preserve legacy semantics.
- **`core.io.media.write_image`**
  - Operates on torch tensors, converts RGBA tensors to PNG automatically, and uses PIL or `torchvision` depending on the requested format.
  - Returns the resolved path even when retries exhaust, mirroring the legacy helper so callers can decide how to respond.
- **Metadata bundling**
  - `write_metadata_bundle` infers the artifact type from `MetadataSaveConfig`/file suffix, calls the existing writers, and reports failures through the notifications logger.
- **Orchestration touchpoints**
  - `wgp.generate_video` uses `write_metadata_bundle` for JSON embedding (audio/image/video variants) while retaining JSON sidecar support.
  - `preprocessing.matanyone.app._save_outputs` writes foreground/alpha MP4 metadata through the bundler and tags each artifact role.
  - CLI queue tooling records resulting paths in `state["gen"]["file_list"]`; downstream consumers can toggle metadata via `server_config["metadata_type"]`.

## Metadata Migration Notes
- `MetadataSaveConfig` should capture the target artifact type (`"video"`, `"image"`, `"audio"`), container override, and any format-specific kwargs (e.g. encoder versions, embedded image payloads).
- `write_metadata_bundle` will dispatch to:
  - `shared.utils.video_metadata.save_video_metadata` for MP4/MKV outputs.
  - `shared.utils.audio_metadata.save_audio_metadata` for audio sidecars (when present).
- The helper should accept a logger (defaulting to the notifications logger) and translate the current `print` statements into `logger.warning`/`logger.error` calls.
- Next step: surface CLI/runner-level controls for toggling metadata manifests (e.g. JSON sidecars, structured summaries) now that `ProductionManager` ships the per-type `MetadataSaveConfig` templates.

## Validation
- Extend smoke tests: run `python -m cli.generate --prompt "smoke test prompt" --dry-run` plus a low-res render to confirm output paths.
- MatAnyOne regression: `python -m cli.matanyone --input sample.mp4 --template-mask mask.png --dry-run` and a short real run to confirm codec overrides.
- Capture timings and VRAM usage in `docs/WORK_HISTORY.md` once the new helpers land.
