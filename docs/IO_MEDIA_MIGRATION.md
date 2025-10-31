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
1. **Introduce module and dataclasses** – add `core/io/media.py` with the new helpers, unit-friendly interfaces, and logging hooks.
2. **Implement compatibility shims** – update `shared.utils.audio_video.save_video/save_image` and metadata writers to call into `core/io/media.py`, emitting deprecation warnings.
3. **Update orchestration layers** – adjust `wgp.generate_video`, `ProductionManager`, and MatAnyOne to instantiate configs and call the new helpers directly; remove reliance on the legacy shims once all call sites migrate.
4. **Delete shims** – after validation, drop the old functions from `shared.utils.audio_video` and clean up imports.
5. **Document the new surfaces** – refresh `docs/CLI.md`, `docs/APPENDIX_HEADLESS.md`, and this project plan with the final module structure.

## Dependency Map
- **`shared.utils.audio_video.save_video`**
  - Imports: `torch`, `torchvision`, `imageio`, `numpy`, and `shared.utils.misc.rand_name` for temp files.
  - Inputs: tensor (torch or numpy), FPS, codec string (matches server_config `video_output_codec`), container (`video_container`), retry count, and optional `save_file`.
  - Behaviour: writes to `/tmp` when `save_file` is `None`, mutates tensor normalization in-place, prints errors on failure.
- **`shared.utils.audio_video.save_image`**
  - Imports: `torch`, `torchvision`, `PIL.Image`.
  - Inputs: torch tensor, target path, grid params, codec string (server_config `image_output_codec`), retry count.
  - Behaviour: switches extension based on codec string, uses PIL for RGBA/WebP variants, prints errors when retries exhaust.
- **Metadata writers**
  - `save_video_metadata` (MP4/MKV), `save_image_metadata` (PNG/JPEG/WebP via `PIL` + `piexif`), `save_audio_metadata` (WAV via `mutagen`), all emit `print` on failure.
- **Orchestration touchpoints**
  - `wgp.generate_video` passes codec/container/quality from `server_config` and expects string paths back; error logging currently bubbles to stdout.
  - `preprocessing.matanyone.app` passes per-request codec overrides, works exclusively with numpy frame lists, and expects the writer to honour supplied extensions.
  - CLI queue tooling records resulting paths in `state["gen"]["file_list"]`; any helper changes must preserve those contracts.

## Validation
- Extend smoke tests: run `python -m cli.generate --prompt "smoke test prompt" --dry-run` plus a low-res render to confirm output paths.
- MatAnyOne regression: `python -m cli.matanyone --input sample.mp4 --template-mask mask.png --dry-run` and a short real run to confirm codec overrides.
- Capture timings and VRAM usage in `docs/WORK_HISTORY.md` once the new helpers land.
