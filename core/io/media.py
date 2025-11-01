from __future__ import annotations

import logging
import os
import secrets
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import imageio
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image as torchvision_save_image

try:  # pragma: no cover - optional dependency; exercised in integration
    import soundfile as sf
except ImportError:  # pragma: no cover - dependency handling
    sf = None  # type: ignore[assignment]

LoggerCallable = Union[Callable[[str], None], logging.Logger]
MetadataHandler = Callable[[str, Dict[str, Any], Dict[str, Any]], bool]


def _emit(logger: Optional[LoggerCallable], message: str, level: str = "debug") -> None:
    """
    Emit a diagnostic message using the provided logger or callable.

    Accepts either a ``logging.Logger`` instance or a callable that expects a single
    string argument. All exceptions are swallowed so logging never interrupts media
    persistence.
    """

    if logger is None:
        return

    if isinstance(logger, logging.Logger):
        try:
            log_fn = getattr(logger, level, logger.debug)
            log_fn(message)
            return
        except Exception:  # pragma: no cover - logging failures are non-fatal
            pass

    if callable(logger):
        try:
            logger(message)  # type: ignore[misc]
        except Exception:  # pragma: no cover - logging failures are non-fatal
            pass


@dataclass
class VideoSaveConfig:
    """Configuration payload for video persistence."""

    fps: int = 30
    codec_type: str = "libx264_8"
    container: str = "mp4"
    nrow: int = 8
    normalize: bool = True
    value_range: Tuple[float, float] = (-1.0, 1.0)
    retry: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageSaveConfig:
    """Configuration payload for image persistence."""

    nrow: int = 8
    normalize: bool = True
    value_range: Tuple[float, float] = (-1.0, 1.0)
    quality: str = "jpeg_95"
    retry: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSaveConfig:
    """Configuration payload for audio persistence."""

    sample_rate: Optional[int] = None
    subtype: Optional[str] = None
    format: Optional[str] = None
    retry: int = 5
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskSaveConfig:
    """Configuration payload for mask archive persistence."""

    retry: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataSaveConfig:
    """Configuration payload for metadata persistence."""

    format_hint: Optional[str] = None
    handlers: Dict[str, MetadataHandler] = field(default_factory=dict)
    extra_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MediaPersistenceContext:
    """
    Bundle media persistence defaults for a generation run.

    Callers clone the stored config templates through ``video_config`` /
    ``image_config`` so each save operation receives an isolated dataclass
    instance. Audio and mask helpers follow the same pattern so queue/CLI
    orchestration can persist every artifact through a single context.
    ``save_debug_masks`` mirrors the legacy ``save_masks`` flag on
    ``server_config`` and allows the runner to decide whether mask previews
    should be persisted for inspection.
    """

    video_template: VideoSaveConfig
    image_template: ImageSaveConfig
    audio_template: AudioSaveConfig = field(default_factory=AudioSaveConfig)
    mask_template: MaskSaveConfig = field(default_factory=MaskSaveConfig)
    save_debug_masks: bool = False

    def video_config(self, **overrides: Any) -> VideoSaveConfig:
        """
        Return a copy of the video template with optional field overrides.
        """

        config = replace(self.video_template)
        config.extra_params = dict(self.video_template.extra_params)
        if overrides:
            fields = config.__dataclass_fields__
            for key, value in overrides.items():
                if key not in fields:
                    raise AttributeError(f"VideoSaveConfig has no field named '{key}'")
                setattr(config, key, value)
        return config

    def image_config(self, **overrides: Any) -> ImageSaveConfig:
        """
        Return a copy of the image template with optional field overrides.
        """

        config = replace(self.image_template)
        config.extra_params = dict(self.image_template.extra_params)
        if overrides:
            fields = config.__dataclass_fields__
            for key, value in overrides.items():
                if key not in fields:
                    raise AttributeError(f"ImageSaveConfig has no field named '{key}'")
                setattr(config, key, value)
        return config

    def audio_config(self, **overrides: Any) -> AudioSaveConfig:
        """
        Return a copy of the audio template with optional field overrides.
        """

        config = replace(self.audio_template)
        config.extra_params = dict(self.audio_template.extra_params)
        if overrides:
            fields = config.__dataclass_fields__
            for key, value in overrides.items():
                if key not in fields:
                    raise AttributeError(f"AudioSaveConfig has no field named '{key}'")
                setattr(config, key, value)
        return config

    def mask_config(self, **overrides: Any) -> MaskSaveConfig:
        """
        Return a copy of the mask template with optional field overrides.
        """

        config = replace(self.mask_template)
        config.extra_params = dict(self.mask_template.extra_params)
        if overrides:
            fields = config.__dataclass_fields__
            for key, value in overrides.items():
                if key not in fields:
                    raise AttributeError(f"MaskSaveConfig has no field named '{key}'")
                setattr(config, key, value)
        return config

    def should_save_masks(self) -> bool:
        """
        Indicate whether debug mask artifacts should be persisted.
        """

        return self.save_debug_masks

    def save_video(
        self,
        data: Any,
        target_path: Optional[str],
        *,
        logger: Optional[LoggerCallable] = None,
        config: Optional[VideoSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        """
        Persist a video artifact using either the provided config or template overrides.
        """

        effective_config: VideoSaveConfig
        if config is None:
            effective_config = self.video_config(**dict(overrides or {}))
        else:
            effective_config = replace(config)
            effective_config.extra_params = dict(config.extra_params)
            if overrides:
                fields = effective_config.__dataclass_fields__
                for key, value in overrides.items():
                    if key not in fields:
                        raise AttributeError(f"VideoSaveConfig has no field named '{key}'")
                    setattr(effective_config, key, value)
        return write_video(data, target_path, config=effective_config, logger=logger)

    def save_image(
        self,
        data: Any,
        target_path: str,
        *,
        logger: Optional[LoggerCallable] = None,
        config: Optional[ImageSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """
        Persist an image artifact using either the provided config or template overrides.
        """

        effective_config: ImageSaveConfig
        if config is None:
            effective_config = self.image_config(**dict(overrides or {}))
        else:
            effective_config = replace(config)
            effective_config.extra_params = dict(config.extra_params)
            if overrides:
                fields = effective_config.__dataclass_fields__
                for key, value in overrides.items():
                    if key not in fields:
                        raise AttributeError(f"ImageSaveConfig has no field named '{key}'")
                    setattr(effective_config, key, value)
        return write_image(data, target_path, config=effective_config, logger=logger)

    @staticmethod
    def _coerce_audio_array(data: Any) -> np.ndarray:
        """
        Convert supported audio inputs into a numpy array suitable for soundfile.
        """

        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

    def save_audio(
        self,
        data: Any,
        target_path: str,
        *,
        sample_rate: Optional[int] = None,
        logger: Optional[LoggerCallable] = None,
        config: Optional[AudioSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """
        Persist an audio artifact using either the provided config or template overrides.
        """

        if sf is None:  # pragma: no cover - dependent on optional package
            raise RuntimeError("soundfile dependency is required to persist audio artifacts.")

        effective_config: AudioSaveConfig
        if config is None:
            effective_config = self.audio_config(**dict(overrides or {}))
        else:
            effective_config = replace(config)
            effective_config.extra_params = dict(config.extra_params)
            if overrides:
                fields = effective_config.__dataclass_fields__
                for key, value in overrides.items():
                    if key not in fields:
                        raise AttributeError(f"AudioSaveConfig has no field named '{key}'")
                    setattr(effective_config, key, value)

        resolved_sample_rate = sample_rate if sample_rate is not None else effective_config.sample_rate
        if resolved_sample_rate is None:
            raise ValueError("save_audio requires a sample rate via argument or AudioSaveConfig.sample_rate")

        array = self._coerce_audio_array(data)
        error: Optional[Exception] = None
        for attempt in range(1, effective_config.retry + 1):
            try:
                sf.write(
                    target_path,
                    array,
                    resolved_sample_rate,
                    subtype=effective_config.subtype,
                    format=effective_config.format,
                    **effective_config.extra_params,
                )
                return target_path
            except Exception as exc:  # pragma: no cover - dependent on codec availability
                error = exc
                _emit(
                    logger,
                    f"save_audio attempt {attempt}/{effective_config.retry} failed for {target_path}: {exc}",
                    level="warning",
                )

        _emit(logger, f"save_audio exhausted retries for {target_path}: {error}", level="error")
        return target_path

    def save_mask_archive(
        self,
        frames: Any,
        target_path: str,
        *,
        logger: Optional[LoggerCallable] = None,
        config: Optional[MaskSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        """
        Persist a mask archive (typically RGBA frames) when mask saving is enabled.
        """

        if frames is None:
            return None
        if not force and not self.should_save_masks():
            _emit(logger, f"save_mask_archive skipped for {target_path}; mask persistence disabled.", level="info")
            return None

        effective_config: MaskSaveConfig
        if config is None:
            effective_config = self.mask_config(**dict(overrides or {}))
        else:
            effective_config = replace(config)
            effective_config.extra_params = dict(config.extra_params)
            if overrides:
                fields = effective_config.__dataclass_fields__
                for key, value in overrides.items():
                    if key not in fields:
                        raise AttributeError(f"MaskSaveConfig has no field named '{key}'")
                    setattr(effective_config, key, value)

        error: Optional[Exception] = None
        for attempt in range(1, effective_config.retry + 1):
            try:
                from models.wan.alpha.utils import write_zip_file

                write_zip_file(target_path, frames, **effective_config.extra_params)
                return target_path
            except Exception as exc:  # pragma: no cover - dependent on optional deps
                error = exc
                _emit(
                    logger,
                    f"save_mask_archive attempt {attempt}/{effective_config.retry} failed for {target_path}: {exc}",
                    level="warning",
                )

        _emit(logger, f"save_mask_archive exhausted retries for {target_path}: {error}", level="error")
        return None


def build_media_context(server_config: Mapping[str, Any]) -> MediaPersistenceContext:
    """
    Construct a ``MediaPersistenceContext`` from the provided server configuration.
    """

    video_template = VideoSaveConfig(
        codec_type=server_config.get("video_output_codec"),
        container=server_config.get("video_container", "mp4"),
    )
    image_template = ImageSaveConfig(
        quality=server_config.get("image_output_codec"),
    )
    raw_audio_rate = server_config.get("audio_sample_rate")
    audio_sample_rate: Optional[int]
    try:
        audio_sample_rate = int(raw_audio_rate) if raw_audio_rate is not None else None
    except (TypeError, ValueError):
        audio_sample_rate = None
    audio_template = AudioSaveConfig(
        sample_rate=audio_sample_rate,
        subtype=server_config.get("audio_output_subtype"),
        format=server_config.get("audio_output_format"),
    )
    raw_mask_retry = server_config.get("mask_archive_retry")
    mask_retry = MaskSaveConfig().retry
    if raw_mask_retry is not None:
        try:
            mask_retry = max(1, int(raw_mask_retry))
        except (TypeError, ValueError):
            mask_retry = MaskSaveConfig().retry
    mask_template = MaskSaveConfig(retry=mask_retry)
    save_debug_masks = bool(server_config.get("save_masks", False))
    return MediaPersistenceContext(
        video_template=video_template,
        image_template=image_template,
        audio_template=audio_template,
        mask_template=mask_template,
        save_debug_masks=save_debug_masks,
    )


def clone_metadata_config(
    template: MetadataSaveConfig,
    *,
    fallback_hint: str,
) -> MetadataSaveConfig:
    """
    Return a deep copy of a metadata configuration template.

    The clone keeps handler bindings and per-format options isolated so callers can
    mutate the result without affecting shared templates.
    """

    cloned = replace(template)
    cloned.handlers = dict(template.handlers)
    cloned.extra_options = {key: dict(value) for key, value in template.extra_options.items()}
    if not cloned.format_hint:
        cloned.format_hint = fallback_hint
    return cloned


def default_metadata_config_templates() -> Dict[str, MetadataSaveConfig]:
    """
    Provide the default metadata configuration templates for video, image, and audio.
    """

    return {
        "video": MetadataSaveConfig(format_hint="video"),
        "image": MetadataSaveConfig(format_hint="image"),
        "audio": MetadataSaveConfig(format_hint="audio"),
    }


def build_metadata_config(
    format_hint: str,
    *,
    templates: Optional[Dict[str, MetadataSaveConfig]] = None,
) -> MetadataSaveConfig:
    """
    Construct a metadata configuration for the requested format.

    When templates are supplied the function clones the matching entry; otherwise it
    falls back to the module defaults.
    """

    candidates = templates if templates is not None else default_metadata_config_templates()
    template = candidates.get(format_hint)
    if template is not None:
        return clone_metadata_config(template, fallback_hint=format_hint)
    return MetadataSaveConfig(format_hint=format_hint)


def _ensure_container_suffix(path: str, container: Optional[str]) -> str:
    if not container:
        return path
    suffix = f".{container}" if not container.startswith(".") else container
    if path.endswith(suffix):
        return path
    root, _ = os.path.splitext(path)
    return f"{root}{suffix}"


def _resolve_target_path(target_path: Optional[str], container: Optional[str]) -> str:
    suffix = ""
    if container:
        suffix = container if container.startswith(".") else f".{container}"

    if target_path is None:
        temp_dir = tempfile.gettempdir()
        name = secrets.token_hex(8)
        return os.path.join(temp_dir, f"{name}{suffix}")

    return _ensure_container_suffix(target_path, container)


def _get_codec_params(codec_type: Optional[str], container: Optional[str]) -> Dict[str, Any]:
    if codec_type == "libx264_8":
        return {"codec": "libx264", "quality": 8, "pixelformat": "yuv420p"}
    if codec_type == "libx264_10":
        return {"codec": "libx264", "quality": 10, "pixelformat": "yuv420p"}
    if codec_type == "libx265_28":
        return {
            "codec": "libx265",
            "pixelformat": "yuv420p",
            "output_params": ["-crf", "28", "-x265-params", "log-level=none", "-hide_banner", "-nostats"],
        }
    if codec_type == "libx265_8":
        return {
            "codec": "libx265",
            "pixelformat": "yuv420p",
            "output_params": ["-crf", "8", "-x265-params", "log-level=none", "-hide_banner", "-nostats"],
        }
    if codec_type == "libx264_lossless":
        if container == "mkv":
            return {"codec": "ffv1", "pixelformat": "rgb24"}
        return {"codec": "libx264", "output_params": ["-crf", "0"], "pixelformat": "yuv444p"}
    return {"codec": "libx264", "pixelformat": "yuv420p"}


def _prepare_video_frames(data: Any, config: VideoSaveConfig) -> Any:
    if not torch.is_tensor(data):
        return data

    tensor = data
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.clamp(min=config.value_range[0], max=config.value_range[1])
    grids = [
        make_grid(frame, nrow=config.nrow, normalize=config.normalize, value_range=config.value_range)
        for frame in tensor.unbind(2)
    ]
    stacked = torch.stack(grids, dim=1).permute(1, 2, 3, 0)
    stacked = (stacked * 255).type(torch.uint8).cpu()
    return stacked.numpy()


def _get_format_info(quality: str) -> Dict[str, Any]:
    formats: Dict[str, Dict[str, Any]] = {
        "jpeg_95": {"ext": ".jpg", "params": {"quality": 95}, "use_pil": True},
        "jpeg_85": {"ext": ".jpg", "params": {"quality": 85}, "use_pil": True},
        "jpeg_70": {"ext": ".jpg", "params": {"quality": 70}, "use_pil": True},
        "jpeg_50": {"ext": ".jpg", "params": {"quality": 50}, "use_pil": True},
        "png": {"ext": ".png", "params": {}, "use_pil": False},
        "webp_95": {"ext": ".webp", "params": {"quality": 95}, "use_pil": True},
        "webp_85": {"ext": ".webp", "params": {"quality": 85}, "use_pil": True},
        "webp_70": {"ext": ".webp", "params": {"quality": 70}, "use_pil": True},
        "webp_50": {"ext": ".webp", "params": {"quality": 50}, "use_pil": True},
        "webp_lossless": {"ext": ".webp", "params": {"lossless": True}, "use_pil": True},
    }
    return formats.get(quality, formats["jpeg_95"])


def write_video(
    data: Any,
    target_path: Optional[str],
    *,
    config: VideoSaveConfig,
    logger: Optional[LoggerCallable] = None,
) -> Optional[str]:
    """
    Persist a video tensor/array using the provided configuration.

    Returns the resolved output path when successful or ``None`` if all retries fail.
    """

    output_path = _resolve_target_path(target_path, config.container)
    codec_params = _get_codec_params(config.codec_type, config.container)
    if config.extra_params:
        codec_params.update(config.extra_params)

    error: Optional[Exception] = None
    for attempt in range(1, config.retry + 1):
        try:
            frames = _prepare_video_frames(data, config)
            with imageio.get_writer(
                output_path,
                fps=config.fps,
                ffmpeg_log_level="error",
                **codec_params,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            return output_path
        except Exception as exc:  # pragma: no cover - dependent on codec availability
            error = exc
            _emit(
                logger,
                f"write_video attempt {attempt}/{config.retry} failed for {output_path}: {exc}",
                level="warning",
            )

    _emit(logger, f"write_video exhausted retries for {output_path}: {error}", level="error")
    return None


def write_image(
    data: Any,
    target_path: str,
    *,
    config: ImageSaveConfig,
    logger: Optional[LoggerCallable] = None,
) -> str:
    """
    Persist an image tensor using the provided configuration.

    Returns the resolved output path (mirrors the legacy helper behaviour even when all
    retries fail, so callers can decide how to handle missing files).
    """

    if not torch.is_tensor(data):
        raise TypeError("write_image expects a torch.Tensor input.")

    tensor = data
    rgba = tensor.shape[0] == 4
    effective_quality = "png" if rgba else config.quality
    format_info = _get_format_info(effective_quality)
    output_path = os.path.splitext(target_path)[0] + format_info["ext"]
    save_kwargs = dict(format_info["params"])
    save_kwargs.update(config.extra_params)

    error: Optional[Exception] = None
    for attempt in range(1, config.retry + 1):
        try:
            tensor = tensor.clamp(min=config.value_range[0], max=config.value_range[1])
            if format_info["use_pil"] or rgba:
                grid = make_grid(
                    tensor,
                    nrow=config.nrow,
                    normalize=config.normalize,
                    value_range=config.value_range,
                )
                grid = (
                    grid.mul(255)
                    .add_(0.5)
                    .clamp_(0, 255)
                    .permute(1, 2, 0)
                    .to("cpu", torch.uint8)
                    .numpy()
                )
                mode = "RGBA" if rgba else "RGB"
                Image.fromarray(grid, mode=mode).save(output_path, **save_kwargs)
            else:
                torchvision_save_image(
                    tensor,
                    output_path,
                    nrow=config.nrow,
                    normalize=config.normalize,
                    value_range=config.value_range,
                    **save_kwargs,
                )
            return output_path
        except Exception as exc:  # pragma: no cover - dependent on codec availability
            error = exc
            _emit(
                logger,
                f"write_image attempt {attempt}/{config.retry} failed for {output_path}: {exc}",
                level="warning",
            )

    _emit(logger, f"write_image exhausted retries for {output_path}: {error}", level="error")
    return output_path


def write_metadata_bundle(
    target_path: str,
    payload: Dict[str, Any],
    *,
    config: MetadataSaveConfig,
    logger: Optional[LoggerCallable] = None,
) -> bool:
    """
    Persist metadata alongside a media artifact.

    Dispatches to the appropriate metadata handler based on ``format_hint`` or
    the file extension. Callers may override the handler mapping and per-format
    options through ``MetadataSaveConfig``.
    """

    metadata_type = _infer_metadata_type(target_path, config.format_hint)
    if metadata_type is None:
        _emit(logger, f"write_metadata_bundle could not infer metadata type for {target_path}", level="warning")
        return False

    handlers = _build_handlers(config.handlers)
    handler = handlers.get(metadata_type)
    if handler is None:
        _emit(logger, f"write_metadata_bundle has no handler for type '{metadata_type}' ({target_path})", level="warning")
        return False

    options = config.extra_options.get(metadata_type, {})
    try:
        result = handler(target_path, payload, options)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        _emit(logger, f"write_metadata_bundle failed for {target_path}: {exc}", level="error")
        return False

    if not result:
        _emit(logger, f"write_metadata_bundle handler reported failure for {target_path}", level="warning")
        return False

    return True


def _infer_metadata_type(path: str, hint: Optional[str]) -> Optional[str]:
    if hint:
        return hint.strip().lower()

    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".wav"}:
        return "audio"
    if suffix in {".mp4", ".mkv"}:
        return "video"
    return None


def _build_handlers(overrides: Dict[str, MetadataHandler]) -> Dict[str, MetadataHandler]:
    handlers = {key.strip().lower(): value for key, value in overrides.items() if callable(value)}
    for key, handler in _default_metadata_handlers().items():
        handlers.setdefault(key, handler)
    return handlers


def _default_metadata_handlers() -> Dict[str, MetadataHandler]:
    def image_handler(target_path: str, payload: Dict[str, Any], options: Dict[str, Any]) -> bool:
        from shared.utils.audio_video import _legacy_save_image_metadata

        save_kwargs = options.get("save_kwargs", {})
        return bool(_legacy_save_image_metadata(target_path, payload, **save_kwargs))

    def audio_handler(target_path: str, payload: Dict[str, Any], options: Dict[str, Any]) -> bool:
        from shared.utils.audio_metadata import save_audio_metadata

        save_audio_metadata(target_path, payload)
        return True

    def video_handler(target_path: str, payload: Dict[str, Any], options: Dict[str, Any]) -> bool:
        from shared.utils.video_metadata import save_video_metadata

        source_images = options.get("source_images")
        return bool(save_video_metadata(target_path, payload, source_images))

    return {
        "image": image_handler,
        "audio": audio_handler,
        "video": video_handler,
    }


__all__ = [
    "AudioSaveConfig",
    "ImageSaveConfig",
    "MaskSaveConfig",
    "MediaPersistenceContext",
    "MetadataSaveConfig",
    "VideoSaveConfig",
    "build_media_context",
    "build_metadata_config",
    "clone_metadata_config",
    "default_metadata_config_templates",
    "write_image",
    "write_metadata_bundle",
    "write_video",
]
