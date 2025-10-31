from __future__ import annotations

import logging
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import imageio
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image as torchvision_save_image

LoggerCallable = Union[Callable[[str], None], logging.Logger]


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
class MetadataSaveConfig:
    """Configuration payload for metadata persistence."""

    format_hint: Optional[str] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)


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

    The final implementation will route to image/audio/video specific handlers.
    Until then this placeholder always raises ``NotImplementedError`` so legacy
    callers continue to rely on the existing helpers.
    """

    _emit(logger, "write_metadata_bundle invoked (scaffolding); falling back to legacy implementation.")
    raise NotImplementedError("write_metadata_bundle is not implemented yet.")


__all__ = [
    "ImageSaveConfig",
    "MetadataSaveConfig",
    "VideoSaveConfig",
    "write_image",
    "write_metadata_bundle",
    "write_video",
]
