from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .media import (
    build_metadata_config,
    clone_metadata_config,
    default_metadata_config_templates,
    ImageSaveConfig,
    MetadataSaveConfig,
    VideoSaveConfig,
    write_image,
    write_metadata_bundle,
    write_video,
)

PathLikeStr = Union[str, Path]


def get_available_filename(
    target_directory: PathLikeStr,
    source_path: PathLikeStr,
    suffix: str = "",
    force_extension: Optional[str] = None,
) -> str:
    """
    Return a file path inside ``target_directory`` that does not collide with existing files.

    The helper derives a stem and extension from ``source_path``, applies the optional
    ``suffix`` and ``force_extension`` overrides, and appends ``(n)`` if needed to avoid
    overwriting existing files.
    """

    directory = Path(target_directory)
    name = Path(source_path).name
    stem = Path(name).stem
    extension = Path(name).suffix

    if force_extension is not None:
        extension = force_extension if force_extension.startswith(".") else f".{force_extension}"

    candidate = directory / f"{stem}{suffix}{extension}"
    if not candidate.exists():
        return str(candidate)

    counter = 2
    while True:
        candidate = directory / f"{stem}{suffix}({counter}){extension}"
        if not candidate.exists():
            return str(candidate)
        counter += 1


__all__ = [
    "ImageSaveConfig",
    "MetadataSaveConfig",
    "VideoSaveConfig",
    "build_metadata_config",
    "clone_metadata_config",
    "default_metadata_config_templates",
    "get_available_filename",
    "write_image",
    "write_metadata_bundle",
    "write_video",
]
