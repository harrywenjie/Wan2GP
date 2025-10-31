from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

__all__ = ["PREVIEW_KEYS", "prepare_preview_inputs"]

PREVIEW_KEYS: Sequence[str] = (
    "image_start",
    "video_source",
    "image_end",
    "video_guide",
    "image_guide",
    "video_mask",
    "image_mask",
    "image_refs",
)


def prepare_preview_inputs(
    base_inputs: Mapping[str, Any],
    refresh_updates: Mapping[str, Any],
    *,
    extra_sources: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Merge the original generation inputs with freshly generated preview assets.

    The legacy implementation passed a raw ``locals()`` snapshot into the
    notifier. This helper narrows the payload to the keys required by
    ``get_preview_images`` / ``update_task_thumbnails`` so headless callers can
    avoid leaking unrelated state.
    """

    preview: Dict[str, Any] = {}

    if extra_sources is None:
        extra_sources = {}

    for key in PREVIEW_KEYS:
        if key in refresh_updates:
            value = refresh_updates[key]
        elif key in extra_sources:
            value = extra_sources[key]
        else:
            value = base_inputs.get(key)
        if value is not None:
            preview[key] = value

    for key, value in refresh_updates.items():
        if value is not None:
            preview[key] = value

    return preview
