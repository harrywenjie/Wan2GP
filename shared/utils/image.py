from __future__ import annotations

import base64
import io
from typing import Any, Optional

from PIL import Image  # type: ignore

from shared.utils.utils import get_video_frame

__all__ = ["pil_to_base64_uri"]


def pil_to_base64_uri(image: Any, *, format: str = "png", quality: int = 75) -> Optional[str]:
    """
    Serialise a PIL image (or compatible source) into a data URI.

    Accepts Pillow ``Image`` instances, file paths, or objects already storing
    base64 data. When a file path is provided the first frame is extracted using
    ``shared.utils.utils.get_video_frame`` to mirror the legacy behaviour in
    ``wgp.py``. Returns ``None`` on failure to avoid surfacing conversion errors
    during logging or queue preview generation.
    """

    if image is None:
        return None

    if isinstance(image, str):
        image = get_video_frame(image, 0)

    if not isinstance(image, Image.Image):
        return None

    buffer = io.BytesIO()
    try:
        to_save = image
        fmt = format.lower()
        if fmt == "jpeg" and image.mode == "RGBA":
            to_save = image.convert("RGB")
        elif fmt == "png" and image.mode not in {"RGB", "RGBA", "L", "P"}:
            to_save = image.convert("RGBA")
        elif image.mode == "P":
            to_save = image.convert("RGBA" if "transparency" in image.info else "RGB")

        save_kwargs = {"format": format}
        if fmt == "jpeg":
            save_kwargs["quality"] = quality

        to_save.save(buffer, **save_kwargs)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{fmt};base64,{encoded}"
    except Exception:
        return None
