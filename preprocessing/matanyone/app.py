"""
Headless MatAnyOne preprocessing helpers.

This module exposes a CLI-friendly wrapper around the MatAnyOne mask propagation
workflow. It removes all Gradio dependencies in favour of pure data structures
so callers can orchestrate preprocessing directly from Python or the command
line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from core.io.media import build_metadata_config, write_metadata_bundle
from shared.utils import files_locator as fl
from shared.utils.audio_video import (
    cleanup_temp_audio_files,
    combine_video_with_audio_tracks,
    extract_audio_tracks,
    save_video,
)
from shared.utils.notifications import get_notifications_logger
from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources
from shared.utils.utils import (
    calculate_new_dimensions,
    has_image_file_extension,
    has_video_file_extension,
    sanitize_file_name,
    truncate_for_filesystem,
)

from .matanyone.inference.inference_core import InferenceCore
from .matanyone_wrapper import matanyone
from .utils.get_default_model import get_matanyone_model


_LOGGER = logging.getLogger("wan2gp.matanyone")
_DEFAULT_DEVICE = "cuda"
_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


class GPUSession:
    """Context manager to gate access to the shared GPU lock."""

    def __init__(self, state: Dict, notifier: Optional[Callable[[str], None]] = None):
        self._state = state
        self._notifier = notifier

    def __enter__(self) -> None:
        acquire_GPU_ressources(
            self._state,
            process_id="matanyone",
            process_name="MatAnyOne",
            notifier=self._notifier,
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        release_GPU_ressources(self._state, "matanyone")


@dataclass
class MatAnyOneRequest:
    """Parameters required to propagate a mask through a video."""

    input_path: Path
    template_mask_path: Path
    output_dir: Path = Path("mask_outputs")
    start_frame: int = 0
    end_frame: Optional[int] = None
    new_dim: str = ""
    matting_type: str = "Foreground"
    mask_type: str = "wangp"
    erode_kernel_size: int = 0
    dilate_kernel_size: int = 0
    warmup_frames: int = 10
    device: str = _DEFAULT_DEVICE
    attach_audio: bool = True
    codec: str = "libx264_8"
    notifier: Optional[Callable[[str], None]] = None

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.template_mask_path = Path(self.template_mask_path)
        self.output_dir = Path(self.output_dir)
        self.start_frame = int(self.start_frame)
        if self.end_frame is not None:
            self.end_frame = int(self.end_frame)

    @property
    def normalized_mask_type(self) -> str:
        return (self.mask_type or "wangp").strip().lower()

    @property
    def normalized_matting_type(self) -> str:
        return (self.matting_type or "Foreground").strip().lower()


@dataclass
class MatAnyOneResult:
    """Artifacts emitted by a mask propagation run."""

    foreground_path: Path
    alpha_path: Path
    rgba_zip_path: Optional[Path]
    frames_processed: int
    fps: float
    metadata: Dict[str, object] = field(default_factory=dict)


def generate_masks(state: Optional[Dict], request: MatAnyOneRequest) -> MatAnyOneResult:
    """
    Run the MatAnyOne propagation pipeline for the provided request.

    Args:
        state: Shared CLI state dictionary (mutated with GPU lock bookkeeping).
        request: Fully populated MatAnyOneRequest.

    Returns:
        MatAnyOneResult describing written artifacts and run metadata.
    """
    state = state or {}

    if request.normalized_mask_type not in {"wangp", "", "greenscreen", "alpha"}:
        raise ValueError(f"Unsupported mask type '{request.mask_type}'.")

    if request.normalized_matting_type not in {"foreground", "background"}:
        raise ValueError(f"Unsupported matting type '{request.matting_type}'.")

    frames, fps, audio_tracks, audio_metadata = _load_frames(request)
    template_mask = _load_template_mask(request.template_mask_path, frames[0].shape)

    device = _resolve_device(request.device)

    with GPUSession(state, request.notifier):
        model = _load_model(device)
        processor = InferenceCore(model, cfg=model.cfg)
        _, alpha_frames = matanyone(
            processor,
            frames,
            template_mask,
            r_erode=request.erode_kernel_size,
            r_dilate=request.dilate_kernel_size,
            n_warmup=request.warmup_frames,
        )

    _offload_model(model)

    (
        output_frames,
        adjusted_alpha_frames,
        foreground_suffix,
        alpha_suffix,
        rgba_frames,
    ) = _compose_outputs(frames, alpha_frames, request)

    result = _save_outputs(
        request=request,
        frames=output_frames,
        alpha_frames=adjusted_alpha_frames,
        fps=fps,
        audio_tracks=audio_tracks if request.attach_audio else [],
        audio_metadata=audio_metadata if request.attach_audio else None,
        foreground_suffix=foreground_suffix,
        alpha_suffix=alpha_suffix,
        rgba_frames=rgba_frames,
    )

    return result


def _resolve_device(device: Optional[str]) -> str:
    resolved = (device or _DEFAULT_DEVICE).strip().lower()
    if resolved.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("MatAnyOne requires CUDA, but no compatible GPU is available.")
        return resolved
    if resolved != "cpu":
        _LOGGER.warning("Unknown device '%s'; defaulting to CPU. Expect significantly slower inference.", device)
        resolved = "cpu"
    return resolved


def _load_model(device: str) -> torch.nn.Module:
    model = _MODEL_CACHE.get(device)
    if model is None:
        weights_path = fl.locate_folder("mask")
        model = get_matanyone_model(weights_path, device=device if device != "cpu" else None)
        _MODEL_CACHE[device] = model
    else:
        model.to(device)
    return model


def _offload_model(model: torch.nn.Module) -> None:
    if model is None:
        return
    model.to("cpu")
    torch.cuda.empty_cache()


def _load_frames(
    request: MatAnyOneRequest,
) -> Tuple[List[np.ndarray], float, List[str], Optional[Sequence[Dict[str, object]]]]:
    path = request.input_path
    if not path.exists():
        raise FileNotFoundError(f"Input path '{path}' does not exist.")

    if has_video_file_extension(path.name):
        frames, fps = _load_video_frames(path, request.end_frame)
        audio_tracks, audio_metadata = extract_audio_tracks(str(path))
    elif has_image_file_extension(path.name):
        frames = [_load_image_frame(path)]
        fps = 1.0
        audio_tracks, audio_metadata = [], None
    else:
        raise ValueError(f"Unsupported input type for '{path}'. Provide a video or image file.")

    if request.new_dim:
        frames = _perform_spatial_upsampling(frames, request.new_dim)

    total_frames = len(frames)
    start = max(request.start_frame, 0)
    end = total_frames if request.end_frame is None else min(request.end_frame, total_frames)
    if start >= end:
        raise ValueError(f"Invalid frame range [{start}, {end}) for source with {total_frames} frames.")

    sliced = frames[start:end]
    if not sliced:
        raise ValueError("No frames selected after applying the requested frame range.")

    return sliced, fps, audio_tracks, audio_metadata


def _load_video_frames(path: Path, end_frame: Optional[int]) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video '{path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []
    frame_limit = end_frame if end_frame is None else max(end_frame, 0)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frame_limit and len(frames) >= frame_limit:
                break
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from '{path}'.")

    return frames, float(fps)


def _load_image_frame(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def _perform_spatial_upsampling(frames: List[np.ndarray], new_dim: str) -> List[np.ndarray]:
    spec = new_dim.strip()
    if not spec:
        return frames

    fit_into_canvas = "outer" in spec.lower()
    base_spec = spec.split()[0]

    if base_spec.lower() == "1080p":
        canvas_w, canvas_h = 1920, 1088
    elif base_spec.lower() == "720p":
        canvas_w, canvas_h = 1280, 720
    else:
        canvas_w, canvas_h = 832, 480

    h, w = frames[0].shape[:2]
    new_h, new_w = calculate_new_dimensions(
        canvas_h,
        canvas_w,
        h,
        w,
        fit_into_canvas=fit_into_canvas,
        block_size=16,
    )

    resized: List[np.ndarray] = []
    for frame in frames:
        resized_frame = np.array(
            Image.fromarray(frame).resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        )
        resized.append(resized_frame)
    return resized


def _load_template_mask(mask_path: Path, reference_shape: Tuple[int, int, int]) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Template mask '{mask_path}' not found.")

    mask = Image.open(mask_path).convert("L")
    ref_h, ref_w = reference_shape[:2]
    if mask.size != (ref_w, ref_h):
        mask = mask.resize((ref_w, ref_h), resample=Image.Resampling.NEAREST)

    mask_np = np.array(mask, dtype=np.uint8)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    if mask_np.max() <= 1:
        mask_np = (mask_np * 255).astype(np.uint8)
    return mask_np


def _compose_outputs(
    frames: List[np.ndarray],
    alpha_frames: List[np.ndarray],
    request: MatAnyOneRequest,
) -> Tuple[List[np.ndarray], List[np.ndarray], str, str, Optional[List[np.ndarray]]]:
    matting = request.normalized_matting_type
    mask_type = request.normalized_mask_type

    adjusted_alpha = []
    for alpha in alpha_frames:
        frame = alpha
        if frame.ndim == 2:
            frame = frame[..., None]
        if matting == "background":
            frame = 255 - frame
        adjusted_alpha.append(frame.astype(np.uint8))

    if mask_type in {"", "wangp"}:
        return frames, adjusted_alpha, "", "_alpha", None

    if mask_type == "greenscreen":
        green = np.zeros_like(frames[0], dtype=np.uint8)
        green[:, :, 1] = 255
        output_frames: List[np.ndarray] = []
        recomputed_alpha: List[np.ndarray] = []

        for base_frame, alpha_frame in zip(frames, adjusted_alpha):
            alpha_u16 = alpha_frame.astype(np.uint16)
            base_u16 = base_frame.astype(np.uint16)
            green_u16 = green.astype(np.uint16)
            composite = (base_u16 * (255 - alpha_u16) + green_u16 * alpha_u16) // 255
            output_frames.append(composite.astype(np.uint8))
            recomputed_alpha.append(alpha_frame)

        return output_frames, recomputed_alpha, "_greenscreen", "_alpha", None

    if mask_type == "alpha":
        from models.wan.alpha.utils import render_video, write_zip_file  # local import to avoid heavy dependency at import time

        output_frames, bgra_frames = render_video(frames, adjusted_alpha)
        return output_frames, adjusted_alpha, "_RGBA", "_alpha", bgra_frames

    raise ValueError(f"Unsupported mask type '{request.mask_type}'.")


def _save_outputs(
    request: MatAnyOneRequest,
    frames: List[np.ndarray],
    alpha_frames: List[np.ndarray],
    fps: float,
    audio_tracks: List[str],
    audio_metadata: Optional[Sequence[Dict[str, object]]],
    foreground_suffix: str,
    alpha_suffix: str,
    rgba_frames: Optional[List[np.ndarray]],
) -> MatAnyOneResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)

    media_logger = get_notifications_logger()
    timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
    base_name = sanitize_file_name(request.input_path.stem) or "matanyone"
    base_name = truncate_for_filesystem(base_name)

    suffix_parts = [timestamp]
    if request.new_dim:
        suffix_parts.append(request.new_dim.replace(" ", "_"))
    prefix = truncate_for_filesystem("_".join(filter(None, [base_name] + suffix_parts)))

    foreground_path = request.output_dir / f"{prefix}{foreground_suffix}.mp4"
    alpha_path = request.output_dir / f"{prefix}{alpha_suffix}.mp4"

    if audio_tracks:
        temp_foreground = request.output_dir / f"{prefix}{foreground_suffix}_tmp.mp4"
        temp_video_path = save_video(
            frames,
            str(temp_foreground),
            fps=fps,
            codec_type=request.codec,
            logger=media_logger,
        )
        combine_video_with_audio_tracks(
            temp_video_path,
            audio_tracks,
            str(foreground_path),
            audio_metadata=audio_metadata,
        )
        cleanup_temp_audio_files(audio_tracks)
        temp_file = Path(temp_video_path)
        if temp_file.exists():
            temp_file.unlink()
        final_foreground_path = foreground_path
    else:
        saved_path = save_video(
            frames,
            str(foreground_path),
            fps=fps,
            codec_type=request.codec,
            logger=media_logger,
        )
        final_foreground_path = Path(saved_path)

    saved_alpha_path = save_video(
        alpha_frames,
        str(alpha_path),
        fps=fps,
        codec_type=request.codec,
        logger=media_logger,
    )
    final_alpha_path = Path(saved_alpha_path)

    rgba_zip_path: Optional[Path] = None
    if rgba_frames is not None:
        from models.wan.alpha.utils import write_zip_file  # type: ignore (local import)

        rgba_zip_path = request.output_dir / f"{prefix}{foreground_suffix}.zip"
        write_zip_file(str(rgba_zip_path), rgba_frames)

    metadata = {
        "mask_type": request.mask_type,
        "matting_type": request.matting_type,
        "frames": len(frames),
        "fps": fps,
        "codec": request.codec,
        "attach_audio": bool(audio_tracks),
        "start_frame": request.start_frame,
        "end_frame": request.end_frame,
        "new_dim": request.new_dim,
    }

    metadata_config = build_metadata_config("video")
    foreground_metadata = dict(metadata)
    foreground_metadata["artifact_role"] = "foreground"
    write_metadata_bundle(
        str(final_foreground_path),
        foreground_metadata,
        config=metadata_config,
        logger=media_logger,
    )

    alpha_metadata = dict(metadata)
    alpha_metadata["artifact_role"] = "alpha"
    write_metadata_bundle(
        str(final_alpha_path),
        alpha_metadata,
        config=metadata_config,
        logger=media_logger,
    )

    return MatAnyOneResult(
        foreground_path=final_foreground_path,
        alpha_path=final_alpha_path,
        rgba_zip_path=rgba_zip_path,
        frames_processed=len(frames),
        fps=fps,
        metadata=metadata,
    )
