from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.io.media import (
    AudioSaveConfig,
    ImageSaveConfig,
    MediaPersistenceContext,
    VideoSaveConfig,
)


@dataclass(frozen=True)
class ArtifactCapture:
    """Record a persisted artifact for later manifest assembly."""

    path: Path
    kind: str
    config: Optional[Any] = None


def _clone_config(config: Any) -> Any:
    """Return a shallow clone of the supplied dataclass config."""

    if config is None:
        return None
    cloned = replace(config)
    if hasattr(cloned, "extra_params"):
        cloned.extra_params = dict(getattr(config, "extra_params", {}))
    return cloned


def _apply_overrides(config: Any, overrides: Optional[Mapping[str, Any]]) -> Any:
    """Apply dataclass field overrides in place, mirroring MediaPersistenceContext logic."""

    if not overrides:
        return config
    fields = getattr(config, "__dataclass_fields__", {})
    for key, value in overrides.items():
        if key not in fields:
            raise AttributeError(f"{config.__class__.__name__} has no field named '{key}'")
        setattr(config, key, value)
    return config


class _RecordingMediaContext:
    """Proxy ``MediaPersistenceContext`` that records persisted artifacts."""

    def __init__(self, context: MediaPersistenceContext, recorder: "ManifestRecorder") -> None:
        self._context = context
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)

    def save_video(
        self,
        data: Any,
        target_path: Optional[str],
        *,
        logger: Optional[Any] = None,
        config: Optional[VideoSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        effective = _clone_config(config) if config is not None else self._context.video_config()
        effective = _apply_overrides(effective, overrides)
        result = self._context.save_video(
            data,
            target_path,
            logger=logger,
            config=effective,
        )
        if result:
            self._recorder.record(result, "video", effective)
        return result

    def save_image(
        self,
        data: Any,
        target_path: str,
        *,
        logger: Optional[Any] = None,
        config: Optional[ImageSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> str:
        effective = _clone_config(config) if config is not None else self._context.image_config()
        effective = _apply_overrides(effective, overrides)
        result = self._context.save_image(
            data,
            target_path,
            logger=logger,
            config=effective,
        )
        self._recorder.record(result, "image", effective)
        return result

    def save_audio(
        self,
        data: Any,
        target_path: Optional[str],
        *,
        logger: Optional[Any] = None,
        config: Optional[AudioSaveConfig] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        effective = _clone_config(config) if config is not None else self._context.audio_config()
        effective = _apply_overrides(effective, overrides)
        result = self._context.save_audio(
            data,
            target_path,
            logger=logger,
            config=effective,
        )
        if result:
            self._recorder.record(result, "audio", effective)
        return result

    def save_mask_archive(
        self,
        frames: Any,
        base_output_path: str,
        *,
        force: bool = False,
    ) -> Optional[str]:
        result = self._context.save_mask_archive(frames, base_output_path, force=force)
        if result:
            self._recorder.record(result, "mask_archive", None)
        return result


class ManifestRecorder:
    """Capture artifact persistence events to enrich manifest entries."""

    def __init__(self) -> None:
        self._captures: List[ArtifactCapture] = []

    def wrap(self, context: MediaPersistenceContext) -> MediaPersistenceContext:
        return _RecordingMediaContext(context, self)

    def record(self, path: Optional[str], kind: str, config: Optional[Any]) -> None:
        if not path:
            return
        self._captures.append(
            ArtifactCapture(
                path=Path(path),
                kind=kind,
                config=_clone_config(config),
            )
        )

    @property
    def captures(self) -> Sequence[ArtifactCapture]:
        return tuple(self._captures)


def canonicalize_structure(value: Any) -> Any:
    """Convert nested values into JSON-serialisable primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): canonicalize_structure(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize_structure(item) for item in value]
    return str(value)


def compute_adapter_hashes(payloads: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Produce SHA-256 hashes for adapter payloads using canonical JSON serialisation."""

    if not payloads:
        return {}
    hashes: Dict[str, Dict[str, Any]] = {}
    for name, payload in payloads.items():
        canonical = canonicalize_structure(payload)
        serialised = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        hashes[name] = {"sha256": digest, "source_bytes": len(serialised)}
    return hashes


def write_manifest_entry(path: Path, entry: Dict[str, Any]) -> None:
    """Append a manifest entry to the JSONL file at ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialised)
        handle.write("\n")


def resolve_manifest_path(value: Any) -> Path:
    """Expand and resolve manifest-related filesystem paths."""

    path = value if isinstance(value, Path) else Path(str(value))
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser()


def _pop_capture(
    capture_index: Dict[Tuple[str, Path], List[ArtifactCapture]],
    kind: str,
    target_path: Path,
) -> Optional[ArtifactCapture]:
    resolved = resolve_manifest_path(target_path)
    key = (kind, resolved)
    captures = capture_index.get(key)
    if captures:
        capture = captures.pop(0)
        if not captures:
            capture_index.pop(key, None)
        return capture
    if kind == "video":
        tmp_key = (kind, resolved.with_stem(f"{resolved.stem}_tmp"))
        captures = capture_index.get(tmp_key)
        if captures:
            capture = captures.pop(0)
            if not captures:
                capture_index.pop(tmp_key, None)
            return capture
    return None


def build_matanyone_artifacts(
    *,
    foreground_path: Path,
    alpha_path: Path,
    rgba_zip_path: Optional[Path],
    frames_processed: int,
    fps: float,
    metadata_mode: str,
    captures: Sequence[ArtifactCapture],
    codec: Optional[str],
    container: Optional[str],
) -> List[Dict[str, Any]]:
    """Assemble MatAnyOne artifact descriptors for manifest emission."""

    capture_index: Dict[Tuple[str, Path], List[ArtifactCapture]] = defaultdict(list)
    for capture in captures:
        resolved = resolve_manifest_path(capture.path)
        capture_index[(capture.kind, resolved)].append(capture)

    def _build_video_entry(role: str, path: Path) -> Dict[str, Any]:
        capture = _pop_capture(capture_index, "video", path)
        resolved = resolve_manifest_path(path)
        entry_container = container
        entry_codec = codec
        effective_fps = fps
        if capture and capture.config is not None:
            entry_container = getattr(capture.config, "container", entry_container) or entry_container
            entry_codec = getattr(capture.config, "codec_type", entry_codec) or entry_codec
            effective_fps = getattr(capture.config, "fps", effective_fps) or effective_fps

        if entry_container:
            normalized_container = str(entry_container)
            entry_container = normalized_container[1:] if normalized_container.startswith(".") else normalized_container
        else:
            entry_container = resolved.suffix.lstrip(".") or None

        duration_value: Optional[float] = None
        try:
            if frames_processed is not None and effective_fps:
                duration_value = frames_processed / float(effective_fps)
        except (ZeroDivisionError, TypeError, ValueError):
            duration_value = None

        metadata_sidecar: Optional[str] = None
        if metadata_mode == "json":
            metadata_sidecar = str(resolved.with_suffix(".json"))

        return {
            "role": role,
            "path": str(resolved),
            "container": entry_container,
            "codec": entry_codec,
            "frames": frames_processed,
            "duration_s": duration_value,
            "metadata_sidecar": metadata_sidecar,
        }

    artifacts: List[Dict[str, Any]] = [
        _build_video_entry("mask_foreground", foreground_path),
        _build_video_entry("mask_alpha", alpha_path),
    ]

    for (kind, resolved), captures_list in list(capture_index.items()):
        if kind != "audio":
            continue
        while captures_list:
            capture = captures_list.pop(0)
            entry_container = None
            entry_codec = None
            if capture.config is not None:
                entry_container = getattr(capture.config, "format", None)
                entry_codec = getattr(capture.config, "subtype", None)
            normalized_container = str(entry_container) if entry_container else None
            if normalized_container:
                entry_container = (
                    normalized_container[1:]
                    if normalized_container.startswith(".")
                    else normalized_container
                )
            else:
                entry_container = resolved.suffix.lstrip(".") or None
            metadata_sidecar = str(resolved.with_suffix(".json")) if metadata_mode == "json" else None
            artifacts.append(
                {
                    "role": "audio",
                    "path": str(resolved),
                    "container": entry_container,
                    "codec": entry_codec,
                    "frames": None,
                    "duration_s": None,
                    "metadata_sidecar": metadata_sidecar,
                }
            )
        capture_index.pop((kind, resolved), None)

    if rgba_zip_path is not None:
        _pop_capture(capture_index, "mask_archive", rgba_zip_path)
        resolved = resolve_manifest_path(rgba_zip_path)
        artifacts.append(
            {
                "role": "rgba_archive",
                "path": str(resolved),
                "container": resolved.suffix.lstrip(".") or None,
                "codec": None,
                "frames": None,
                "duration_s": None,
                "metadata_sidecar": None,
            }
        )

    return artifacts
