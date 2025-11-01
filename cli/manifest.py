from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

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
