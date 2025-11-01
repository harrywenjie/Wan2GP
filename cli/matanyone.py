#!/usr/bin/env python3
"""CLI wrapper for the headless MatAnyOne mask propagation pipeline."""

from __future__ import annotations

import argparse
import uuid
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cli.manifest import (
    ManifestRecorder,
    build_matanyone_artifacts,
    canonicalize_structure,
    resolve_manifest_path,
    write_manifest_entry,
)
from cli.telemetry import configure_logging
from shared.utils.notifications import configure_notifications
from preprocessing.matanyone.app import MatAnyOneRequest, MatAnyOneResult, generate_masks
from core.production_manager import MetadataState, ProductionManager

try:  # Optional import for typing without runtime dependency
    from core.io.media import MediaPersistenceContext
except Exception:  # pragma: no cover - typing fallback
    MediaPersistenceContext = None  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MatAnyOne mask propagation headlessly. "
            "Accepts a source video or image and a template mask to generate foreground/alpha outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source media path (video or image) used for mask propagation.",
    )
    parser.add_argument(
        "--template-mask",
        type=Path,
        required=True,
        help="Grayscale mask aligned to the first frame of the source media.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mask_outputs"),
        help="Directory where propagated masks and composites are written.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame (inclusive) to propagate. Defaults to 0.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame (exclusive) to propagate. Defaults to the end of the source.",
    )
    parser.add_argument(
        "--new-dim",
        default="",
        help="Optional resize directive (e.g. '1080p outer'). Leave empty to preserve source resolution.",
    )
    parser.add_argument(
        "--matting",
        choices=["foreground", "background"],
        default="foreground",
        help="Select whether to keep the foreground or background region.",
    )
    parser.add_argument(
        "--mask-type",
        choices=["wangp", "greenscreen", "alpha"],
        default="wangp",
        help="Output format: raw mask (wangp), composited greenscreen, or RGBA zip bundle.",
    )
    parser.add_argument(
        "--erode-kernel",
        type=int,
        default=0,
        help="Erode kernel size applied before propagation.",
    )
    parser.add_argument(
        "--dilate-kernel",
        type=int,
        default=0,
        help="Dilate kernel size applied after propagation.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Number of warm-up frames to stabilise the model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device identifier (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument(
        "--codec",
        default="libx264_8",
        help="FFmpeg codec used when writing MP4 outputs.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=["metadata", "json"],
        default="metadata",
        help="Select metadata persistence: embed into media files or write JSON sidecars.",
    )
    parser.add_argument(
        "--no-audio",
        dest="attach_audio",
        action="store_false",
        default=True,
        help="Disable audio reattachment when the source contains audio tracks.",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity for CLI telemetry output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved request without running propagation.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path to append manifest entries (defaults to <output_dir>/manifests/run_history.jsonl).",
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def _normalize_audio_metadata(value: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    normalized: List[Dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, Mapping):
            normalized.append(dict(entry))
    return normalized or None


def _resolve_runtime_contexts(
    logger,
    metadata_choice: str,
) -> Tuple[Optional[MetadataState], Optional["MediaPersistenceContext"]]:
    metadata_state: Optional[MetadataState] = None
    media_context: Optional["MediaPersistenceContext"] = None

    try:
        import wgp  # type: ignore  # pylint: disable=import-error
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        logger.debug("MatAnyOne runtime contexts unavailable (import error): %s", exc)
        return metadata_state, media_context

    if hasattr(wgp, "ensure_runtime_initialized"):
        try:
            wgp.ensure_runtime_initialized()
        except Exception as exc:  # pragma: no cover - runtime init is environment-specific
            logger.debug("MatAnyOne runtime contexts unavailable (runtime init): %s", exc)
            return metadata_state, media_context

    try:
        manager = ProductionManager(wgp_module=wgp)
    except Exception as exc:  # pragma: no cover - ProductionManager surface may evolve
        logger.debug("MatAnyOne runtime contexts unavailable (manager error): %s", exc)
        return metadata_state, media_context

    try:
        metadata_state = manager.metadata_state(choice_override=metadata_choice)
    except Exception as exc:  # pragma: no cover - ProductionManager surface may evolve
        logger.debug("MatAnyOne metadata state unavailable (metadata error): %s", exc)

    try:
        media_context = manager.media_context()
    except Exception as exc:  # pragma: no cover - context wiring depends on runtime state
        logger.debug("MatAnyOne media context unavailable (context error): %s", exc)

    return metadata_state, media_context


def _build_manifest_inputs(
    args: argparse.Namespace,
    request: Optional[MatAnyOneRequest],
) -> Dict[str, Any]:
    if request is not None:
        return {
            "input_path": str(request.input_path),
            "template_mask_path": str(request.template_mask_path),
            "output_dir": str(request.output_dir),
            "start_frame": request.start_frame,
            "end_frame": request.end_frame,
            "new_dim": request.new_dim,
            "mask_type": request.mask_type,
            "matting_type": request.matting_type,
            "erode_kernel_size": request.erode_kernel_size,
            "dilate_kernel_size": request.dilate_kernel_size,
            "warmup_frames": request.warmup_frames,
            "device": request.device,
            "codec": request.codec,
            "attach_audio": request.attach_audio,
            "metadata_mode": request.metadata_mode,
        }

    return {
        "input_path": str(args.input.expanduser()),
        "template_mask_path": str(args.template_mask.expanduser()),
        "output_dir": str(args.output_dir.expanduser()),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "new_dim": args.new_dim,
        "mask_type": args.mask_type,
        "matting_type": args.matting,
        "erode_kernel_size": args.erode_kernel,
        "dilate_kernel_size": args.dilate_kernel,
        "warmup_frames": args.warmup_frames,
        "device": args.device,
        "codec": args.codec,
        "attach_audio": args.attach_audio,
        "metadata_mode": args.metadata_mode,
    }


def _emit_manifest_entry(
    *,
    path: Optional[Path],
    run_id: str,
    output_dir: Path,
    metadata_mode: str,
    status: str,
    inputs: Dict[str, Any],
    recorder: Optional[ManifestRecorder],
    result: Optional[MatAnyOneResult],
    error: Optional[str],
) -> None:
    if path is None:
        return

    entry: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(resolve_manifest_path(output_dir)),
        "metadata_mode": metadata_mode,
        "status": status,
        "inputs": canonicalize_structure(inputs),
        "adapter_payload_hashes": {},
    }

    if status == "success" and result is not None:
        metadata_payload = result.metadata if isinstance(result.metadata, dict) else {}
        codec = metadata_payload.get("codec")
        container = metadata_payload.get("container")
        audio_metadata = _normalize_audio_metadata(metadata_payload.get("audio_tracks"))
        captures = recorder.captures if recorder is not None else ()
        entry["artifacts"] = build_matanyone_artifacts(
            foreground_path=result.foreground_path,
            alpha_path=result.alpha_path,
            rgba_zip_path=result.rgba_zip_path,
            frames_processed=result.frames_processed,
            fps=result.fps,
            metadata_mode=metadata_mode,
            captures=captures,
            codec=codec,
            container=container,
            audio_metadata=audio_metadata,
        )
    else:
        entry["artifacts"] = []
        if error:
            entry["error"] = error

    write_manifest_entry(path, entry)


def _resolve_path(path: Path, description: str) -> Path:
    resolved = path.expanduser()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise SystemExit(f"{description} must be a file: {resolved}")
    return resolved.resolve()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.log_level)
    configure_notifications(logger)

    metadata_state, media_context = _resolve_runtime_contexts(logger, args.metadata_mode)
    metadata_mode = (
        metadata_state.choice if metadata_state and metadata_state.choice else args.metadata_mode
    )

    resolved_output_dir = args.output_dir.expanduser().resolve()
    manifest_recorder: Optional[ManifestRecorder] = None
    manifest_path: Optional[Path] = None
    run_id = str(uuid.uuid4())
    if not args.dry_run:
        manifest_recorder = ManifestRecorder()
        default_manifest = resolved_output_dir / "manifests" / "run_history.jsonl"
        manifest_target = args.manifest_path if args.manifest_path is not None else default_manifest
        manifest_path = resolve_manifest_path(manifest_target)
        if media_context is not None:
            media_context = manifest_recorder.wrap(media_context)

    request: Optional[MatAnyOneRequest] = None
    manifest_status = "success"
    manifest_error: Optional[str] = None

    try:
        request = MatAnyOneRequest(
            input_path=_resolve_path(args.input, "Source input"),
            template_mask_path=_resolve_path(args.template_mask, "Template mask"),
            output_dir=resolved_output_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            new_dim=args.new_dim or "",
            matting_type=args.matting,
            mask_type=args.mask_type,
            erode_kernel_size=args.erode_kernel,
            dilate_kernel_size=args.dilate_kernel,
            warmup_frames=args.warmup_frames,
            device=args.device,
            attach_audio=args.attach_audio,
            codec=args.codec,
            metadata_mode=metadata_mode,
            metadata_state=metadata_state,
            media_context=media_context,
            notifier=logger.info,
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to build MatAnyOne request: %s", exc)
        manifest_status = "error"
        manifest_error = f"Failed to build request: {exc}"
        inputs_payload = _build_manifest_inputs(args, None)
        _emit_manifest_entry(
            path=manifest_path,
            run_id=run_id,
            output_dir=resolved_output_dir,
            metadata_mode=metadata_mode,
            status=manifest_status,
            inputs=inputs_payload,
            recorder=manifest_recorder,
            result=None,
            error=manifest_error,
        )
        return 1

    metadata_mode = request.metadata_mode

    request.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("MatAnyOne request ready:")
    logger.info("  input: %s", request.input_path)
    logger.info("  template_mask: %s", request.template_mask_path)
    logger.info("  output_dir: %s", request.output_dir)
    logger.info("  frame_range: [%s, %s)", request.start_frame, request.end_frame or "end")
    logger.info("  new_dim: %s", request.new_dim or "<unchanged>")
    logger.info("  matting: %s", request.matting_type)
    logger.info("  mask_type: %s", request.mask_type)
    logger.info(
        "  kernels: erode=%d dilate=%d warmup=%d",
        request.erode_kernel_size,
        request.dilate_kernel_size,
        request.warmup_frames,
    )
    logger.info("  device: %s", request.device)
    logger.info("  codec: %s", request.codec)
    logger.info("  metadata_mode: %s", request.metadata_mode)
    logger.info("  attach_audio: %s", request.attach_audio)

    if args.dry_run:
        print("MatAnyOne dry-run successful; no propagation executed.")
        return 0

    manifest_inputs = _build_manifest_inputs(args, request)

    state = {"gen": {}}
    result: Optional[MatAnyOneResult] = None
    exit_code = 0
    try:
        result = generate_masks(state, request)
    except Exception as exc:  # pragma: no cover - pipeline errors surfaced to CLI
        manifest_status = "error"
        manifest_error = str(exc)
        exit_code = 1
        logger.error("MatAnyOne propagation failed: %s", exc, exc_info=True)

    _emit_manifest_entry(
        path=manifest_path,
        run_id=run_id,
        output_dir=request.output_dir,
        metadata_mode=metadata_mode,
        status=manifest_status,
        inputs=manifest_inputs,
        recorder=manifest_recorder,
        result=result if manifest_status == "success" else None,
        error=manifest_error,
    )

    if exit_code != 0 or result is None:
        return exit_code

    logger.info("MatAnyOne propagation completed successfully.")
    logger.info("  foreground_path: %s", result.foreground_path)
    logger.info("  alpha_path: %s", result.alpha_path)
    if result.rgba_zip_path:
        logger.info("  rgba_zip_path: %s", result.rgba_zip_path)
    logger.info("  frames_processed: %d", result.frames_processed)
    logger.info("  fps: %.2f", result.fps)
    audio_metadata = _normalize_audio_metadata(result.metadata.get("audio_tracks"))
    if audio_metadata:
        logger.info("  audio_tracks: %d", len(audio_metadata))
        for index, track in enumerate(audio_metadata, start=1):
            raw_path = track.get("path", "<unknown>")
            path_str = str(raw_path)
            sample_rate = track.get("sample_rate")
            sample_rate_text = str(sample_rate) if sample_rate is not None else "<unknown>"
            duration_value = track.get("duration")
            try:
                duration_value = float(duration_value) if duration_value is not None else None
            except (TypeError, ValueError):
                duration_value = None
            duration_text = f"{duration_value:.2f}s" if duration_value is not None else "<unknown>"
            language = track.get("language") or "<unknown>"
            logger.info(
                "    #%d path=%s sample_rate=%s duration=%s language=%s",
                index,
                path_str,
                sample_rate_text,
                duration_text,
                language,
            )

    for key, value in result.metadata.items():
        logger.debug("  metadata[%s]=%s", key, value)
    print(f"Foreground written to: {result.foreground_path}")
    print(f"Alpha mask written to: {result.alpha_path}")
    if result.rgba_zip_path:
        print(f"RGBA archive written to: {result.rgba_zip_path}")
    if request.metadata_mode == "json":
        foreground_json = result.foreground_path.with_suffix(".json")
        alpha_json = result.alpha_path.with_suffix(".json")
        if foreground_json.exists():
            print(f"Foreground metadata written to: {foreground_json}")
        if alpha_json.exists():
            print(f"Alpha metadata written to: {alpha_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
