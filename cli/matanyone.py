#!/usr/bin/env python3
"""CLI wrapper for the headless MatAnyOne mask propagation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from cli.telemetry import configure_logging
from shared.utils.notifications import configure_notifications
from preprocessing.matanyone.app import MatAnyOneRequest, generate_masks


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
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


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

    try:
        request = MatAnyOneRequest(
            input_path=_resolve_path(args.input, "Source input"),
            template_mask_path=_resolve_path(args.template_mask, "Template mask"),
            output_dir=args.output_dir.expanduser().resolve(),
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
            metadata_mode=args.metadata_mode,
            notifier=logger.info,
        )
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to build MatAnyOne request: %s", exc)
        return 1

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

    state = {"gen": {}}
    try:
        result = generate_masks(state, request)
    except Exception as exc:  # pragma: no cover - pipeline errors surfaced to CLI
        logger.error("MatAnyOne propagation failed: %s", exc, exc_info=True)
        return 1

    logger.info("MatAnyOne propagation completed successfully.")
    logger.info("  foreground_path: %s", result.foreground_path)
    logger.info("  alpha_path: %s", result.alpha_path)
    if result.rgba_zip_path:
        logger.info("  rgba_zip_path: %s", result.rgba_zip_path)
    logger.info("  frames_processed: %d", result.frames_processed)
    logger.info("  fps: %.2f", result.fps)
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
