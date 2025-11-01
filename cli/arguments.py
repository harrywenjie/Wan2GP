from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from shared.bootstrap_defaults import DEFAULT_BOOTSTRAP_VALUES


def _coerce_int(defaults, key: str, fallback: int) -> int:
    try:
        value = defaults.get(key, fallback)
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


_PROFILE_DEFAULT = _coerce_int(DEFAULT_BOOTSTRAP_VALUES, "profile", -1)
_PRELOAD_DEFAULT = _coerce_int(DEFAULT_BOOTSTRAP_VALUES, "preload", 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless entry point for Wan2GP video generation."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Primary text prompt used to drive generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt; falls back to preset defaults when omitted.",
    )
    parser.add_argument(
        "--prompt-enhancer",
        choices=["off", "text", "image", "text+image"],
        default=None,
        help=(
            "Enable the prompt enhancer (text, image, or combined). Overrides apply to this run only; "
            "omit to reuse stored defaults from wgp_config.json."
        ),
    )
    parser.add_argument(
        "--prompt-enhancer-provider",
        choices=["llama3_2", "joycaption"],
        default=None,
        help=(
            "Select the prompt enhancer backend for this run. Defaults to llama3_2 when --prompt-enhancer "
            "is active; per-run override only."
        ),
    )
    parser.add_argument(
        "--model-type",
        default="t2v",
        help="Model identifier (e.g. t2v, i2v_2_2).",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Override output resolution (e.g. 832x480). Uses preset defaults when omitted.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Total output frames. Uses preset defaults when omitted.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps. Uses preset defaults when omitted.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG guidance scale applied uniformly across phases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to reproduce runs (-1 triggers randomized seeds).",
    )
    parser.add_argument(
        "--force-fps",
        default=None,
        help="Override output FPS (auto, control, source, or explicit integer).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory for generated assets. Overrides are per-run; next execution falls back to "
            "the configured save_path unless provided again."
        ),
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        default=None,
        help="Load generation defaults from a saved settings JSON or media file with embedded metadata.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=["metadata", "json"],
        default=None,
        help=(
            "Select how run metadata is persisted: 'metadata' embeds into media files, "
            "while 'json' writes sidecar manifests. Defaults to server_config['metadata_type']."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Override the manifest JSONL file path. Defaults to <output_dir>/manifests/run_history.jsonl "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--image-start",
        type=Path,
        default=None,
        help="Path to a start/reference image for i2v workflows.",
    )
    parser.add_argument(
        "--image-end",
        type=Path,
        default=None,
        help="Path to an ending reference image.",
    )
    parser.add_argument(
        "--image-ref",
        dest="image_refs",
        action="append",
        type=Path,
        default=None,
        help="Additional reference image. Repeat to provide multiple images.",
    )
    parser.add_argument(
        "--video-source",
        type=Path,
        default=None,
        help="Source video to extend or edit.",
    )
    parser.add_argument(
        "--video-guide",
        type=Path,
        default=None,
        help="Guidance video used for motion or style transfer.",
    )
    parser.add_argument(
        "--image-guide",
        type=Path,
        default=None,
        help="Guidance image used for conditioning.",
    )
    parser.add_argument(
        "--video-mask",
        type=Path,
        default=None,
        help="Path to a video mask to constrain edits.",
    )
    parser.add_argument(
        "--image-mask",
        type=Path,
        default=None,
        help="Path to an image mask used for static conditioning.",
    )
    parser.add_argument(
        "--audio-guide",
        type=Path,
        default=None,
        help="Audio conditioning track.",
    )
    parser.add_argument(
        "--audio-guide2",
        type=Path,
        default=None,
        help="Secondary audio conditioning track.",
    )
    parser.add_argument(
        "--audio-source",
        type=Path,
        default=None,
        help="Source audio track for edit workflows.",
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
        help="Print the resolved configuration without starting generation.",
    )
    parser.add_argument(
        "--reset-lora-cache",
        action="store_true",
        help="Clear cached LoRA discovery before the run begins.",
    )
    parser.add_argument(
        "--reset-prompt-enhancer",
        action="store_true",
        help="Reset the prompt enhancer bridge before priming for this run.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help="Expose a TCP control server on the given port for pause/resume/status commands.",
    )
    parser.add_argument(
        "--control-host",
        default="127.0.0.1",
        help="Host interface for the TCP control server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--attention",
        choices=["auto", "sdpa", "sage", "sage2", "flash", "xformers"],
        default=None,
        help="Select the attention backend; falls back to server_config when omitted.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable the torch.compile transformer path defined in server_config['compile'].",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=None,
        help=(
            "Override the VRAM/profile budget used during model initialisation for this execution "
            f"(default {_PROFILE_DEFAULT} when omitted)."
        ),
    )
    parser.add_argument(
        "--preload",
        type=int,
        default=None,
        help=(
            "Megabytes of diffusion weights to preload into VRAM "
            f"(default {_PRELOAD_DEFAULT}; 0 disables preloading)."
        ),
    )
    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 transformer weights (overrides server_config['transformer_dtype_policy']).",
    )
    dtype_group.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 transformer weights (overrides server_config['transformer_dtype_policy']).",
    )
    parser.add_argument(
        "--transformer-quantization",
        default=None,
        help="Override transformer quantisation choice (e.g. int8, fp8, none).",
    )
    parser.add_argument(
        "--text-encoder-quantization",
        default=None,
        help="Override text encoder quantisation choice (e.g. int8, fp8, none).",
    )
    parser.add_argument(
        "--tea-cache-level",
        type=float,
        default=None,
        help="Enable TeaCache by specifying a multiplier (>0 activates skip-step caching).",
    )
    parser.add_argument(
        "--tea-cache-start-perc",
        type=float,
        default=None,
        help="Percentage of the denoising schedule to apply TeaCache skipping from (0-100).",
    )
    parser.add_argument(
        "--save-masks",
        dest="save_masks",
        action="store_true",
        default=None,
        help="Persist intermediate mask assets according to the legacy UI toggle. Defaults to stored preference.",
    )
    parser.add_argument(
        "--no-save-masks",
        dest="save_masks",
        action="store_false",
        default=None,
        help="Disable mask persistence for this run.",
    )
    parser.add_argument(
        "--save-quantized",
        dest="save_quantized",
        action="store_true",
        default=None,
        help="Keep freshly quantised transformer weights on disk. Defaults to stored preference.",
    )
    parser.add_argument(
        "--no-save-quantized",
        dest="save_quantized",
        action="store_false",
        default=None,
        help="Skip saving quantised transformers after this run.",
    )
    parser.add_argument(
        "--save-speakers",
        dest="save_speakers",
        action="store_true",
        default=None,
        help="Persist extracted speaker tracks (for MMAudio/Chatterbox) when available. Defaults to stored preference.",
    )
    parser.add_argument(
        "--no-save-speakers",
        dest="save_speakers",
        action="store_false",
        default=None,
        help="Disable speaker track persistence for this run.",
    )
    parser.add_argument(
        "--check-loras",
        dest="check_loras",
        action="store_true",
        default=None,
        help="Validate LoRA file availability before generation and exit on missing assets.",
    )
    parser.add_argument(
        "--no-check-loras",
        dest="check_loras",
        action="store_false",
        default=None,
        help="Bypass LoRA availability checks even if stored defaults enable them.",
    )
    parser.add_argument(
        "--list-loras",
        action="store_true",
        help="List available LoRA weights for the selected model type and exit.",
    )
    parser.add_argument(
        "--list-lora-presets",
        action="store_true",
        help="List available LoRA presets/settings for the selected model type and exit.",
    )
    parser.add_argument(
        "--loras",
        dest="loras",
        action="append",
        default=None,
        help="Activate a LoRA by file name. Repeat to enable multiple LoRAs.",
    )
    parser.add_argument(
        "--lora-preset",
        default=None,
        help="Apply a LoRA preset (.lset or .json) from the model's LoRA directory.",
    )
    parser.add_argument(
        "--lora-multipliers",
        default=None,
        help="Override the LoRA multipliers string. Accepts the same syntax as legacy UI presets.",
    )
    parser.set_defaults(
        save_masks=None,
        save_quantized=None,
        save_speakers=None,
        check_loras=None,
    )
    return parser


def parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)
