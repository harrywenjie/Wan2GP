#!/usr/bin/env python3
"""Minimal CLI wrapper around `wgp.generate_video`."""

from __future__ import annotations

import json

from argparse import Namespace
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from cli.arguments import parse_cli_args
from cli.telemetry import configure_logging
from shared.utils.notifications import configure_notifications

PROMPT_ENHANCER_MODE_MAP = {
    "text": "T",
    "image": "I",
    "text+image": "TI",
}
PROMPT_ENHANCER_PROVIDERS = {
    "llama3_2": 1,
    "joycaption": 2,
}
PROMPT_ENHANCER_PROVIDER_NAMES = {code: name for name, code in PROMPT_ENHANCER_PROVIDERS.items()}
PROMPT_ENHANCER_PROVIDER_NAMES[0] = None


def _resolve_prompt_enhancer_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    if mode == "off":
        return ""
    return PROMPT_ENHANCER_MODE_MAP[mode]


def parse_args(argv: Optional[Iterable[str]] = None) -> Namespace:
    return parse_cli_args(argv)


def import_wgp():
    import wgp  # type: ignore  # pylint: disable=import-error
    if hasattr(wgp, "ensure_runtime_initialized"):
        wgp.ensure_runtime_initialized()
    return wgp


def validate_input_paths(args: Namespace) -> None:
    """Ensure every provided file-based argument points to an existing file."""
    path_flags = {
        "image_start": "--image-start",
        "image_end": "--image-end",
        "video_source": "--video-source",
        "video_guide": "--video-guide",
        "image_guide": "--image-guide",
        "video_mask": "--video-mask",
        "image_mask": "--image-mask",
        "audio_guide": "--audio-guide",
        "audio_guide2": "--audio-guide2",
        "audio_source": "--audio-source",
        "settings_file": "--settings-file",
    }

    errors: List[str] = []

    for attr, flag in path_flags.items():
        path = getattr(args, attr, None)
        if path is None:
            continue
        resolved = path.expanduser()
        if not resolved.exists():
            errors.append(f"{flag} missing file: {resolved}")
            continue
        if not resolved.is_file():
            errors.append(f"{flag} must point to a file: {resolved}")
            continue
        setattr(args, attr, resolved.resolve())

    if args.image_refs:
        validated_refs: List[Path] = []
        for ref in args.image_refs:
            resolved = ref.expanduser()
            if not resolved.exists():
                errors.append(f"--image-ref missing file: {resolved}")
                continue
            if not resolved.is_file():
                errors.append(f"--image-ref must point to a file: {resolved}")
                continue
            validated_refs.append(resolved.resolve())
        args.image_refs = validated_refs if validated_refs else None

    if errors:
        bullet_list = "\n".join(f"  - {msg}" for msg in errors)
        raise SystemExit(f"Invalid file inputs:\n{bullet_list}")


def ensure_output_dirs(wgp, output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        return Path(wgp.save_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    wgp.server_config["save_path"] = str(target)
    wgp.server_config["image_save_path"] = str(target)
    wgp.save_path = str(target)
    wgp.image_save_path = str(target)
    return target


def validate_runtime_overrides(args: Namespace) -> None:
    if args.preload is not None and args.preload < 0:
        raise SystemExit("--preload expects a non-negative integer value.")
    if args.profile is not None and args.profile < -1:
        raise SystemExit("--profile expects a value >= -1.")
    if args.tea_cache_level is not None and args.tea_cache_level < 0:
        raise SystemExit("--tea-cache-level expects a non-negative value.")
    if args.tea_cache_start_perc is not None:
        if args.tea_cache_level is None or args.tea_cache_level <= 0:
            raise SystemExit("--tea-cache-start-perc requires --tea-cache-level > 0.")
        if not 0 <= args.tea_cache_start_perc <= 100:
            raise SystemExit("--tea-cache-start-perc must be between 0 and 100 (percent).")
    if args.prompt_enhancer_provider is not None:
        if args.prompt_enhancer is None:
            raise SystemExit("--prompt-enhancer-provider requires --prompt-enhancer.")
        if args.prompt_enhancer == "off":
            raise SystemExit("--prompt-enhancer-provider is meaningless when --prompt-enhancer is set to 'off'.")


def _format_override_value(value: Any) -> str:
    if value is None or value == "":
        return "<none>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalise_quant_arg(raw_value: Optional[str]) -> Optional[str]:
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return None
    lowered = stripped.casefold()
    if lowered in {"default", "auto"}:
        return None
    if lowered in {"none", "off", "disabled", "disable"}:
        return ""
    return stripped if stripped == lowered else lowered


def apply_runtime_overrides(wgp, args: Namespace, logger: Logger) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary: Dict[str, Any] = {}
    applied: Dict[str, Any] = {}

    if args.attention:
        attention = args.attention
        installed = getattr(wgp, "attention_modes_installed", [])
        supported = getattr(wgp, "attention_modes_supported", [])
        if installed and attention not in installed:
            logger.warning(
                "Attention backend '%s' is not reported as installed; generation may fall back to default.",
                attention,
            )
        elif supported and attention not in supported:
            logger.warning(
                "Attention backend '%s' is not reported as supported on this host; generation may fail.",
                attention,
            )
        wgp.attention_mode = attention
        if hasattr(wgp, "args"):
            setattr(wgp.args, "attention", attention)
        summary["attention_mode"] = attention
        applied["attention_mode"] = attention
    else:
        summary["attention_mode"] = getattr(wgp, "attention_mode", "auto")

    summary["compile"] = "transformer" if args.compile else getattr(wgp, "compile", "")
    if args.compile:
        applied["compile"] = "transformer"

    if args.preload is not None:
        preload_value = max(0, args.preload)
        if hasattr(wgp, "args"):
            setattr(wgp.args, "preload", preload_value)
        summary["preload_mb"] = preload_value
        applied["preload_mb"] = preload_value
    else:
        try:
            summary["preload_mb"] = int(getattr(getattr(wgp, "args", None), "preload", 0))
        except (TypeError, ValueError):
            summary["preload_mb"] = 0

    transformer_quant_override = _normalise_quant_arg(args.transformer_quantization)
    if transformer_quant_override is not None:
        wgp.transformer_quantization = transformer_quant_override
        applied["transformer_quantization"] = transformer_quant_override
    summary["transformer_quantization"] = getattr(wgp, "transformer_quantization", "")

    text_quant_override = _normalise_quant_arg(args.text_encoder_quantization)
    if text_quant_override is not None:
        wgp.text_encoder_quantization = text_quant_override
        applied["text_encoder_quantization"] = text_quant_override
    summary["text_encoder_quantization"] = getattr(wgp, "text_encoder_quantization", "")

    summary["transformer_dtype_policy"] = getattr(wgp, "transformer_dtype_policy", "")
    if args.fp16:
        applied["transformer_dtype_policy"] = "fp16"
    elif args.bf16:
        applied["transformer_dtype_policy"] = "bf16"

    def _runtime_flag(name: str) -> bool:
        runtime_args = getattr(wgp, "args", None)
        if runtime_args is None:
            return False
        return bool(getattr(runtime_args, name, False))

    for attr in ("save_masks", "save_quantized", "save_speakers"):
        toggle_choice = getattr(args, attr, None)
        if toggle_choice is not None:
            bool_choice = bool(toggle_choice)
            if hasattr(wgp, "args"):
                setattr(wgp.args, attr, bool_choice)
            applied[attr] = bool_choice
        summary[attr] = bool(toggle_choice) if toggle_choice is not None else _runtime_flag(attr)

    check_loras_choice = getattr(args, "check_loras", None)
    if check_loras_choice is not None:
        bool_choice = bool(check_loras_choice)
        if hasattr(wgp, "args"):
            setattr(wgp.args, "check_loras", bool_choice)
        setattr(wgp, "check_loras", bool_choice)
        applied["check_loras"] = bool_choice
    summary["check_loras"] = (
        bool(check_loras_choice)
        if check_loras_choice is not None
        else getattr(wgp, "check_loras", _runtime_flag("check_loras"))
    )

    if args.profile is not None:
        summary["profile_override"] = args.profile
        applied["profile_override"] = args.profile
    else:
        summary["profile_override"] = None

    previous_provider_code = wgp.server_config.get("enhancer_enabled", 0)
    mode_choice = args.prompt_enhancer
    provider_choice = args.prompt_enhancer_provider
    if mode_choice is not None:
        if mode_choice == "off":
            if previous_provider_code != 0:
                applied["prompt_enhancer_provider"] = "off"
            applied["prompt_enhancer_mode"] = "off"
            wgp.server_config["enhancer_enabled"] = 0
        else:
            resolved_provider = provider_choice or "llama3_2"
            provider_code = PROMPT_ENHANCER_PROVIDERS[resolved_provider]
            wgp.server_config["enhancer_enabled"] = provider_code
            applied["prompt_enhancer_mode"] = mode_choice
            if provider_choice is not None or provider_code != previous_provider_code:
                applied["prompt_enhancer_provider"] = resolved_provider

    summary["prompt_enhancer_mode"] = mode_choice
    summary["prompt_enhancer_provider"] = PROMPT_ENHANCER_PROVIDER_NAMES.get(
        wgp.server_config.get("enhancer_enabled", 0),
    )

    return summary, applied


def build_state(
    wgp,
    model_type: str,
    model_filename: str,
    loras: List[str],
    loras_presets: List[str],
) -> Dict[str, Any]:
    gen_state = {
        "queue": [],
        "file_list": [],
        "file_settings_list": [],
        "audio_file_list": [],
        "audio_file_settings_list": [],
        "selected": 0,
        "audio_selected": 0,
        "process_status": "process:main",
        "status": "",
        "status_display": False,
        "prompt_no": 0,
        "prompts_max": 0,
        "in_progress": False,
        "progress_args": None,
        "extra_windows": 0,
        "total_windows": 1,
        "window_no": 1,
        "repeat_no": 0,
        "total_generation": 0,
    }
    return {
        "model_type": model_type,
        "model_filename": model_filename,
        "advanced": False,
        "loras": loras,
        "loras_presets": loras_presets,
        "last_model_per_family": {},
        "last_model_per_type": {},
        "last_resolution_per_group": {},
        "gen": gen_state,
    }


def load_settings_defaults(
    wgp,
    state: Dict[str, Any],
    settings_path: Path,
    logger: Logger,
) -> Dict[str, Any]:
    configs, any_media, any_audio = wgp.get_settings_from_file(
        state,
        str(settings_path),
        True,
        True,
        True,
    )
    if configs is None:
        raise SystemExit(f"Unsupported settings file: {settings_path}")

    current_model = state["model_type"]
    file_model = configs.get("model_type", current_model)
    if file_model != current_model:
        raise SystemExit(
            "Settings file targets model "
            f"'{file_model}', but CLI initialised with '{current_model}'. "
            "Rerun with --model-type matching the settings file."
        )

    extracted_images = 0
    if settings_path.suffix.lower() in {".mp4", ".mkv"}:
        extracted_images = wgp.extract_and_apply_source_images(str(settings_path), configs)

    wgp.set_model_settings(state, current_model, configs)
    configs["model_type"] = current_model

    prompt_preview = (configs.get("prompt") or "").replace("\n", " ").strip()
    source_desc = "audio metadata" if any_audio else "media metadata" if any_media else "JSON settings"
    logger.info("Loaded %s defaults from %s.", source_desc, settings_path)
    if prompt_preview:
        logger.info("Settings prompt preview: %s", prompt_preview[:120])
    if extracted_images:
        logger.info("Extracted %d embedded source image(s) from %s.", extracted_images, settings_path.name)

    return configs


class LoraResolution(NamedTuple):
    activated: List[str]
    multipliers: Optional[str]
    preset_name: Optional[str]
    preset_prompt: Optional[str]
    preset_prompt_is_full: bool
    missing: List[str]


def _build_choice_maps(available: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    full: Dict[str, str] = {}
    stems: Dict[str, List[str]] = {}
    for value in available:
        name = Path(value).name
        lowered = name.casefold()
        full[lowered] = name
        stem_key = Path(name).stem.casefold()
        stems.setdefault(stem_key, []).append(name)
    return full, stems


def _resolve_choice(
    raw_name: str,
    full_map: Dict[str, str],
    stem_map: Dict[str, List[str]],
) -> Optional[str]:
    candidate = Path(raw_name).name
    lowered = candidate.casefold()
    if lowered in full_map:
        return full_map[lowered]
    stem_key = Path(candidate).stem.casefold()
    matches = stem_map.get(stem_key, [])
    if len(matches) == 1:
        return matches[0]
    return None


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalise_multiplier_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value).strip()


def _load_json_lora_preset(preset_path: Path) -> Tuple[List[str], str, Optional[str], bool]:
    with preset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_loras = data.get("activated_loras", [])
    if raw_loras is None:
        raw_loras = []
    if not isinstance(raw_loras, list):
        raise SystemExit(f"LoRA preset '{preset_path.name}' has invalid 'activated_loras' format.")
    loras = [Path(item).name for item in raw_loras if isinstance(item, str)]
    multipliers = _normalise_multiplier_value(data.get("loras_multipliers", ""))
    preset_prompt = data.get("prompt")
    if not isinstance(preset_prompt, str) or not preset_prompt.strip():
        preset_prompt = None
    full_prompt = bool(data.get("full_prompt", False))
    return loras, multipliers, preset_prompt, full_prompt


def load_lora_preset_data(
    wgp,
    model_type: str,
    preset_name: str,
    available_loras: List[str],
    lora_dir: Optional[Path],
) -> Tuple[List[str], str, Optional[str], bool]:
    if preset_name.endswith(".lset"):
        loras_choices, loras_mult, preset_prompt, full_prompt, error = wgp.extract_preset(
            model_type,
            preset_name,
            available_loras,
        )
        if error:
            raise SystemExit(f"LoRA preset '{preset_name}' failed to load: {error}")
        loras = [Path(choice).name for choice in loras_choices]
        multipliers = _normalise_multiplier_value(loras_mult)
        prompt_value = preset_prompt if isinstance(preset_prompt, str) and preset_prompt.strip() else None
        return loras, multipliers, prompt_value, bool(full_prompt)

    if lora_dir is None:
        raise SystemExit("LoRA presets require a configured LoRA directory; none found.")
    preset_path = lora_dir / preset_name
    if not preset_path.exists():
        raise SystemExit(f"LoRA preset '{preset_name}' not found in {lora_dir}.")
    return _load_json_lora_preset(preset_path)


def resolve_lora_selection(
    wgp,
    args: Namespace,
    available_loras: List[str],
    available_presets: List[str],
    model_type: str,
    lora_dir: Optional[Path],
) -> LoraResolution:
    manual_requests = args.loras or []
    preset_request = args.lora_preset
    multipliers = args.lora_multipliers

    lora_full_map, lora_stem_map = _build_choice_maps(available_loras)
    preset_full_map, preset_stem_map = _build_choice_maps(available_presets)

    selected_from_preset: List[str] = []
    preset_name: Optional[str] = None
    preset_prompt: Optional[str] = None
    preset_prompt_is_full = False
    missing_from_preset: List[str] = []

    if preset_request:
        if lora_dir is None:
            raise SystemExit("Unable to apply a LoRA preset because no LoRA directory is configured.")
        resolved_preset = _resolve_choice(preset_request, preset_full_map, preset_stem_map)
        if resolved_preset is None:
            candidate = Path(preset_request).name
            if (lora_dir / candidate).exists():
                resolved_preset = candidate
        if resolved_preset is None:
            available_display = ", ".join(sorted(available_presets)) or "none"
            raise SystemExit(
                f"Unknown LoRA preset '{preset_request}'. Available presets: {available_display}."
            )
        preset_name = resolved_preset
        preset_loras, preset_multipliers, preset_prompt_value, preset_full_prompt = load_lora_preset_data(
            wgp,
            model_type,
            resolved_preset,
            available_loras,
            lora_dir,
        )
        if multipliers is None:
            multipliers = preset_multipliers
        preset_prompt = preset_prompt_value
        preset_prompt_is_full = preset_full_prompt
        for choice in preset_loras:
            resolved_choice = _resolve_choice(choice, lora_full_map, lora_stem_map)
            if resolved_choice is not None:
                selected_from_preset.append(resolved_choice)
            else:
                normalised = Path(choice).name
                selected_from_preset.append(normalised)
                missing_from_preset.append(normalised)

    manual_selected: List[str] = []
    if manual_requests:
        if not available_loras:
            search_hint = f" in {lora_dir}" if lora_dir is not None else ""
            raise SystemExit(
                f"No LoRA weights are available for the current model{search_hint}; cannot satisfy --loras."
            )
        unresolved: List[str] = []
        for request in manual_requests:
            resolved_choice = _resolve_choice(request, lora_full_map, lora_stem_map)
            if resolved_choice is None:
                unresolved.append(request)
            else:
                manual_selected.append(resolved_choice)
        if unresolved:
            available_display = ", ".join(sorted(available_loras)) or "none"
            formatted = ", ".join(unresolved)
            raise SystemExit(
                f"Unknown LoRA selections ({formatted}). Available choices: {available_display}."
            )

    activated = _dedupe_preserve_order(selected_from_preset + manual_selected)

    return LoraResolution(
        activated=activated,
        multipliers=multipliers,
        preset_name=preset_name,
        preset_prompt=preset_prompt,
        preset_prompt_is_full=preset_prompt_is_full,
        missing=_dedupe_preserve_order(missing_from_preset),
    )


def build_params(
    wgp,
    args: Namespace,
    state: Dict[str, Any],
    activated_loras: Optional[List[str]] = None,
    lora_multipliers: Optional[str] = None,
    base_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model_type = state["model_type"]
    raw_params: Dict[str, Any] = dict(base_settings) if base_settings else {}
    raw_params.pop("state", None)
    raw_params["prompt"] = args.prompt
    raw_params["model_type"] = model_type
    raw_params["model_filename"] = state["model_filename"]
    raw_params["state"] = state
    if args.negative_prompt is not None:
        raw_params["negative_prompt"] = args.negative_prompt
    if args.resolution is not None:
        raw_params["resolution"] = args.resolution
    if args.frames is not None:
        raw_params["video_length"] = args.frames
    if args.steps is not None:
        raw_params["num_inference_steps"] = args.steps
    if args.guidance_scale is not None:
        raw_params["guidance_scale"] = args.guidance_scale
        raw_params["guidance2_scale"] = args.guidance_scale
        raw_params["guidance3_scale"] = args.guidance_scale
    if args.seed is not None:
        raw_params["seed"] = args.seed
    if args.force_fps is not None:
        raw_params["force_fps"] = args.force_fps

    if args.image_refs:
        raw_params["image_refs"] = [str(path) for path in args.image_refs]

    path_overrides = {
        "image_start": args.image_start,
        "image_end": args.image_end,
        "video_source": args.video_source,
        "video_guide": args.video_guide,
        "image_guide": args.image_guide,
        "video_mask": args.video_mask,
        "image_mask": args.image_mask,
        "audio_guide": args.audio_guide,
        "audio_guide2": args.audio_guide2,
        "audio_source": args.audio_source,
    }
    for key, value in path_overrides.items():
        if value is not None:
            raw_params[key] = str(value)

    prompt_enhancer_value = _resolve_prompt_enhancer_mode(args.prompt_enhancer)
    if prompt_enhancer_value is not None:
        raw_params["prompt_enhancer"] = prompt_enhancer_value

    if activated_loras:
        raw_params["activated_loras"] = activated_loras
    if lora_multipliers is not None:
        raw_params["loras_multipliers"] = lora_multipliers

    if args.profile is not None:
        raw_params["override_profile"] = args.profile

    tea_cache_level = args.tea_cache_level
    if tea_cache_level is not None:
        if tea_cache_level <= 0:
            raise SystemExit("--tea-cache-level must be greater than zero to enable TeaCache.")
        raw_params["skip_steps_cache_type"] = "tea"
        raw_params["skip_steps_multiplier"] = tea_cache_level
        raw_params["tea_cache_setting"] = tea_cache_level
        start_perc = args.tea_cache_start_perc
        if start_perc is not None:
            raw_params["skip_steps_start_step_perc"] = start_perc
            raw_params["tea_cache_start_step_perc"] = start_perc
    elif args.tea_cache_start_perc is not None:
        raise SystemExit("--tea-cache-start-perc requires --tea-cache-level to be provided.")

    return wgp.assemble_generation_params(raw_params, state=state)


def build_send_cmd(state: Dict[str, Any], logger: Logger):
    def send_cmd(command: str, payload: Any = None):
        if command == "progress":
            if isinstance(payload, list) and payload:
                step_info = payload[0]
                message = payload[1] if len(payload) > 1 else ""
                if isinstance(step_info, tuple) and len(step_info) == 2:
                    current, total = step_info
                    logger.info(
                        "[progress] %s/%s %s",
                        current,
                        total,
                        message or "",
                    )
                else:
                    logger.info("[progress] %s", message or step_info)
            return
        if command == "status":
            logger.info("[status] %s", payload)
            return
        if command == "info":
            logger.info("[info] %s", payload)
            return
        if command == "error":
            logger.error("[error] %s", payload)
            raise RuntimeError(str(payload))
        if command == "output":
            outputs = state["gen"].get("file_list", [])
            if outputs:
                logger.info("[output] %s", outputs[-1])
            else:
                logger.debug("[output] event emitted without recorded files")
            return
        if command in {"preview", "exit"}:
            logger.debug("[%s] %s", command, payload)
            return
        logger.debug("[event] %s: %s", command, payload)

    return send_cmd


def maybe_handle_lora_listing(
    args: Namespace,
    model_type: str,
    available_loras: List[str],
    available_presets: List[str],
    lora_dir: Optional[Path],
    logger: Logger,
) -> bool:
    handled = False
    if args.list_loras:
        logger.info("Listing LoRA weights for model '%s'", model_type)
        if available_loras:
            print(f"LoRA weights for model '{model_type}':")
            for name in available_loras:
                print(f"  - {name}")
        else:
            print(f"No LoRA weights found for model '{model_type}'.")
        if lora_dir is not None:
            print(f"Search path: {lora_dir}")
        handled = True
    if args.list_lora_presets:
        logger.info("Listing LoRA presets for model '%s'", model_type)
        if available_presets:
            print(f"LoRA presets for model '{model_type}':")
            for name in available_presets:
                print(f"  - {name}")
        else:
            print(f"No LoRA presets found for model '{model_type}'.")
        if lora_dir is not None:
            print(f"Preset path: {lora_dir}")
        handled = True
    return handled


def dry_run_report(
    params: Dict[str, Any],
    output_dir: Path,
    runtime_overrides: Dict[str, Any],
) -> None:
    summary_keys = [
        "model_type",
        "model_filename",
        "prompt",
        "negative_prompt",
        "resolution",
        "video_length",
        "num_inference_steps",
        "guidance_scale",
        "prompt_enhancer",
        "seed",
        "force_fps",
        "image_start",
        "image_end",
        "video_source",
        "video_guide",
        "image_guide",
        "video_mask",
        "image_mask",
        "audio_guide",
        "audio_guide2",
        "audio_source",
        "activated_loras",
        "loras_multipliers",
        "override_profile",
        "skip_steps_cache_type",
        "skip_steps_multiplier",
        "skip_steps_start_step_perc",
        "tea_cache_setting",
        "tea_cache_start_step_perc",
    ]
    print("Resolved CLI configuration:")
    for key in summary_keys:
        value = params.get(key)
        if key == "prompt" and isinstance(value, str):
            value = value.replace("\n", " ")
        print(f"  {key}: {value}")
    print(f"  output_dir: {output_dir}")
    if runtime_overrides:
        print("Runtime overrides:")
        for key, value in runtime_overrides.items():
            print(f"  {key}: {_format_override_value(value)}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    validate_runtime_overrides(args)
    logger = configure_logging(args.log_level)
    configure_notifications(logger)
    validate_input_paths(args)
    wgp = import_wgp()
    runtime_summary, applied_runtime = apply_runtime_overrides(wgp, args, logger)

    model_type = args.model_type
    model_filename = wgp.get_model_filename(
        model_type,
        wgp.transformer_quantization,
        wgp.transformer_dtype_policy,
    )
    lora_dir_str = wgp.get_lora_dir(model_type)
    lora_dir = Path(lora_dir_str).expanduser() if lora_dir_str else None
    preselected_preset = "" if args.lora_preset else wgp.lora_preselected_preset
    if args.lora_preset:
        wgp.lora_preselected_preset = ""
    loras, loras_presets, _, _, _, _ = wgp.setup_loras(
        model_type,
        None,
        lora_dir_str,
        preselected_preset,
        None,
    )

    if maybe_handle_lora_listing(args, model_type, loras, loras_presets, lora_dir, logger):
        return 0

    lora_resolution = resolve_lora_selection(
        wgp,
        args,
        loras,
        loras_presets,
        model_type,
        lora_dir,
    )

    output_dir = ensure_output_dirs(wgp, args.output_dir)
    state = build_state(wgp, model_type, model_filename, loras, loras_presets)
    base_settings: Optional[Dict[str, Any]] = None
    if args.settings_file is not None:
        base_settings = load_settings_defaults(wgp, state, args.settings_file, logger)
        runtime_summary["settings_file"] = str(args.settings_file)
    params = build_params(
        wgp,
        args,
        state,
        activated_loras=lora_resolution.activated or None,
        lora_multipliers=lora_resolution.multipliers,
        base_settings=base_settings,
    )

    runtime_summary["prompt_enhancer"] = params.get("prompt_enhancer")
    runtime_summary["override_profile"] = params.get("override_profile")
    if params.get("skip_steps_cache_type") == "tea":
        runtime_summary["tea_cache_level"] = params.get("skip_steps_multiplier")
        runtime_summary["tea_cache_start_perc"] = params.get("skip_steps_start_step_perc")
    else:
        runtime_summary["tea_cache_level"] = None
        runtime_summary["tea_cache_start_perc"] = None
    if args.tea_cache_level is not None:
        applied_runtime["tea_cache_level"] = args.tea_cache_level
    if args.tea_cache_start_perc is not None:
        applied_runtime["tea_cache_start_perc"] = args.tea_cache_start_perc

    if applied_runtime:
        formatted = ", ".join(f"{key}={_format_override_value(value)}" for key, value in applied_runtime.items())
        logger.info("Runtime overrides: %s", formatted)
    logger.debug("Effective runtime configuration: %s", runtime_summary)

    if args.prompt_enhancer is not None:
        provider_name = runtime_summary.get("prompt_enhancer_provider") or "disabled"
        if args.prompt_enhancer == "off":
            logger.info("Prompt enhancer disabled for this run.")
        else:
            logger.info(
                "Prompt enhancer enabled (%s mode via %s).",
                args.prompt_enhancer,
                provider_name,
            )

    if lora_resolution.activated:
        logger.info(
            "Activating LoRAs (%d): %s",
            len(lora_resolution.activated),
            ", ".join(lora_resolution.activated),
        )
    if lora_resolution.multipliers is not None:
        logger.debug(
            "LoRA multipliers string: %s",
            lora_resolution.multipliers or "<empty>",
        )
    if lora_resolution.preset_name:
        logger.info("LoRA preset applied: %s", lora_resolution.preset_name)
        if lora_resolution.preset_prompt:
            prompt_desc = (
                "full prompt" if lora_resolution.preset_prompt_is_full else "prompt prefix"
            )
            logger.info(
                "Preset supplies a %s; CLI keeps the provided --prompt. Merge manually if desired.",
                prompt_desc,
            )
        elif lora_resolution.preset_prompt_is_full:
            logger.info(
                "Preset requests a full prompt replacement; CLI keeps the user-supplied --prompt."
            )
    if lora_resolution.missing:
        logger.warning(
            "Preset references missing LoRAs: %s",
            ", ".join(lora_resolution.missing),
        )

    if args.dry_run:
        dry_run_report(params, output_dir, runtime_summary)
        if args.settings_file is not None:
            print(f"  settings_file: {args.settings_file}")
        if lora_resolution.preset_name:
            print(f"  applied_lora_preset: {lora_resolution.preset_name}")
        if lora_resolution.missing:
            print("  missing_loras_from_preset: " + ", ".join(lora_resolution.missing))
        return 0

    task = {"id": 1, "prompt": params["prompt"], "params": params.copy()}
    send_cmd = build_send_cmd(state, logger)
    wgp.generate_video(task, send_cmd, plugin_data={}, **params)
    outputs = state["gen"].get("file_list", [])
    if outputs:
        logger.info("Generation complete: %s", outputs[-1])
        print(f"Generation complete: {outputs[-1]}")
    else:
        logger.info("Generation complete but no outputs were recorded by the pipeline.")
        print("Generation complete: no outputs recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
