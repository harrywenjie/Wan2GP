from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple, Literal, Mapping

from core.lora.manager import LoRAHydrationResult, LoRAInjectionManager
from core.prompt_enhancer.bridge import PromptEnhancerBridge
from shared.utils.audio_metadata import read_audio_metadata
from shared.utils.audio_video import read_image_metadata
from shared.utils.loras_mutipliers import merge_loras_settings
from shared.utils.process_locks import get_gen_info
from shared.utils.video_metadata import read_metadata_from_video
from shared.utils.utils import (
    has_audio_file_extension,
    has_image_file_extension,
    has_video_file_extension,
)

SettingsLoader = Callable[..., Tuple[Optional[Dict[str, Any]], bool, bool]]


def _normalise_for_hash(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_for_hash(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_for_hash(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_server_config_snapshot(server_config: Mapping[str, Any]) -> str:
    normalised = _normalise_for_hash(server_config)
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SaveInputsPayload:
    image_mask_guide: Optional[Dict[str, Any]] = None
    lset_name: Optional[str] = None
    image_mode: int = 0
    prompt: str = ""
    negative_prompt: Optional[str] = None
    resolution: Optional[str] = None
    video_length: Optional[int] = None
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    force_fps: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance2_scale: Optional[float] = None
    guidance3_scale: Optional[float] = None
    switch_threshold: Optional[float] = None
    switch_threshold2: Optional[float] = None
    guidance_phases: Optional[int] = None
    model_switch_phase: Optional[int] = None
    audio_guidance_scale: Optional[float] = None
    flow_shift: Optional[float] = None
    sample_solver: Optional[str] = None
    embedded_guidance_scale: Optional[float] = None
    repeat_generation: Optional[int] = None
    multi_prompts_gen_type: Optional[str] = None
    multi_images_gen_type: Optional[str] = None
    skip_steps_cache_type: Optional[str] = None
    skip_steps_multiplier: Optional[float] = None
    skip_steps_start_step_perc: Optional[float] = None
    loras_choices: Optional[List[str]] = None
    loras_multipliers: Optional[str] = None
    image_prompt_type: Optional[str] = None
    image_start: Optional[Any] = None
    image_end: Optional[Any] = None
    model_mode: Optional[str] = None
    video_source: Optional[str] = None
    keep_frames_video_source: Optional[Any] = None
    video_guide_outpainting: Optional[Any] = None
    video_prompt_type: Optional[str] = None
    image_refs: Optional[List[str]] = None
    frames_positions: Optional[Any] = None
    video_guide: Optional[str] = None
    image_guide: Optional[Any] = None
    keep_frames_video_guide: Optional[Any] = None
    denoising_strength: Optional[float] = None
    video_mask: Optional[Any] = None
    image_mask: Optional[Any] = None
    control_net_weight: Optional[float] = None
    control_net_weight2: Optional[float] = None
    control_net_weight_alt: Optional[float] = None
    mask_expand: Optional[float] = None
    audio_guide: Optional[str] = None
    audio_guide2: Optional[str] = None
    audio_source: Optional[str] = None
    audio_prompt_type: Optional[str] = None
    speakers_locations: Optional[str] = None
    sliding_window_size: Optional[int] = None
    sliding_window_overlap: Optional[float] = None
    sliding_window_color_correction_strength: Optional[float] = None
    sliding_window_overlap_noise: Optional[float] = None
    sliding_window_discard_last_frames: Optional[int] = None
    image_refs_relative_size: Optional[float] = None
    remove_background_images_ref: Optional[bool] = None
    temporal_upsampling: Optional[bool] = None
    spatial_upsampling: Optional[bool] = None
    film_grain_intensity: Optional[float] = None
    film_grain_saturation: Optional[float] = None
    MMAudio_setting: Optional[str] = None
    MMAudio_prompt: Optional[str] = None
    MMAudio_neg_prompt: Optional[str] = None
    RIFLEx_setting: Optional[str] = None
    NAG_scale: Optional[float] = None
    NAG_tau: Optional[float] = None
    NAG_alpha: Optional[float] = None
    slg_switch: Optional[bool] = None
    slg_layers: Optional[str] = None
    slg_start_perc: Optional[float] = None
    slg_end_perc: Optional[float] = None
    apg_switch: Optional[bool] = None
    cfg_star_switch: Optional[bool] = None
    cfg_zero_step: Optional[int] = None
    prompt_enhancer: Optional[str] = None
    min_frames_if_references: Optional[int] = None
    override_profile: Optional[int] = None
    pace: Optional[float] = None
    exaggeration: Optional[float] = None
    temperature: Optional[float] = None
    mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SaveInputsRequest:
    target: Literal["state", "settings", "metadata"]
    payload: SaveInputsPayload
    state: Dict[str, Any]
    plugin_data: Dict[str, Any] = field(default_factory=dict)

    def to_inputs_dict(self) -> Dict[str, Any]:
        inputs = self.payload.to_dict()
        inputs["state"] = self.state
        inputs["plugin_data"] = self.plugin_data or {}
        return inputs


class TaskInputManager:
    def __init__(
        self,
        *,
        server_config: MutableMapping[str, Any],
        settings_version: float,
        get_model_record: Callable[[str], str],
        get_model_name: Callable[[str], str],
        get_model_def: Callable[[str], Dict[str, Any]],
        get_base_model_type: Callable[[str], str],
        get_model_family: Callable[[str], str],
        test_vace_module: Callable[[str], bool],
        test_class_t2v: Callable[[str], bool],
        test_any_sliding_window: Callable[[str], bool],
        any_audio_track: Callable[[str], bool],
        get_lora_dir: Callable[[str], str],
        settings_loader: Optional[SettingsLoader],
        get_settings_file_name: Callable[[str], str],
        set_model_settings: Callable[[Dict[str, Any], str, Dict[str, Any]], None],
        notify_info: Callable[[str], None],
        lock: Lock,
        get_model_type: Callable[[str], Optional[str]],
        are_model_types_compatible: Callable[[str, str], bool],
        get_default_settings: Callable[[str], Dict[str, Any]],
        get_model_settings: Callable[[Dict[str, Any], str], Optional[Dict[str, Any]]],
        fix_settings: Callable[[str, Dict[str, Any], float], None],
        model_types: Sequence[str],
        loras_cache_path: Optional[Path] = None,
        lora_manager: Optional[LoRAInjectionManager] = None,
        prompt_enhancer: Optional[PromptEnhancerBridge] = None,
    ) -> None:
        self._server_config = server_config
        self._settings_version = settings_version
        self._get_model_record = get_model_record
        self._get_model_name = get_model_name
        self._get_model_def = get_model_def
        self._get_base_model_type = get_base_model_type
        self._get_model_family = get_model_family
        self._test_vace_module = test_vace_module
        self._test_class_t2v = test_class_t2v
        self._test_any_sliding_window = test_any_sliding_window
        self._any_audio_track = any_audio_track
        self._get_lora_dir = get_lora_dir
        self._settings_loader = settings_loader
        self._get_settings_file_name = get_settings_file_name
        self._set_model_settings = set_model_settings
        self._notify_info = notify_info
        self._lock = lock
        self._get_model_type = get_model_type
        self._are_model_types_compatible = are_model_types_compatible
        self._get_default_settings = get_default_settings
        self._get_model_settings = get_model_settings
        self._fix_settings = fix_settings
        self._model_types = frozenset(model_types)
        self._loras_cache_path = loras_cache_path or Path("loras_url_cache.json")
        self._loras_url_cache: Optional[Dict[str, str]] = None
        self._lora_manager = lora_manager
        self._prompt_enhancer_bridge = prompt_enhancer
        self._lora_hydrations: Dict[str, LoRAHydrationResult] = {}

    @property
    def server_config(self) -> MutableMapping[str, Any]:
        return self._server_config

    def lora_inventory(self, model_type: str, *, refresh: bool = False) -> Optional[LoRAHydrationResult]:
        if self._lora_manager is None:
            return None
        hydration = self._lora_manager.hydrate(model_type, refresh=refresh)
        self._lora_hydrations[model_type] = hydration
        return hydration

    def _server_config_hash(self) -> str:
        server_config_obj = self._server_config
        if isinstance(server_config_obj, Mapping):
            mapping: Mapping[str, Any] = server_config_obj
        else:
            mapping = dict(server_config_obj or {})  # type: ignore[arg-type]
        return _hash_server_config_snapshot(mapping)

    def build_lora_payload(
        self,
        model_type: str,
        activated_loras: Optional[Sequence[str]],
        *,
        multipliers: Optional[str],
        refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        hydration = self.lora_inventory(model_type, refresh=refresh)
        if hydration is None:
            return None
        library = hydration.library
        payload: Dict[str, Any] = {
            "model_type": library.model_type,
            "server_config_hash": library.server_config_hash,
            "available": list(library.loras),
            "presets": list(library.presets),
            "default_choices": list(hydration.default_choices),
            "default_multipliers": hydration.default_multipliers,
            "default_prompt": hydration.default_prompt,
            "default_preset": hydration.default_preset,
            "activated": list(activated_loras or ()),
        }
        if library.lora_dir is not None:
            payload["lora_dir"] = str(library.lora_dir)
        if multipliers is not None:
            payload["multipliers"] = str(multipliers)
        return payload

    def resolve_prompt_enhancer(self, prompt_enhancer: Optional[str]) -> Optional[Dict[str, Any]]:
        if prompt_enhancer is None:
            return None
        server_config = self._server_config or {}
        provider_code = int(server_config.get("enhancer_enabled", 0) or 0)
        enhancer_mode = int(server_config.get("enhancer_mode", 0) or 0)
        payload: Dict[str, Any] = {
            "mode": prompt_enhancer,
            "provider": provider_code,
            "enhancer_mode": enhancer_mode,
            "server_config_hash": self._server_config_hash(),
        }
        bridge = self._prompt_enhancer_bridge
        if bridge is not None:
            payload["cache_state"] = bridge.snapshot_state()
        return payload

    def _lora_dir_for_model(self, model_type: str) -> Optional[str]:
        hydration = self.lora_inventory(model_type)
        if hydration is not None and hydration.library.lora_dir is not None:
            return str(hydration.library.lora_dir)
        return self._get_lora_dir(model_type)

    def prepare_inputs_dict(
        self,
        target: str,
        inputs: Dict[str, Any],
        model_type: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = inputs.pop("state")
        inputs.pop("plugin_data", None)
        if "loras_choices" in inputs:
            loras_choices = inputs.pop("loras_choices")
            inputs.pop("model_filename", None)
        else:
            loras_choices = inputs.get("activated_loras")
        if model_type is None:
            model_type = state["model_type"]

        lora_dir = self._lora_dir_for_model(model_type)
        resolved_loras = self.resolve_loras_selection(lora_dir, loras_choices)
        inputs["activated_loras"] = resolved_loras

        if target == "state":
            return inputs

        inputs.pop("lset_name", None)

        unsaved_params = [
            "image_start",
            "image_end",
            "image_refs",
            "video_guide",
            "image_guide",
            "video_source",
            "video_mask",
            "image_mask",
            "audio_guide",
            "audio_guide2",
            "audio_source",
        ]
        for key in unsaved_params:
            inputs.pop(key, None)

        inputs["type"] = self._get_model_record(self._get_model_name(model_type))
        inputs["settings_version"] = self._settings_version
        model_def = self._get_model_def(model_type)
        base_model_type = self._get_base_model_type(model_type)
        model_family = self._get_model_family(base_model_type)
        if model_type != base_model_type:
            inputs["base_model_type"] = base_model_type

        diffusion_forcing = base_model_type in ["sky_df_1.3B", "sky_df_14B"]
        vace = self._test_vace_module(base_model_type)
        t2v = self._test_class_t2v(base_model_type)
        ltxv = base_model_type in ["ltxv_13B"]

        if target == "settings":
            return inputs

        image_outputs = inputs.get("image_mode", 0) > 0
        keys_to_strip: List[str] = []

        if not model_def.get("audio_only", False):
            keys_to_strip += ["pace", "exaggeration", "temperature"]

        if not inputs.get("force_fps"):
            keys_to_strip.append("force_fps")

        if model_def.get("sample_solvers") is None:
            keys_to_strip.append("sample_solver")

        if self._any_audio_track(base_model_type) or self._server_config.get("mmaudio_enabled", 0) == 0:
            keys_to_strip += ["MMAudio_setting", "MMAudio_prompt", "MMAudio_neg_prompt"]

        video_prompt_type = inputs.get("video_prompt_type", "")
        if "G" not in video_prompt_type:
            keys_to_strip.append("denoising_strength")

        enhancer_enabled = self._server_config.get("enhancer_enabled", 0) > 0
        enhancer_mode = self._server_config.get("enhancer_mode", 0)
        if not (enhancer_enabled and enhancer_mode == 0):
            keys_to_strip.append("prompt_enhancer")

        if model_def.get("model_modes") is None:
            keys_to_strip.append("model_mode")

        guide_supported = model_def.get("guide_custom_choices") or model_def.get("guide_preprocessing")
        if not guide_supported:
            keys_to_strip += ["keep_frames_video_guide", "mask_expand"]

        if "I" not in video_prompt_type:
            keys_to_strip.append("remove_background_images_ref")
            if not model_def.get("any_image_refs_relative_size", False):
                keys_to_strip.append("image_refs_relative_size")

        if not vace:
            keys_to_strip += ["frames_positions", "control_net_weight", "control_net_weight2"]

        if not len(model_def.get("control_net_weight_alt_name", "")):
            keys_to_strip.append("control_net_weight_alt")

        if model_def.get("video_guide_outpainting") is None:
            keys_to_strip.append("video_guide_outpainting")

        if not (vace or t2v):
            keys_to_strip.append("min_frames_if_references")

        if not (diffusion_forcing or ltxv or vace):
            keys_to_strip.append("keep_frames_video_source")

        if not self._test_any_sliding_window(base_model_type):
            keys_to_strip += [
                "sliding_window_size",
                "sliding_window_overlap",
                "sliding_window_overlap_noise",
                "sliding_window_discard_last_frames",
                "sliding_window_color_correction_strength",
            ]

        if not model_def.get("audio_guidance", False):
            keys_to_strip += ["audio_guidance_scale", "speakers_locations"]

        if not model_def.get("embedded_guidance", False):
            keys_to_strip.append("embedded_guidance_scale")

        if not (model_def.get("tea_cache", False) or model_def.get("mag_cache", False)):
            keys_to_strip += ["skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc"]

        guidance_max_phases = model_def.get("guidance_max_phases", 0)
        guidance_phases = inputs.get("guidance_phases", 1)
        if guidance_max_phases < 1:
            keys_to_strip += ["guidance_scale", "guidance_phases"]
        if guidance_max_phases < 2 or guidance_phases < 2:
            keys_to_strip += ["guidance2_scale", "switch_threshold"]
        if guidance_max_phases < 3 or guidance_phases < 3:
            keys_to_strip += ["guidance3_scale", "switch_threshold2", "model_switch_phase"]

        if ltxv or image_outputs:
            keys_to_strip.append("flow_shift")

        if model_def.get("no_negative_prompt", False):
            keys_to_strip.append("negative_prompt")

        if not model_def.get("skip_layer_guidance", False):
            keys_to_strip += ["slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc"]

        if not model_def.get("cfg_zero", False):
            keys_to_strip.append("cfg_zero_step")

        if not model_def.get("cfg_star", False):
            keys_to_strip.append("cfg_star_switch")

        if not model_def.get("adaptive_projected_guidance", False):
            keys_to_strip.append("apg_switch")

        if model_family != "wan" or diffusion_forcing:
            keys_to_strip += ["NAG_scale", "NAG_tau", "NAG_alpha"]

        for key in keys_to_strip:
            inputs.pop(key, None)

        if target == "metadata":
            adapter_payloads: Dict[str, Any] = {}
            lora_payload = self.build_lora_payload(
                model_type,
                resolved_loras,
                multipliers=inputs.get("loras_multipliers"),
            )
            if lora_payload:
                adapter_payloads["lora"] = lora_payload
            prompt_payload = self.resolve_prompt_enhancer(inputs.get("prompt_enhancer"))
            if prompt_payload:
                adapter_payloads["prompt_enhancer"] = prompt_payload
            if adapter_payloads:
                inputs["adapter_payloads"] = adapter_payloads
            inputs = {k: v for k, v in inputs.items() if v is not None}

        return inputs

    def get_file_list(
        self,
        state: Dict[str, Any],
        input_file_list: Optional[Sequence[Any]],
        *,
        audio_files: bool = False,
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        gen = get_gen_info(state)
        file_list_key = "audio_file_list" if audio_files else "file_list"
        settings_key = "audio_file_settings_list" if audio_files else "file_settings_list"

        with self._lock:
            if file_list_key in gen:
                file_list = gen[file_list_key]
                file_settings = gen[settings_key]
            else:
                file_list = []
                file_settings = []
                if input_file_list is not None:
                    for file_path in input_file_list:
                        path = file_path[0] if isinstance(file_path, tuple) else file_path
                        loader = self._settings_loader or self.load_settings_from_file
                        configs, _, _ = loader(state, path, False, False, False)
                        file_list.append(path)
                        file_settings.append(configs)
                gen[file_list_key] = file_list
                gen[settings_key] = file_settings
        return file_list, file_settings

    @staticmethod
    def set_file_choice(
        gen: Dict[str, Any],
        file_list: Sequence[Any],
        choice: int,
        *,
        audio_files: bool = False,
    ) -> None:
        if file_list:
            choice = max(choice, 0)
        key_selected = "audio_selected" if audio_files else "selected"
        key_last = "audio_last_selected" if audio_files else "last_selected"
        gen[key_last] = (choice + 1) >= len(file_list)
        gen[key_selected] = choice

    def save_inputs(self, request: SaveInputsRequest) -> None:
        model_type = request.state["model_type"]
        payload_data = request.payload.to_dict()

        image_mask_guide = payload_data.get("image_mask_guide")
        image_mode = payload_data.get("image_mode", 0)
        video_prompt_type = payload_data.get("video_prompt_type")
        if (
            image_mask_guide is not None
            and image_mode >= 1
            and video_prompt_type is not None
            and "A" in video_prompt_type
            and "U" not in video_prompt_type
        ):
            background = image_mask_guide.get("background")
            layers = image_mask_guide.get("layers") or []
            if background is not None:
                payload_data["image_guide"] = background
            if layers:
                payload_data["image_mask"] = layers[0]
            payload_data["image_mask_guide"] = None

        inputs = dict(payload_data)
        inputs["state"] = request.state
        inputs["plugin_data"] = request.plugin_data or {}

        cleaned_inputs = self.prepare_inputs_dict(request.target, inputs, model_type=model_type)

        if request.target == "settings":
            defaults_filename = self._get_settings_file_name(model_type)
            with open(defaults_filename, "w", encoding="utf-8") as handle:
                json.dump(cleaned_inputs, handle, indent=4)
            self._notify_info("New Default Settings saved")
        elif request.target == "state":
            self._set_model_settings(request.state, model_type, cleaned_inputs)

    def _resolve_loras_url_cache(
        self,
        lora_dir: Optional[str],
        loras_selected: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if loras_selected is None:
            return None

        if self._loras_url_cache is None:
            self._loras_url_cache = self._load_loras_cache()

        cache = self._loras_url_cache
        updated = False
        resolved: List[str] = []
        base_dir = lora_dir or ""
        for lora in loras_selected:
            base_name = os.path.basename(lora)
            local_name = os.path.join(base_dir, base_name) if base_dir else base_name
            url = cache.get(local_name, base_name)
            if (lora.startswith("http:") or lora.startswith("https:")) and url != lora:
                cache[local_name] = lora
                url = lora
                updated = True
            resolved.append(url)
        if updated:
            self._write_loras_cache(cache)
        return resolved

    def resolve_loras_selection(
        self,
        lora_dir: Optional[str],
        loras_selected: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        return self._resolve_loras_url_cache(lora_dir, loras_selected)

    def get_loras_url_cache(self) -> Dict[str, str]:
        if self._loras_url_cache is None:
            self._loras_url_cache = self._load_loras_cache()
        return self._loras_url_cache

    def load_settings_from_file(
        self,
        state: Dict[str, Any],
        file_path: str,
        allow_json: bool,
        merge_with_defaults: bool,
        switch_type_if_compatible: bool,
        min_settings_version: float = 0,
        merge_loras: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
        configs: Optional[Dict[str, Any]] = None
        any_media = False
        any_audio = False

        lower_path = file_path.lower()
        if lower_path.endswith(".json") and allow_json:
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                configs = loaded if isinstance(loaded, dict) else None
            except Exception:
                configs = None
        elif has_video_file_extension(file_path):
            try:
                video_metadata = read_metadata_from_video(file_path)
                if isinstance(video_metadata, dict) and video_metadata:
                    configs = video_metadata
                    any_media = True
            except Exception:
                configs = None
        elif has_image_file_extension(file_path):
            try:
                image_metadata = read_image_metadata(file_path)
                if isinstance(image_metadata, dict) and image_metadata:
                    configs = image_metadata
                    any_media = True
            except Exception:
                configs = None
        elif has_audio_file_extension(file_path):
            try:
                audio_metadata = read_audio_metadata(file_path)
                if isinstance(audio_metadata, dict) and audio_metadata:
                    configs = audio_metadata
                    any_audio = True
            except Exception:
                configs = None

        if configs is None:
            return None, False, False

        if not merge_with_defaults:
            type_field = str(configs.get("type", ""))
            if "WanGP" not in type_field:
                return None, False, False

        current_model_type = state["model_type"]
        model_type = configs.get("model_type")
        base_model_type = self._get_base_model_type(model_type) if model_type else None

        if base_model_type is None:
            base_model_type = configs.get("base_model_type")
            if base_model_type is not None:
                model_type = base_model_type

        if model_type is None:
            model_filename = str(configs.get("model_filename", ""))
            derived_model_type = self._get_model_type(model_filename) if model_filename else None
            model_type = derived_model_type or current_model_type
        elif model_type not in self._model_types:
            model_type = current_model_type

        if switch_type_if_compatible and self._are_model_types_compatible(model_type, current_model_type):
            model_type = current_model_type

        old_loras_selected: Optional[List[str]] = None
        old_loras_multipliers: Optional[str] = None
        if merge_with_defaults:
            defaults = self._get_model_settings(state, model_type) or self._get_default_settings(model_type) or {}
            if merge_loras is not None and model_type == current_model_type:
                old_loras_selected = list(defaults.get("activated_loras", []))
                old_loras_multipliers = defaults.get("loras_multipliers", "")
            defaults.update(configs)
            configs = defaults

        lora_dir = self._lora_dir_for_model(model_type)
        loras_selected = configs.get("activated_loras") or []
        loras_multipliers = configs.get("loras_multipliers", "")
        if loras_selected:
            resolved = self.resolve_loras_selection(lora_dir, loras_selected)
            configs["activated_loras"] = resolved or []
        else:
            configs["activated_loras"] = []

        if old_loras_selected is not None:
            previous = self.resolve_loras_selection(lora_dir, old_loras_selected) or []
            merged_loras, merged_multipliers = merge_loras_settings(
                previous,
                old_loras_multipliers or "",
                configs["activated_loras"],
                loras_multipliers,
                merge_loras,
            )
            configs["activated_loras"] = merged_loras
            configs["loras_multipliers"] = merged_multipliers
        else:
            configs["loras_multipliers"] = loras_multipliers

        self._fix_settings(model_type, configs, min_settings_version)
        configs["model_type"] = model_type

        return configs, any_media, any_audio

    def _load_loras_cache(self) -> Dict[str, str]:
        if not self._loras_cache_path.exists():
            return {}
        try:
            with open(self._loras_cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            return {}
        except Exception:
            return {}

    def _write_loras_cache(self, cache: Dict[str, str]) -> None:
        try:
            with open(self._loras_cache_path, "w", encoding="utf-8") as handle:
                json.dump(cache, handle, indent=2)
        except Exception:
            pass
