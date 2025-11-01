from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def _normalise_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise_config(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _hash_server_config(server_config: Mapping[str, Any]) -> str:
    normalised = _normalise_config(server_config)
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LoRALibrary:
    model_type: str
    server_config_hash: str
    lora_dir: Optional[Path]
    loras: Tuple[str, ...]
    presets: Tuple[str, ...]


@dataclass(frozen=True)
class LoRAHydrationResult:
    library: LoRALibrary
    default_choices: Tuple[str, ...]
    default_multipliers: str
    default_prompt: Optional[str]
    default_preset: Optional[str]


@dataclass(frozen=True)
class LoRAPresetResolution:
    names: Tuple[str, ...]
    multipliers: str
    prompt: Optional[str]
    prompt_is_full: bool


class LoRAInjectionManager:
    """
    Shim adapter that wraps ``wgp.setup_loras`` while caching discovery results.
    """

    def __init__(self, wgp_module: Any) -> None:
        self._wgp = wgp_module
        self._hydration_cache: Dict[Tuple[str, str, str], LoRAHydrationResult] = {}

    def reset(self) -> None:
        """Clear cached discovery results."""

        self._hydration_cache.clear()

    def _server_config_hash(self) -> str:
        server_config = getattr(self._wgp, "server_config", {}) or {}
        if not isinstance(server_config, Mapping):
            return "static"
        return _hash_server_config(server_config)

    def _cache_key(self, model_type: str, preselected: Optional[str]) -> Tuple[str, str, str]:
        preset_key = preselected or ""
        return model_type, self._server_config_hash(), preset_key

    def _resolve_lora_dir(self, model_type: str) -> Optional[Path]:
        get_lora_dir = getattr(self._wgp, "get_lora_dir", None)
        if callable(get_lora_dir):
            lora_dir = get_lora_dir(model_type)
            if isinstance(lora_dir, str):
                return Path(lora_dir)
        return None

    def hydrate(
        self,
        model_type: str,
        *,
        transformer: Any = None,
        preselected_preset: Optional[str] = None,
        refresh: bool = False,
    ) -> LoRAHydrationResult:
        """
        Discover available LoRA weights and presets for ``model_type``.

        When ``refresh`` is true the cache entry is rebuilt even if the server
        configuration hash is unchanged.
        """

        effective_preset = (
            preselected_preset if preselected_preset is not None else getattr(self._wgp, "lora_preselected_preset", "")
        )
        key = self._cache_key(model_type, effective_preset)
        if refresh or key not in self._hydration_cache:
            lora_dir = self._resolve_lora_dir(model_type)
            lora_dir_str = str(lora_dir) if lora_dir is not None else ""
            loras, presets, default_choices, default_multipliers, default_prompt, default_preset = (
                self._wgp.setup_loras(
                    model_type,
                    transformer,
                    lora_dir_str,
                    effective_preset or "",
                    None,
                )
            )
            library = LoRALibrary(
                model_type=model_type,
                server_config_hash=key[1],
                lora_dir=lora_dir,
                loras=tuple(loras),
                presets=tuple(presets),
            )
            result = LoRAHydrationResult(
                library=library,
                default_choices=tuple(default_choices),
                default_multipliers=str(default_multipliers or ""),
                default_prompt=default_prompt or None,
                default_preset=default_preset or None,
            )
            self._hydration_cache[key] = result
        return self._hydration_cache[key]

    def presets(self, model_type: str) -> Tuple[str, ...]:
        """Return cached preset names for ``model_type``."""

        hydration = self.hydrate(model_type)
        return hydration.library.presets

    def resolve_preset(
        self,
        model_type: str,
        preset_name: str,
        *,
        available_loras: Optional[Sequence[str]] = None,
    ) -> LoRAPresetResolution:
        """
        Resolve ``preset_name`` to a list of LoRA file names and multipliers.
        """

        library = self.hydrate(model_type)
        lora_dir = library.library.lora_dir
        available = tuple(available_loras) if available_loras is not None else library.library.loras

        if preset_name.endswith(".lset"):
            extractor = getattr(self._wgp, "extract_preset", None)
            if extractor is None:
                raise RuntimeError("wgp module missing extract_preset; cannot resolve '.lset' preset.")
            loras_choices, loras_mult, preset_prompt, full_prompt, error = extractor(
                model_type,
                preset_name,
                list(available),
            )
            if error:
                raise RuntimeError(f"LoRA preset '{preset_name}' failed to load: {error}")
            return LoRAPresetResolution(
                names=tuple(Path(choice).name for choice in loras_choices),
                multipliers=str(loras_mult or ""),
                prompt=preset_prompt or None,
                prompt_is_full=bool(full_prompt),
            )

        if lora_dir is None:
            raise RuntimeError("LoRA presets require a configured LoRA directory; none found.")
        preset_path = lora_dir / preset_name
        if not preset_path.exists():
            raise RuntimeError(f"LoRA preset '{preset_name}' not found in {lora_dir}.")
        with preset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        raw_loras = data.get("activated_loras", [])
        if raw_loras is None:
            raw_loras = []
        if not isinstance(raw_loras, list):
            raise RuntimeError(f"LoRA preset '{preset_name}' has invalid 'activated_loras' format.")
        names = [Path(str(item)).name for item in raw_loras if isinstance(item, str)]
        multipliers = str(data.get("loras_multipliers", "") or "")
        preset_prompt = data.get("prompt")
        prompt_value = preset_prompt if isinstance(preset_prompt, str) and preset_prompt.strip() else None
        prompt_is_full = bool(data.get("full_prompt", False))
        return LoRAPresetResolution(
            names=tuple(names),
            multipliers=multipliers,
            prompt=prompt_value,
            prompt_is_full=prompt_is_full,
        )

    def snapshot_state(self) -> Dict[str, Any]:
        """Expose cache metadata for debugging."""

        return {
            "entries": [
                {
                    "model_type": result.library.model_type,
                    "server_config_hash": result.library.server_config_hash,
                    "preset_key": key[2],
                    "loras_count": len(result.library.loras),
                    "presets_count": len(result.library.presets),
                }
                for key, result in self._hydration_cache.items()
            ]
        }
