from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


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


@dataclass
class PromptEnhancerSpec:
    prompt_enhancer: Optional[str]
    pipe: MutableMapping[str, Any]
    kwargs: MutableMapping[str, Any]
    force: bool = False


@dataclass
class PromptEnhancerContext:
    prompts: Sequence[str]
    image_start: Optional[Sequence[Any]]
    image_refs: Optional[Sequence[Any]]
    is_image: bool
    audio_only: bool
    seed: int


class PromptEnhancerBridge:
    """
    Shim around ``wgp.setup_prompt_enhancer`` and related helpers.
    """

    def __init__(self, wgp_module: Any) -> None:
        self._wgp = wgp_module
        self._primed_configs: Dict[str, bool] = {}

    def _server_config_hash(self) -> str:
        server_config = getattr(self._wgp, "server_config", {}) or {}
        if not isinstance(server_config, Mapping):
            return "static"
        return _hash_server_config(server_config)

    def reset(self) -> None:
        """Release enhancer resources and clear caches."""

        reset_fn = getattr(self._wgp, "reset_prompt_enhancer", None)
        if callable(reset_fn):
            reset_fn()
        self._primed_configs.clear()

    def prime(self, spec: PromptEnhancerSpec) -> None:
        """
        Ensure the enhancer runtime is initialised for the current server config.
        """

        if not spec.prompt_enhancer:
            return
        key = self._server_config_hash()
        if not spec.force and self._primed_configs.get(key):
            return
        setup_fn = getattr(self._wgp, "setup_prompt_enhancer", None)
        if setup_fn is None:
            raise RuntimeError("wgp module missing setup_prompt_enhancer; cannot prime prompt enhancer.")
        setup_fn(spec.pipe, spec.kwargs)
        self._primed_configs[key] = True

    def enhance(
        self,
        prompt_enhancer: Optional[str],
        context: PromptEnhancerContext,
    ) -> Optional[Sequence[str]]:
        """
        Run the prompt enhancer for the provided context.
        """

        if not prompt_enhancer:
            return None
        processor = getattr(self._wgp, "process_prompt_enhancer", None)
        if processor is None:
            raise RuntimeError("wgp module missing process_prompt_enhancer; cannot enhance prompts.")
        image_start = list(context.image_start) if context.image_start is not None else None
        image_refs = list(context.image_refs) if context.image_refs is not None else None
        return processor(
            prompt_enhancer,
            list(context.prompts),
            image_start,
            image_refs,
            context.is_image,
            context.audio_only,
            context.seed,
        )

    def snapshot_state(self) -> Dict[str, Any]:
        """Expose cache metadata for debugging."""

        return {"primed_configs": list(self._primed_configs.keys())}
