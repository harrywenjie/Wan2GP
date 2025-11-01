from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.lora.manager import LoRAInjectionManager
from core.prompt_enhancer.bridge import PromptEnhancerBridge, PromptEnhancerSpec
from core.task_inputs import TaskInputManager
from shared.notifications import GenerationNotifier
from core.io.media import (
    MediaPersistenceContext,
    MetadataSaveConfig,
    build_media_context,
    clone_metadata_config,
    default_metadata_config_templates,
)

if TYPE_CHECKING:
    from cli.queue_state import QueueStateTracker
else:
    QueueStateTracker = Any  # type: ignore[misc]

SendCommand = Callable[[str, Any], None]
CallbackBuilder = Callable[[Dict[str, Any], SendCommand, str, int], Optional[Callable[..., Any]]]

_SENTINEL = object()


@dataclass(frozen=True)
class MetadataState:
    """Snapshot of metadata persistence configuration for a generation run."""

    choice: str
    configs: Dict[str, MetadataSaveConfig]


@dataclass
class GenerationRuntime:
    """
    Wrap the legacy `wgp` runtime so modern callers can run headless generations
    without mutating module-level globals.
    """

    wgp: Any
    state: Dict[str, Any]
    output_dir_override: Optional[Path] = None
    image_output_dir_override: Optional[Path] = None
    attr_overrides: Dict[str, Any] = field(default_factory=dict)
    server_config_overrides: Dict[str, Any] = field(default_factory=dict)
    task_seed: int = 1
    task_stub: Optional[Dict[str, Any]] = None
    plugin_data: Dict[str, Any] = field(default_factory=dict)
    notifier: Optional[GenerationNotifier] = None
    callback_builder: Optional[CallbackBuilder] = None
    metadata_state: Optional[MetadataState] = None
    media_context: Optional[MediaPersistenceContext] = None
    adapter_payloads: Optional[Dict[str, Any]] = None
    lora_manager: Optional[LoRAInjectionManager] = None
    prompt_enhancer: Optional[PromptEnhancerBridge] = None

    def build_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.task_stub is not None:
            return self.task_stub
        prompt = params.get("prompt", "")
        return {
            "id": self.task_seed,
            "prompt": prompt,
            "params": params.copy(),
        }

    def _merged_attr_overrides(self) -> Dict[str, Any]:
        overrides = dict(self.attr_overrides)
        if self.output_dir_override is not None:
            override_path = str(self.output_dir_override)
            overrides.setdefault("save_path", override_path)
        image_override = self.image_output_dir_override or self.output_dir_override
        if image_override is not None:
            overrides.setdefault("image_save_path", str(image_override))
        return overrides

    def _merged_server_overrides(self) -> Dict[str, Any]:
        overrides = dict(self.server_config_overrides)
        if self.output_dir_override is not None:
            override_path = str(self.output_dir_override)
            overrides.setdefault("save_path", override_path)
        image_override = self.image_output_dir_override or self.output_dir_override
        if image_override is not None:
            overrides.setdefault("image_save_path", str(image_override))
        return overrides

    @contextmanager
    def apply_runtime_overrides(self):
        attr_overrides = self._merged_attr_overrides()
        server_overrides = self._merged_server_overrides()

        wgp = self.wgp
        attr_restore: Dict[str, Any] = {}
        config_restore: Dict[str, Any] = {}
        try:
            for attr, value in attr_overrides.items():
                attr_restore[attr] = getattr(wgp, attr, _SENTINEL)
                setattr(wgp, attr, value)
            if server_overrides:
                server_config = getattr(wgp, "server_config", None)
                if server_config is None:
                    raise RuntimeError("wgp runtime missing server_config for override application.")
                for key, value in server_overrides.items():
                    config_restore[key] = server_config.get(key, _SENTINEL)
                    server_config[key] = value
            yield
        finally:
            for attr, previous in attr_restore.items():
                if previous is _SENTINEL:
                    try:
                        delattr(wgp, attr)
                    except AttributeError:
                        pass
                else:
                    setattr(wgp, attr, previous)
            if server_overrides:
                server_config = getattr(wgp, "server_config", None)
                if server_config is not None:
                    for key, previous in config_restore.items():
                        if previous is _SENTINEL:
                            server_config.pop(key, None)
                        else:
                            server_config[key] = previous

    def run(self, params: Dict[str, Any], send_cmd: SendCommand) -> List[str]:
        task = self.build_task(params)
        self._apply_adapter_payloads(params)
        with self.apply_runtime_overrides():
            self.wgp.generate_video(
                task,
                send_cmd,
                plugin_data=self.plugin_data,
                notifier=self.notifier,
                callback_builder=self.callback_builder,
                metadata_state=self.metadata_state,
                media_context=self.media_context,
                adapter_payloads=self.adapter_payloads,
                **params,
            )
        gen_state = self.state.get("gen", {}) or {}
        outputs = gen_state.get("file_list", []) or []
        return list(outputs)

    def _apply_adapter_payloads(self, params: Dict[str, Any]) -> None:
        payloads = self.adapter_payloads if isinstance(self.adapter_payloads, dict) else {}
        model_type = params.get("model_type")
        lora_payload = payloads.get("lora")
        prompt_payload = payloads.get("prompt_enhancer")

        if lora_payload and isinstance(lora_payload, dict) and isinstance(model_type, str):
            manager = self.lora_manager
            if manager is not None:
                try:
                    manager.hydrate(model_type)
                except Exception:  # pragma: no cover - defensive
                    pass
            activated = list(lora_payload.get("activated", []))
            multipliers = lora_payload.get("multipliers")

            if activated and not params.get("activated_loras"):
                params["activated_loras"] = activated
            if multipliers is not None and not params.get("loras_multipliers"):
                params["loras_multipliers"] = multipliers

        bridge = self.prompt_enhancer
        prompt_selection = params.get("prompt_enhancer")
        force_prime = False
        if prompt_payload and isinstance(prompt_payload, dict):
            provider = prompt_payload.get("provider")
            enhancer_mode = prompt_payload.get("enhancer_mode")
            force_prime = bool(prompt_payload.get("force", False))
            server_config = getattr(self.wgp, "server_config", None)
            if isinstance(server_config, dict):
                if provider is not None:
                    server_config["enhancer_enabled"] = provider
                if enhancer_mode is not None:
                    server_config["enhancer_mode"] = enhancer_mode
        if bridge is not None:
            if prompt_selection:
                get_handles = getattr(self.wgp, "get_prompt_enhancer_runtime_handles", None)
                pipe = kwargs = None
                if callable(get_handles):
                    try:
                        pipe, kwargs = get_handles()
                    except Exception:  # pragma: no cover - defensive
                        pipe = kwargs = None
                if pipe is not None and kwargs is not None:
                    spec = PromptEnhancerSpec(
                        prompt_enhancer=prompt_selection,
                        pipe=pipe,
                        kwargs=kwargs,
                        force=force_prime,
                    )
                    try:
                        bridge.prime(spec)
                    except Exception:  # pragma: no cover - defensive
                        pass
            else:
                try:
                    bridge.reset()
                except Exception:  # pragma: no cover - defensive
                    pass


class ProductionManager:
    """
    Transitional orchestration entrypoint that will replace `wgp.py`.

    The class currently defers to the legacy module while exposing a stable API
    for CLI and future queue controllers. Once the remaining helpers migrate,
    the direct `wgp` dependency can be removed and this will become the canonical
    generation manager.
    """

    def __init__(
        self,
        *,
        wgp_module: Any,
        default_notifier: Optional[GenerationNotifier] = None,
        default_callback_builder: Optional[CallbackBuilder] = None,
        queue_tracker: Optional[QueueStateTracker] = None,
        task_input_manager: Optional[TaskInputManager] = None,
        queue_state_module: Optional[Any] = None,
        lora_manager: Optional[LoRAInjectionManager] = None,
        prompt_enhancer: Optional[PromptEnhancerBridge] = None,
    ):
        self._wgp = wgp_module
        self._default_notifier = default_notifier
        self._default_callback_builder = default_callback_builder
        if queue_state_module is None:
            from cli import queue_state as queue_state_module  # type: ignore
        self._queue_state = queue_state_module
        if queue_tracker is None:
            queue_tracker = queue_state_module.get_default_tracker()
        self._queue_tracker: QueueStateTracker = queue_tracker
        self._update_queue_tracking = queue_state_module.update_queue_tracking
        self._task_inputs_manager = task_input_manager
        self._metadata_config_templates: Optional[Dict[str, MetadataSaveConfig]] = None
        self._lora_manager: Optional[LoRAInjectionManager] = lora_manager
        self._prompt_enhancer_bridge: Optional[PromptEnhancerBridge] = prompt_enhancer

    @property
    def wgp(self) -> Any:
        return self._wgp

    @property
    def queue_tracker(self) -> QueueStateTracker:
        if self._queue_tracker is None:  # type: ignore[unreachable]
            self._queue_tracker = self._queue_state.get_default_tracker()
        return self._queue_tracker

    def lora_manager(self) -> LoRAInjectionManager:
        manager = self._lora_manager
        if manager is None:
            manager = LoRAInjectionManager(self._wgp)
            self._lora_manager = manager
        return manager

    def prompt_enhancer(self) -> PromptEnhancerBridge:
        bridge = self._prompt_enhancer_bridge
        if bridge is None:
            bridge = PromptEnhancerBridge(self._wgp)
            self._prompt_enhancer_bridge = bridge
        return bridge

    def task_inputs(self) -> TaskInputManager:
        manager = self._task_inputs_manager
        wgp = self._wgp
        server_config = getattr(wgp, "server_config", None)
        if server_config is None:
            raise RuntimeError("wgp module missing server_config; cannot build TaskInputManager.")
        if manager is None or manager.server_config is not server_config:
            manager = TaskInputManager(
                server_config=server_config,
                settings_version=getattr(wgp, "settings_version", 0),
                get_model_record=wgp.get_model_record,
                get_model_name=wgp.get_model_name,
                get_model_def=wgp.get_model_def,
                get_base_model_type=wgp.get_base_model_type,
                get_model_family=wgp.get_model_family,
                test_vace_module=wgp.test_vace_module,
                test_class_t2v=wgp.test_class_t2v,
                test_any_sliding_window=wgp.test_any_sliding_window,
                any_audio_track=wgp.any_audio_track,
                get_lora_dir=wgp.get_lora_dir,
                settings_loader=None,
                get_settings_file_name=wgp.get_settings_file_name,
                set_model_settings=wgp.set_model_settings,
                notify_info=wgp.notify_info,
                lock=wgp.lock,
                get_model_type=wgp.get_model_type,
                are_model_types_compatible=wgp.are_model_types_compatible,
                get_default_settings=wgp.get_default_settings,
                get_model_settings=wgp.get_model_settings,
                fix_settings=wgp.fix_settings,
                model_types=tuple(getattr(wgp, "model_types", ())),
                lora_manager=self.lora_manager(),
                prompt_enhancer=self.prompt_enhancer(),
            )
            self._task_inputs_manager = manager
        if hasattr(self._wgp, "_task_inputs_manager"):
            setattr(self._wgp, "_task_inputs_manager", manager)
        return manager

    def _resolve_notifier(self, override: Optional[GenerationNotifier]) -> Optional[GenerationNotifier]:
        if override is not None:
            return override
        if self._default_notifier is not None:
            return self._default_notifier
        raise RuntimeError(
            "ProductionManager requires a GenerationNotifier instance. "
            "Provide one via the constructor or run_generation()."
        )

    def _resolve_callback_builder(self, override: Optional[CallbackBuilder]) -> Optional[CallbackBuilder]:
        if override is not None:
            return override
        return self._default_callback_builder

    def _wait_for_preload(self) -> None:
        preload_policy = getattr(self._wgp, "preload_model_policy", [])
        if "P" in preload_policy and "U" not in preload_policy:
            while getattr(self._wgp, "wan_model", None) is None:
                time.sleep(1)

    def _ensure_model_ready(
        self,
        params: Dict[str, Any],
        send_cmd: SendCommand,
        *,
        adapter_payloads: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_type = params.get("model_type")
        if not model_type:
            return
        override_profile = params.get("override_profile", -1)

        self._wait_for_preload()

        wgp = self._wgp
        transformer_type = getattr(wgp, "transformer_type", None)
        reload_needed = getattr(wgp, "reload_needed", True)
        loaded_profile = getattr(wgp, "loaded_profile", -1)
        default_profile = getattr(wgp, "default_profile", -1)

        needs_reload = (
            model_type != transformer_type
            or reload_needed
            or (override_profile > 0 and override_profile != loaded_profile)
            or (override_profile < 0 and default_profile != loaded_profile)
        )
        if not needs_reload:
            return

        prompt_payload = None
        if isinstance(adapter_payloads, dict):
            candidate = adapter_payloads.get("prompt_enhancer")
            if isinstance(candidate, dict):
                prompt_payload = candidate
        prompt_selection = params.get("prompt_enhancer")

        server_config = getattr(wgp, "server_config", None)
        if isinstance(server_config, dict) and prompt_payload:
            provider = prompt_payload.get("provider")
            enhancer_mode = prompt_payload.get("enhancer_mode")
            if provider is not None:
                server_config["enhancer_enabled"] = provider
            if enhancer_mode is not None:
                server_config["enhancer_mode"] = enhancer_mode

        bridge: Optional[PromptEnhancerBridge]
        bridge = None
        if prompt_selection or prompt_payload:
            try:
                bridge = self.prompt_enhancer()
            except Exception:  # pragma: no cover - defensive
                bridge = None

        force_prime = bool(prompt_payload.get("force", False)) if prompt_payload else False
        if bridge is not None:
            if prompt_selection:
                def _primer(pipe, kwargs, *, _bridge=bridge, _selection=prompt_selection, _force=force_prime):
                    spec = PromptEnhancerSpec(
                        prompt_enhancer=_selection,
                        pipe=pipe,
                        kwargs=kwargs,
                        force=_force,
                    )
                    _bridge.prime(spec)

                setattr(wgp, "prompt_enhancer_primer", _primer)
            else:
                setattr(wgp, "prompt_enhancer_primer", None)
                try:
                    bridge.reset()
                except Exception:  # pragma: no cover - defensive
                    pass
        else:
            setattr(wgp, "prompt_enhancer_primer", None)

        setattr(wgp, "wan_model", None)
        release_model = getattr(wgp, "release_model", None)
        if callable(release_model):
            release_model()
        if send_cmd is not None:
            send_cmd("status", f"Loading model {wgp.get_model_name(model_type)}...")
        try:
            wan_model, offloadobj = wgp.load_models(model_type, override_profile)
        finally:
            if hasattr(wgp, "prompt_enhancer_primer"):
                setattr(wgp, "prompt_enhancer_primer", None)
        setattr(wgp, "wan_model", wan_model)
        setattr(wgp, "offloadobj", offloadobj)
        if send_cmd is not None:
            send_cmd("status", "Model loaded")
        setattr(wgp, "reload_needed", False)

    def _ensure_gen_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        gen = state.get("gen")
        if gen is None:
            gen = {}
            state["gen"] = gen
        gen.setdefault("queue", [])
        gen.setdefault("file_list", [])
        gen.setdefault("file_settings_list", [])
        gen.setdefault("audio_file_list", [])
        gen.setdefault("audio_file_settings_list", [])
        gen.setdefault("status", "")
        gen.setdefault("status_display", False)
        gen.setdefault("progress_status", "")
        gen.setdefault("progress_args", None)
        gen.setdefault("abort", False)
        gen.setdefault("in_progress", False)
        return gen

    def _metadata_choice(self) -> str:
        server_config = getattr(self._wgp, "server_config", {}) or {}
        choice = server_config.get("metadata_type", "metadata")
        return str(choice)

    def _metadata_config_templates(self) -> Dict[str, MetadataSaveConfig]:
        if self._metadata_config_templates is not None:
            return self._metadata_config_templates

        server_config = getattr(self._wgp, "server_config", None)
        templates = default_metadata_config_templates() if server_config is not None else {}
        self._metadata_config_templates = templates
        return templates

    def _clone_metadata_configs(self) -> Dict[str, MetadataSaveConfig]:
        templates = self._metadata_config_templates()
        return {
            key: clone_metadata_config(template, fallback_hint=key)
            for key, template in templates.items()
        }

    def media_context(self) -> MediaPersistenceContext:
        """
        Build a media persistence context for the current ``server_config``.
        """

        server_config = getattr(self._wgp, "server_config", None)
        if server_config is None:
            raise RuntimeError("wgp module missing server_config; cannot build MediaPersistenceContext.")
        return build_media_context(server_config)

    def metadata_state(
        self,
        *,
        choice_override: Optional[str] = None,
        configs_override: Optional[Dict[str, MetadataSaveConfig]] = None,
    ) -> MetadataState:
        """
        Build a metadata persistence snapshot for the upcoming generation run.

        The caller may supply overrides for the metadata mode or config map.
        """

        if configs_override is not None:
            configs = dict(configs_override)
        else:
            configs = self._clone_metadata_configs()
        choice = choice_override if choice_override is not None else self._metadata_choice()
        return MetadataState(choice=str(choice), configs=configs)

    def metadata_config_templates(self) -> Dict[str, MetadataSaveConfig]:
        """
        Return cloned metadata configuration templates for downstream consumers.

        Callers receive new copies so they can mutate handler bindings or options
        without affecting the manager's cached templates.
        """

        return self.metadata_state().configs.copy()

    def _build_adapter_payloads(
        self,
        params: Dict[str, Any],
        provided_payloads: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if provided_payloads is not None:
            return provided_payloads

        manager = self.task_inputs()
        payloads: Dict[str, Any] = {}
        model_type = params.get("model_type")
        activated = params.get("activated_loras")
        multipliers = params.get("loras_multipliers")

        if isinstance(model_type, str):
            lora_payload = manager.build_lora_payload(
                model_type,
                activated,
                multipliers=multipliers,
            )
            if lora_payload:
                payloads["lora"] = lora_payload

        prompt_payload = manager.resolve_prompt_enhancer(params.get("prompt_enhancer"))
        if prompt_payload:
            payloads["prompt_enhancer"] = prompt_payload

        return payloads or None

    def run_generation(
        self,
        params: Dict[str, Any],
        state: Dict[str, Any],
        send_cmd: SendCommand,
        *,
        output_dir_override: Optional[Path] = None,
        image_output_dir_override: Optional[Path] = None,
        attr_overrides: Optional[Dict[str, Any]] = None,
        server_config_overrides: Optional[Dict[str, Any]] = None,
        notifier: Optional[GenerationNotifier] = None,
        callback_builder: Optional[CallbackBuilder] = None,
        plugin_data: Optional[Dict[str, Any]] = None,
        task_stub: Optional[Dict[str, Any]] = None,
        task_seed: int = 1,
        adapter_payloads: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        resolved_notifier = self._resolve_notifier(notifier)
        resolved_callback_builder = self._resolve_callback_builder(callback_builder)

        self._ensure_gen_state(state)
        gen_state = state["gen"]
        queue = gen_state.get("queue", [])
        self._update_queue_tracking(queue, self._queue_tracker)
        if resolved_notifier is not None:
            resolved_notifier.reset_progress(state)

        merged_attr_overrides = dict(attr_overrides or {})
        metadata_choice_override = merged_attr_overrides.pop("metadata_choice", None)
        metadata_configs_override = merged_attr_overrides.pop("metadata_configs", None)
        metadata_state = self.metadata_state(
            choice_override=metadata_choice_override,
            configs_override=metadata_configs_override,
        )
        media_context = self.media_context()
        resolved_adapter_payloads = self._build_adapter_payloads(params, adapter_payloads)
        self._ensure_model_ready(params, send_cmd, adapter_payloads=resolved_adapter_payloads)
        lora_manager = self.lora_manager()
        prompt_enhancer_bridge = self.prompt_enhancer()

        runtime = GenerationRuntime(
            wgp=self._wgp,
            state=state,
            output_dir_override=output_dir_override,
            image_output_dir_override=image_output_dir_override,
            attr_overrides=merged_attr_overrides,
            server_config_overrides=server_config_overrides or {},
            task_seed=task_seed,
            task_stub=task_stub,
            plugin_data=plugin_data or {},
            notifier=resolved_notifier,
            callback_builder=resolved_callback_builder,
            metadata_state=metadata_state,
            media_context=media_context,
            adapter_payloads=resolved_adapter_payloads,
            lora_manager=lora_manager,
            prompt_enhancer=prompt_enhancer_bridge,
        )
        try:
            return runtime.run(params, send_cmd)
        finally:
            if resolved_notifier is not None:
                resolved_notifier.reset_progress(state)
