from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from core.production_manager import CallbackBuilder, ProductionManager, SendCommand
from core.task_inputs import TaskInputManager
from cli.queue_state import (
    QueueStateTracker,
    request_abort,
    reset_generation_counters,
    update_queue_tracking,
)
from shared.notifications import GenerationNotifier
from shared.utils.process_locks import gen_lock
from shared.utils.thread_utils import AsyncStream, async_run
from cli.queue_utils import PreviewImages, clear_queue_action, get_preview_images


class QueueController:
    """
    Minimal headless queue controller that mirrors the legacy wgp loop.

    Tasks are dispatched through `AsyncStream` so generation events can be
    proxied back to the CLI logger while the worker thread invokes the
    production manager.
    """

    def __init__(
        self,
        *,
        manager: ProductionManager,
        state: Dict[str, Any],
        notifier: Optional[GenerationNotifier] = None,
        callback_builder: Optional[CallbackBuilder] = None,
        queue_tracker: Optional[QueueStateTracker] = None,
    ) -> None:
        self._manager = manager
        self._state = state
        self._default_notifier = notifier
        self._default_callback_builder = callback_builder
        self._task_seed = 1
        self._queue_tracker = queue_tracker or manager.queue_tracker
        self._paused = False
        self._task_inputs_manager = manager.task_inputs()

    def enqueue_task(
        self,
        params: Dict[str, Any],
        *,
        plugin_data: Optional[Dict[str, Any]] = None,
        task_stub: Optional[Dict[str, Any]] = None,
        task_seed: Optional[int] = None,
        attr_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gen_state = self._state.setdefault("gen", {})
        queue: List[Dict[str, Any]] = gen_state.setdefault("queue", [])
        active_seed = task_seed if task_seed is not None else self._next_task_seed()
        entry = self._build_queue_entry(
            params,
            plugin_data=plugin_data,
            task_stub=task_stub,
            task_id=active_seed,
            attr_overrides=attr_overrides,
        )
        queue.append(entry)
        self._update_queue_tracking(queue)
        return entry

    def _next_task_seed(self) -> int:
        seed = self._task_seed
        self._task_seed += 1
        return seed

    def _update_queue_tracking(self, queue: List[Dict[str, Any]]) -> None:
        gen_state = self._state.get("gen") if isinstance(self._state, dict) else None
        audio_tracks = None
        if isinstance(gen_state, dict):
            audio_tracks = gen_state.get("audio_tracks")
        update_queue_tracking(queue, self._queue_tracker, audio_tracks=audio_tracks)

    def queue_metrics(self) -> Dict[str, Any]:
        return self._queue_tracker.metrics()

    @property
    def task_inputs(self) -> TaskInputManager:
        manager = self._manager.task_inputs()
        if manager is not self._task_inputs_manager:
            self._task_inputs_manager = manager
        return self._task_inputs_manager

    def _build_queue_entry(
        self,
        params: Dict[str, Any],
        *,
        plugin_data: Optional[Dict[str, Any]],
        task_stub: Optional[Dict[str, Any]],
        task_id: int,
        attr_overrides: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata_inputs = dict(params)
        metadata = self.task_inputs.prepare_inputs_dict(
            "metadata",
            metadata_inputs,
            model_type=params.get("model_type"),
            model_filename=params.get("model_filename"),
        )
        resolved_loras = metadata.get("activated_loras")
        if resolved_loras is not None:
            params["activated_loras"] = resolved_loras
        if "loras_multipliers" in metadata:
            params["loras_multipliers"] = metadata["loras_multipliers"]
        adapter_payloads = metadata.get("adapter_payloads") if isinstance(metadata, dict) else None

        preview = get_preview_images(params)

        queue_entry: Dict[str, Any] = {
            "id": task_id,
            "params": params.copy(),
            "plugin_data": plugin_data.copy() if isinstance(plugin_data, dict) else (plugin_data or {}),
            "task_stub": task_stub,
            "metadata": metadata,
            "start_image_data": preview.start_data,
            "end_image_data": preview.end_data,
        }
        if adapter_payloads:
            queue_entry["adapter_payloads"] = adapter_payloads
        if attr_overrides:
            queue_entry["attr_overrides"] = dict(attr_overrides)
        queue_entry.update(self._build_queue_summary(params, metadata, preview))
        return queue_entry

    def _build_queue_summary(
        self,
        params: Dict[str, Any],
        metadata: Dict[str, Any],
        preview: PreviewImages,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        repeats = metadata.get("repeat_generation")
        if repeats is None:
            repeats = params.get("repeat_generation")
        summary["repeats"] = repeats if repeats not in (None, "") else 1

        length = metadata.get("video_length")
        if length is None:
            length = params.get("video_length")
        summary["length"] = length if length is not None else "-"

        steps = metadata.get("num_inference_steps")
        if steps is None:
            steps = params.get("num_inference_steps")
        summary["steps"] = steps if steps is not None else "-"

        summary["prompt"] = metadata.get("prompt") or params.get("prompt", "")
        summary.update(preview.build_payload())

        return summary

    def _interrupt_active_model(self) -> None:
        wan_model = getattr(self._manager.wgp, "wan_model", None)
        if wan_model is not None and hasattr(wan_model, "_interrupt"):
            wan_model._interrupt = True

    def request_abort(self) -> bool:
        """
        Signal the active generation to abort and propagate the interrupt flag to
        the underlying model, returning True when a generation was in progress.
        """

        gen_state = self._state.setdefault("gen", {})
        lock = getattr(self._manager.wgp, "lock", None)
        lock_ctx = lock if lock is not None else nullcontext()
        with lock_ctx:
            aborted = request_abort(gen_state)
            if aborted:
                self._interrupt_active_model()
        return aborted

    def clear_queue(self) -> Dict[str, bool]:
        """
        Abort the active task (when running) and clear any queued entries.

        Returns a dict describing whether work was aborted and/or pending tasks
        were removed, alongside fresh queue metrics.
        """

        gen_state = self._state.setdefault("gen", {})
        queue: List[Dict[str, Any]] = gen_state.setdefault("queue", [])
        lock = getattr(self._manager.wgp, "lock", None)
        result = clear_queue_action(
            self._state,
            queue=queue,
            lock=lock,
            tracker=self._queue_tracker,
            interrupt_callback=self._interrupt_active_model,
        )
        aborted_current = bool(result.get("aborted"))
        cleared_pending = bool(result.get("cleared"))
        metrics = {k: result[k] for k in ("queue_summary", "queue_length") if k in result}
        if aborted_current or cleared_pending:
            reset_generation_counters(gen_state, preserve_abort=aborted_current)
            self._paused = False
        return {
            "aborted": aborted_current,
            "cleared": cleared_pending,
            "metrics": metrics,
        }

    def is_paused(self) -> bool:
        return self._paused

    def request_pause(
        self,
        message: Optional[str] = None,
        *,
        timeout: float = 5.0,
    ) -> bool:
        """
        Attempt to pause the active generation. Returns True when the worker
        acknowledges the pause state (or when no work is active and the pause
        latch is set for future tasks).
        """

        gen_state = self._state.setdefault("gen", {})
        pause_message = message or "Generation paused by operator."

        with gen_lock:
            if self._paused:
                gen_state["pause_msg"] = pause_message
                return True
            if not gen_state.get("in_progress"):
                gen_state["pause_msg"] = pause_message
                gen_state["process_status"] = "process:pause"
                self._paused = True
                return True

        deadline = time.time() + max(timeout, 0.1)
        requested = False
        while time.time() < deadline:
            with gen_lock:
                if not gen_state.get("in_progress"):
                    gen_state["pause_msg"] = pause_message
                    gen_state["process_status"] = "process:pause"
                    self._paused = True
                    return True
                status = gen_state.get("process_status")
                if status == "process:pause":
                    gen_state["pause_msg"] = pause_message
                    self._paused = True
                    return True
                if not requested and status in (None, "process:main"):
                    gen_state["pause_msg"] = pause_message
                    gen_state["process_status"] = "request:pause"
                    requested = True
            time.sleep(0.05)

        with gen_lock:
            if gen_state.get("process_status") == "process:pause":
                gen_state["pause_msg"] = pause_message
                self._paused = True
                return True
        return False

    def resume(self) -> bool:
        """
        Resume generation after a pause request. Returns True when the pause
        latch transitions back to the active state.
        """

        gen_state = self._state.setdefault("gen", {})
        with gen_lock:
            status = gen_state.get("process_status")
            if status in {"process:pause", "request:pause"}:
                gen_state["process_status"] = "process:main"
                gen_state.pop("pause_msg", None)
                self._paused = False
                return True
            if not gen_state.get("in_progress") and self._paused:
                gen_state["process_status"] = None
                gen_state.pop("pause_msg", None)
                self._paused = False
                return True
        return False

    def _ensure_main_process(self) -> None:
        gen_state = self._state.setdefault("gen", {})
        while True:
            with gen_lock:
                process_status = gen_state.get("process_status")
                if process_status in (None, "process:main"):
                    gen_state["process_status"] = "process:main"
                    return
            time.sleep(0.1)

    def _execute_task(
        self,
        task: Dict[str, Any],
        *,
        send_cmd: SendCommand,
        output_dir_override: Optional[Any],
        image_output_dir_override: Optional[Any],
        attr_overrides: Optional[Dict[str, Any]],
        server_config_overrides: Optional[Dict[str, Any]],
        notifier: Optional[GenerationNotifier],
        callback_builder: Optional[CallbackBuilder],
    ) -> List[str]:
        stream = AsyncStream()
        internal_send = stream.output_queue.push
        resolved_notifier = notifier if notifier is not None else self._default_notifier
        resolved_callback_builder = callback_builder if callback_builder is not None else self._default_callback_builder
        params = task["params"]
        plugin_data = task.get("plugin_data")
        task_stub = task.get("task_stub")
        task_seed = task["id"]
        task_specific_overrides = task.get("attr_overrides")
        merged_attr_overrides = dict(attr_overrides or {})
        if isinstance(task_specific_overrides, dict):
            merged_attr_overrides.update(task_specific_overrides)
        resolved_attr_overrides = merged_attr_overrides or None
        metadata = task.get("metadata")
        adapter_payloads = task.get("adapter_payloads")
        if adapter_payloads is None and isinstance(metadata, dict):
            adapter_payloads = metadata.get("adapter_payloads")

        with gen_lock:
            gen_state = self._state.setdefault("gen", {})
            gen_state.pop("audio_tracks", None)

        def worker() -> None:
            try:
                outputs = self._manager.run_generation(
                    params,
                    state=self._state,
                    send_cmd=internal_send,
                    output_dir_override=output_dir_override,
                    image_output_dir_override=image_output_dir_override,
                    attr_overrides=resolved_attr_overrides,
                    server_config_overrides=server_config_overrides,
                    notifier=resolved_notifier,
                    callback_builder=resolved_callback_builder,
                    plugin_data=plugin_data,
                    task_stub=task_stub,
                    task_seed=task_seed,
                    adapter_payloads=adapter_payloads,
                )
                stream.output_queue.push("outputs", outputs)
            except Exception as exc:  # pragma: no cover - defensive relay
                stream.output_queue.push("error", exc)
            finally:
                stream.output_queue.push("exit", None)

        async_run(worker)

        captured_outputs: List[str] = []
        pending_error: Optional[BaseException] = None
        while True:
            cmd, payload = stream.output_queue.next()
            if cmd == "exit":
                break
            if cmd == "outputs":
                captured_outputs = list(payload or [])
                continue
            try:
                send_cmd(cmd, payload)
            except BaseException as exc:  # propagate after cleanup
                pending_error = exc
                break

        with gen_lock:
            gen_state = self._state.setdefault("gen", {})
            gen_state["process_status"] = None
            gen_state["in_progress"] = False

        if pending_error is not None:
            raise pending_error
        return captured_outputs

    def run_all(
        self,
        *,
        send_cmd: SendCommand,
        output_dir_override: Optional[Any] = None,
        image_output_dir_override: Optional[Any] = None,
        attr_overrides: Optional[Dict[str, Any]] = None,
        server_config_overrides: Optional[Dict[str, Any]] = None,
        notifier: Optional[GenerationNotifier] = None,
        callback_builder: Optional[CallbackBuilder] = None,
    ) -> List[str]:
        """
        Process every queued task sequentially. Returns the list of all output
        artefact paths recorded by the pipeline.
        """

        self._ensure_main_process()
        gen_state = self._state.setdefault("gen", {})
        queue: List[Dict[str, Any]] = gen_state.setdefault("queue", [])
        results: List[str] = []

        while queue:
            self._ensure_main_process()
            with gen_lock:
                gen_state["in_progress"] = True
            task = queue[0]
            should_break = False
            try:
                outputs = self._execute_task(
                    task,
                    send_cmd=send_cmd,
                    output_dir_override=output_dir_override,
                    image_output_dir_override=image_output_dir_override,
                    attr_overrides=attr_overrides,
                    server_config_overrides=server_config_overrides,
                    notifier=notifier,
                    callback_builder=callback_builder,
                )
                if outputs:
                    results.extend(outputs)
            except BaseException:
                queue.clear()
                self._update_queue_tracking(queue)
                raise
            finally:
                if queue:
                    queue.pop(0)
                    self._update_queue_tracking(queue)
            if gen_state.get("abort"):
                gen_state["abort"] = False
                should_break = True
            if should_break:
                break

        reset_generation_counters(gen_state)
        self._paused = False
        self._update_queue_tracking(queue)
        return results

    def run_single(
        self,
        params: Dict[str, Any],
        *,
        send_cmd: SendCommand,
        plugin_data: Optional[Dict[str, Any]] = None,
        output_dir_override: Optional[Any] = None,
        image_output_dir_override: Optional[Any] = None,
        attr_overrides: Optional[Dict[str, Any]] = None,
        server_config_overrides: Optional[Dict[str, Any]] = None,
        notifier: Optional[GenerationNotifier] = None,
        callback_builder: Optional[CallbackBuilder] = None,
        task_stub: Optional[Dict[str, Any]] = None,
        task_seed: Optional[int] = None,
    ) -> List[str]:
        task = self.enqueue_task(
            params,
            plugin_data=plugin_data,
            task_stub=task_stub,
            task_seed=task_seed,
            attr_overrides=attr_overrides,
        )
        try:
            return self.run_all(
                send_cmd=send_cmd,
                output_dir_override=output_dir_override,
                image_output_dir_override=image_output_dir_override,
                attr_overrides=attr_overrides,
                server_config_overrides=server_config_overrides,
                notifier=notifier,
                callback_builder=callback_builder,
            )
        finally:
            gen_state = self._state.setdefault("gen", {})
            queue: List[Dict[str, Any]] = gen_state.setdefault("queue", [])
            queue[:] = [item for item in queue if item["id"] != task["id"]]
            self._update_queue_tracking(queue)
