from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from typing import Any, Callable, Dict, List, Mapping, Optional

from core.production_manager import CallbackBuilder, SendCommand
from shared.notifications import GenerationNotifier
from shared.utils.process_locks import gen_lock
from core.progress import format_duration, merge_status_context
from cli.queue_utils import update_task_thumbnails

import time


def _ensure_gen_state(state: Dict[str, Any]) -> Dict[str, Any]:
    gen_state = state.get("gen")
    if gen_state is None:
        gen_state = {}
        state["gen"] = gen_state
    return gen_state


@dataclass
class CLIGenerationNotifier(GenerationNotifier):
    """Log-only notifier for CLI flows."""

    logger: Logger
    state: Dict[str, Any]
    queue_updater: Callable[[Dict[str, Any], Mapping[str, Any]], None] = update_task_thumbnails

    def reset_progress(self, state: Dict[str, Any]) -> None:
        gen = _ensure_gen_state(state)
        gen["extra_windows"] = 0
        gen["total_windows"] = 1
        gen["window_no"] = 1
        gen["extra_orders"] = 0
        gen["repeat_no"] = 0
        gen["total_generation"] = 0
        gen["in_progress"] = False
        gen["progress_args"] = None
        gen["progress_status"] = ""
        self.logger.debug("CLI notifier reset progress counters.")

    def refresh_preview(self, task: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        task_id = None
        if isinstance(task, dict):
            task_id = task.get("id")

        queue_entry: Optional[Dict[str, Any]] = None
        if task_id is not None:
            gen_state = _ensure_gen_state(self.state)
            queue = gen_state.get("queue", [])
            for entry in queue:
                if entry.get("id") == task_id:
                    queue_entry = entry
                    break

        target = queue_entry if queue_entry is not None else task
        if not isinstance(target, dict):
            self.logger.debug(
                "Preview refresh ignored for task %s; no queue entry available.",
                "<unknown>" if task_id is None else task_id,
            )
            return

        try:
            self.queue_updater(target, inputs)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning(
                "Preview refresh failed for task %s: %s",
                "<unknown>" if task_id is None else task_id,
                exc,
            )
            return

        if queue_entry is not None and target is queue_entry and isinstance(task, dict) and task is not queue_entry:
            for key in (
                "start_image_labels",
                "end_image_labels",
                "start_image_data_base64",
                "end_image_data_base64",
            ):
                if key in queue_entry:
                    task[key] = queue_entry[key]

        input_keys = sorted(inputs.keys()) if inputs else []
        self.logger.debug(
            "Preview refresh applied for task %s with inputs %s.",
            "<unknown>" if task_id is None else task_id,
            input_keys,
        )

    def notify_video_ready(self, video_path: Optional[str] = None) -> None:
        if video_path:
            self.logger.info("Video ready: %s", video_path)
        else:
            self.logger.info("Video ready event received.")

    def notify_generation_complete(self) -> None:
        self.logger.info("All generation tasks completed.")


def build_progress_callback(
    state: Dict[str, Any],
    send_cmd: SendCommand,
    *,
    num_inference_steps: int,
    logger: Optional[Logger] = None,
    pause_cleanup: Optional[Callable[[], None]] = None,
) -> Callable[..., None]:
    gen = _ensure_gen_state(state)
    gen["num_inference_steps"] = num_inference_steps
    start_time = time.time()

    def callback(
        step_idx: int = -1,
        latent: Any = None,
        force_refresh: bool = True,
        read_state: bool = False,
        override_num_inference_steps: int = -1,
        pass_no: int = -1,
        denoising_extra: str = "",
    ) -> None:
        in_pause = False
        pause_msg = None
        with gen_lock:
            process_status = gen.get("process_status")
            if isinstance(process_status, str) and process_status.startswith("request:"):
                gen["process_status"] = "process:" + process_status[len("request:") :]
                if pause_cleanup is not None:
                    try:
                        pause_cleanup()
                    except Exception as exc:  # pragma: no cover - defensive logging
                        if logger is not None:
                            logger.warning("Pause cleanup failed: %s", exc)
                pause_msg = gen.get("pause_msg", "Unknown Pause")
                in_pause = True

        if in_pause:
            send_cmd("progress", [0, pause_msg])
            while True:
                time.sleep(0.1)
                with gen_lock:
                    process_status = gen.get("process_status")
                    if process_status == "process:main":
                        break
            force_refresh = True

        refresh_id = gen.get("refresh", -1)
        if not force_refresh and step_idx < 0:
            if refresh_id < 0:
                return
            ui_refresh = state.get("refresh", 0)
            if ui_refresh >= refresh_id:
                return

        if override_num_inference_steps > 0:
            gen["num_inference_steps"] = override_num_inference_steps

        total_steps = gen.get("num_inference_steps", 0)
        status_text = str(gen.get("progress_status", ""))
        state["refresh"] = refresh_id

        phase = ""
        if read_state:
            phase, stored_step = gen.get("progress_phase", ("", step_idx))
            step_idx = stored_step
        else:
            step_idx += 1
            if gen.get("abort", False):
                phase = "Aborting"
            elif step_idx == total_steps:
                phase = "VAE Decoding"
            else:
                if pass_no <= 0:
                    phase = "Denoising"
                elif pass_no == 1:
                    phase = "Denoising First Pass"
                elif pass_no == 2:
                    phase = "Denoising Second Pass"
                elif pass_no == 3:
                    phase = "Denoising Third Pass"
                else:
                    phase = f"Denoising {pass_no}th Pass"
                if denoising_extra:
                    phase += f" | {denoising_extra}"
            gen["progress_phase"] = (phase, step_idx)
        if not phase:
            phase = gen.get("progress_phase", ("", -1))[0]

        elapsed = time.time() - start_time
        status_with_time = merge_status_context(status_text, f"{phase} | {format_duration(elapsed)}")

        if step_idx >= 0:
            progress_args: List[Any] = [(step_idx, total_steps), status_with_time, total_steps]
        else:
            progress_args = [0, status_with_time]

        gen["progress_args"] = progress_args
        send_cmd("progress", progress_args)

        if latent is not None:
            latent = latent.to("cpu", non_blocking=True)
            send_cmd("preview", latent)

    return callback


def make_cli_callback_builder(wgp_module: Any, logger: Logger) -> CallbackBuilder:
    def builder(state: Dict[str, Any], send_cmd: SendCommand, status: str, num_inference_steps: int):
        _ = status  # status is tracked via state; retained for interface compatibility.
        def pause_cleanup() -> None:
            offload_obj = getattr(wgp_module, "offloadobj", None)
            unload = getattr(offload_obj, "unload_all", None)
            if callable(unload):
                unload()

        return build_progress_callback(
            state,
            send_cmd,
            num_inference_steps=num_inference_steps,
            logger=logger,
            pause_cleanup=pause_cleanup,
        )

    return builder


def build_send_cmd(state: Dict[str, Any], logger: Logger) -> SendCommand:
    def send_cmd(command: str, payload: Any = None):
        gen = _ensure_gen_state(state)
        if command == "progress":
            gen["in_progress"] = True
            gen["progress_args"] = payload
            message = ""
            if isinstance(payload, list) and payload:
                step_info = payload[0]
                message = payload[1] if len(payload) > 1 else ""
                if message is None:
                    message = ""
                else:
                    message = str(message)
                if isinstance(step_info, (list, tuple)) and len(step_info) == 2:
                    current, total = step_info
                    gen["progress_step"] = current
                    gen["progress_total"] = total
                    logger.info("[progress] %s/%s %s", current, total, message or "")
                else:
                    gen.pop("progress_step", None)
                    gen.pop("progress_total", None)
                    logger.info("[progress] %s", message or step_info)
            else:
                gen.pop("progress_step", None)
                gen.pop("progress_total", None)
                message = str(payload) if payload is not None else ""
                logger.info("[progress] %s", message)
            gen["progress_status"] = message
            return

        if command == "status":
            status_message = str(payload) if payload is not None else ""
            state["status"] = status_message
            state["status_display"] = bool(status_message)
            gen["progress_status"] = status_message
            logger.info("[status] %s", status_message)
            return

        if command == "info":
            info_message = str(payload) if payload is not None else ""
            logger.info("[info] %s", info_message)
            return

        if command == "error":
            error_message = str(payload) if payload is not None else ""
            logger.error("[error] %s", error_message)
            raise RuntimeError(error_message)

        if command == "output":
            outputs = gen.get("file_list", [])
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
