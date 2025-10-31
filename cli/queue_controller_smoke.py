"""
Lightweight smoke harness for the headless queue controller.

This module exercises the pause/resume handshake without invoking the full
generation stack, enabling quick regressions checks in headless environments.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from cli.queue_state import QueueStateTracker
from shared.utils.process_locks import gen_lock

from cli.queue_control import send_command
from cli.queue_control_server import QueueControlServer
from cli.queue_controller import QueueController


class _StubManager:
    """
    Minimal stand-in for ProductionManager used by the smoke harness.

    The stub simulates progress events and honours pause/resume requests by
    mirroring the process_status semantics expected by the queue controller.
    """

    def __init__(self) -> None:
        self.queue_tracker = QueueStateTracker()
        self.wgp = SimpleNamespace()

    def run_generation(  # type: ignore[override]
        self,
        params: Dict[str, Any],
        *,
        state: Dict[str, Any],
        send_cmd,
        output_dir_override: Optional[Any] = None,
        image_output_dir_override: Optional[Any] = None,
        attr_overrides: Optional[Dict[str, Any]] = None,
        server_config_overrides: Optional[Dict[str, Any]] = None,
        notifier=None,
        callback_builder=None,
        plugin_data=None,
        task_stub=None,
        task_seed: Optional[int] = None,
    ) -> List[str]:
        gen_state = state.setdefault("gen", {})
        gen_state["in_progress"] = True
        outputs = gen_state.setdefault("file_list", [])
        if not outputs:
            outputs.append("out.mp4")
        steps = 3
        for idx in range(steps):
            send_cmd(
                "progress",
                [(idx, steps), f"stub-step-{idx}", steps],
            )
            time.sleep(0.01)
            while True:
                with gen_lock:
                    status = gen_state.get("process_status")
                    if status == "request:pause":
                        gen_state["process_status"] = "process:pause"
                        status = "process:pause"
                    if status != "process:pause":
                        break
                time.sleep(0.01)
        gen_state["in_progress"] = False
        return list(outputs)


def run_smoke() -> List[str]:
    """
    Execute a single queue task while toggling pause/resume to confirm the
    controller honours the pause handshake and resumes cleanly.
    """

    state: Dict[str, Any] = {"gen": {"queue": []}}
    manager = _StubManager()
    controller = QueueController(manager=manager, state=state)

    events: List[str] = []

    def send_cmd(command: str, payload: Any = None) -> None:
        events.append(command)

    def parse_status(lines: List[str]) -> Dict[str, Any]:
        if not lines:
            raise RuntimeError("Status command returned no data.")
        status_line = lines[-1]
        prefix, _, payload = status_line.partition(" ")
        if prefix != "OK" or not payload:
            raise RuntimeError(f"Unexpected status response: {status_line}")
        return json.loads(payload)

    def wait_for(condition, *, timeout: float, interval: float = 0.01) -> bool:
        deadline = time.time() + max(timeout, 0.0)
        while time.time() < deadline:
            if condition():
                return True
            time.sleep(interval)
        return condition()

    host = "127.0.0.1"
    server = QueueControlServer(
        controller=controller,
        state=state,
        logger=None,
        host=host,
        port=0,
    )
    port = server.start()

    def trigger_pause(port: int) -> None:
        # wait for the worker to enter the processing loop
        time.sleep(0.05)
        pause_resp = send_command(host, port, "pause smoke-harness")
        if not pause_resp or pause_resp[-1] != "OK paused":
            raise RuntimeError(f"Unexpected pause response: {pause_resp}")

        def paused() -> bool:
            status_payload = parse_status(send_command(host, port, "status"))
            return bool(status_payload.get("paused"))

        if not wait_for(paused, timeout=1.0):
            raise RuntimeError("QueueControlServer did not report paused state.")

        time.sleep(0.05)
        resume_resp = send_command(host, port, "resume")
        if not resume_resp or resume_resp[-1] != "OK resumed":
            raise RuntimeError(f"Unexpected resume response: {resume_resp}")

        def resumed() -> bool:
            status_payload = parse_status(send_command(host, port, "status"))
            return not status_payload.get("paused")

        if not wait_for(resumed, timeout=1.0):
            raise RuntimeError("QueueControlServer did not clear paused state after resume.")

    orchestrator = threading.Thread(target=trigger_pause, args=(port,), daemon=True)
    orchestrator.start()

    try:
        outputs = controller.run_single({"prompt": "test"}, send_cmd=send_cmd)
        orchestrator.join(timeout=2.0)

        if orchestrator.is_alive():
            raise RuntimeError("Pause/resume orchestrator did not finish.")
        if not outputs:
            raise RuntimeError("QueueController smoke harness returned no outputs.")
        if "progress" not in events:
            raise RuntimeError("QueueController smoke harness saw no progress events.")
        if controller.is_paused():
            raise RuntimeError("QueueController remained paused after completion.")

        final_status = parse_status(send_command(host, port, "status"))
        if final_status.get("paused"):
            raise RuntimeError("QueueControlServer reported paused state after completion.")
    finally:
        server.stop()

    return outputs


if __name__ == "__main__":
    paths = run_smoke()
    print("Queue controller smoke outputs:", paths)
