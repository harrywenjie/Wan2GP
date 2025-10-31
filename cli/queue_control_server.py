"""TCP control server exposing QueueController pause/resume operations."""

from __future__ import annotations

import json
import socketserver
import threading
from typing import Any, Dict, Optional


class _QueueControlTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        RequestHandlerClass,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.controller = None
        self.state: Dict[str, Any] = {}
        self.logger = None

    def dispatch_command(self, command: str) -> str:
        command = command.strip()
        if not command:
            return "ERR empty_command"

        parts = command.split(" ", 1)
        action = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else None

        controller = self.controller
        if controller is None:
            return "ERR controller_unavailable"

        logger = self.logger
        state = self.state

        if action == "pause":
            message = argument or "Pause requested via control channel."
            success = controller.request_pause(message)
            if success:
                if logger:
                    logger.info("Queue control: paused (%s)", message)
                return "OK paused"
            return "ERR pause_failed"

        if action == "resume":
            success = controller.resume()
            if success:
                if logger:
                    logger.info("Queue control: resumed")
                return "OK resumed"
            return "ERR resume_failed"

        if action == "abort":
            aborted = controller.request_abort()
            if aborted and logger:
                logger.warning("Queue control: abort requested")
            return "OK abort_signalled" if aborted else "ERR abort_not_in_progress"

        if action == "status":
            gen = state.setdefault("gen", {})
            metrics = controller.queue_metrics()
            payload = {
                "paused": controller.is_paused(),
                "in_progress": bool(gen.get("in_progress")),
                "abort": bool(gen.get("abort")),
                "queue_length": metrics.get("queue_length"),
                "queue_summary": metrics.get("queue_summary"),
                "progress_status": gen.get("progress_status"),
                "process_status": gen.get("process_status"),
            }
            return "OK " + json.dumps(payload)

        return "ERR unknown_command"


class _QueueControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                break
            response = self.server.dispatch_command(line.decode("utf-8"))
            self.wfile.write((response + "\n").encode("utf-8"))
            try:
                self.wfile.flush()
            except Exception:  # pragma: no cover - defensive flush
                pass


class QueueControlServer:
    """
    Lightweight TCP server exposing queue control commands.

    The server is optional and only runs while the owning CLI invocation is
    active. Call `start()` to launch the listener and `stop()` to shut it down.
    """

    def __init__(
        self,
        *,
        controller,
        state: Dict[str, Any],
        logger,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._controller = controller
        self._state = state
        self._logger = logger
        self._host = host
        self._port = port
        self._server: Optional[_QueueControlTCPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> int:
        if self._server is not None:
            raise RuntimeError("QueueControlServer already running.")

        server = _QueueControlTCPServer((self._host, self._port), _QueueControlHandler)
        server.controller = self._controller
        server.state = self._state
        server.logger = self._logger
        self._server = server

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread = thread
        thread.start()

        address = server.server_address
        bound_port = address[1]
        if self._logger:
            self._logger.info("Queue control server listening on %s:%s", self._host, bound_port)
        return bound_port

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def running_port(self) -> Optional[int]:
        if self._server is None:
            return None
        address = self._server.server_address
        return address[1]
