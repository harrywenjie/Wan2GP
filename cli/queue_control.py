"""CLI helper for controlling a running queue controller instance."""

from __future__ import annotations

import argparse
import socket
import sys
from typing import List


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send pause/resume commands to a running queue control server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host where the queue control server is listening (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port where the queue control server is listening.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pause", help="Request the active generation to pause.")
    subparsers.add_parser("resume", help="Resume generation after a pause.")
    subparsers.add_parser("status", help="Show queue status from the running process.")
    subparsers.add_parser("abort", help="Signal an abort for the active generation.")
    return parser


def send_command(host: str, port: int, command: str) -> List[str]:
    """Send a queue control command to the TCP server and return its responses."""
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall((command + "\n").encode("utf-8"))
        responses: List[str] = []
        sock.settimeout(0.2)
        with sock.makefile("r", encoding="utf-8", newline="\n") as sock_file:
            while True:
                try:
                    line = sock_file.readline()
                except socket.timeout:
                    break
                if not line:
                    break
                stripped = line.rstrip("\r\n")
                if stripped:
                    responses.append(stripped)
        return responses


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    responses = send_command(args.host, args.port, args.command)
    if not responses:
        print("No response from queue control server.", file=sys.stderr)
        return 1
    for line in responses:
        print(line)
    if responses[-1].startswith("ERR"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
