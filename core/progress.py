from __future__ import annotations

from typing import Any, Dict

_refresh_id = 0


def _ensure_gen_state(state: Dict[str, Any]) -> Dict[str, Any]:
    gen = state.get("gen")
    if gen is None:
        gen = {}
        state["gen"] = gen
    return gen


def format_duration(seconds: float) -> str:
    """
    Convert a duration in seconds into a human-readable string.

    Examples:
        4.2  -> "4.2s"
        75.0 -> "1m 15s"
        3725 -> "1h 02m 05s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if seconds >= 60:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


def merge_status_context(status: str = "", context: str = "") -> str:
    """
    Combine status and context strings while preserving existing time markers.

    Mirrors the legacy behaviour in `wgp.py` so CLI progress logs stay familiar.
    """
    status = status or ""
    context = context or ""
    if not status:
        return context
    if not context:
        return status
    if "|" in context:
        parts = context.split("|")
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""
        return f"{status} - {left} | {right}"
    return f"{status} - {context}"


def clear_status(state: Dict[str, Any]) -> None:
    """Reset progress counters for the active generation state."""

    gen = _ensure_gen_state(state)
    gen["extra_windows"] = 0
    gen["total_windows"] = 1
    gen["window_no"] = 1
    gen["extra_orders"] = 0
    gen["repeat_no"] = 0
    gen["total_generation"] = 0


def get_generation_status(
    prompt_no: int,
    prompts_max: int,
    repeat_no: int,
    repeat_max: int,
    window_no: int,
    total_windows: int,
) -> str:
    """Compose a human-readable status string for the current queue counters."""

    if prompts_max == 1:
        if repeat_max <= 1:
            status = ""
        else:
            status = f"Sample {repeat_no}/{repeat_max}"
    else:
        if repeat_max <= 1:
            status = f"Prompt {prompt_no}/{prompts_max}"
        else:
            status = f"Prompt {prompt_no}/{prompts_max}, Sample {repeat_no}/{repeat_max}"
    if total_windows > 1:
        if status:
            status += ", "
        status += f"Sliding Window {window_no}/{total_windows}"
    return status


def get_latest_status(state: Dict[str, Any], context: str = "") -> str:
    """Return the latest status string, optionally merged with additional context."""

    gen = _ensure_gen_state(state)
    prompt_no = gen.get("prompt_no", 0)
    prompts_max = gen.get("prompts_max", 0)
    total_generation = gen.get("total_generation", 1)
    repeat_no = gen.get("repeat_no", 0)
    total_generation += gen.get("extra_orders", 0)
    total_windows = gen.get("total_windows", 0) + gen.get("extra_windows", 0)
    window_no = gen.get("window_no", 0)
    status = get_generation_status(prompt_no, prompts_max, repeat_no, total_generation, window_no, total_windows)
    return merge_status_context(status, context)


def get_new_refresh_id() -> int:
    """Return a monotonically increasing refresh identifier."""

    global _refresh_id
    _refresh_id += 1
    return _refresh_id


def update_status(state: Dict[str, Any]) -> None:
    """Materialise the latest status string and assign a fresh refresh id."""

    gen = _ensure_gen_state(state)
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()
