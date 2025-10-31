from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _normalize_prompt(text: Any) -> str:
    if text is None:
        return "<empty>"
    prompt_text = str(text).replace("\n", " ").strip()
    if not prompt_text:
        return "<empty>"
    if len(prompt_text) > 64:
        return prompt_text[:61] + "..."
    return prompt_text


def generate_queue_summary(queue: List[Dict[str, Any]]) -> str:
    """
    Produce a plain-text snapshot of the queued tasks for CLI consumption.

    The first entry in the queue represents the active task; subsequent entries
    are displayed with their prompt, repeat count, and high-level metadata.
    """
    if len(queue) <= 1:
        return "Queue is empty."

    header = f"{'Row':>3}  {'Task':>6}  {'Rpt':>3}  {'Frames':>6}  {'Steps':>5}  {'Start':>5}  {'End':>3}  Prompt"
    lines = ["Queued tasks:", "", header, "-" * len(header)]

    for index, item in enumerate(queue[1:], start=1):
        task_id = item.get("id", index)
        repeats = item.get("repeats", 1)
        length = item.get("length", "-")
        steps = item.get("steps", "-")
        start_flag = "yes" if item.get("start_image_data_base64") else "no"
        end_flag = "yes" if item.get("end_image_data_base64") else "no"
        prompt_text = _normalize_prompt(item.get("prompt"))
        task_display = str(task_id)
        if len(task_display) > 6:
            task_display = task_display[-6:]
        attachment_labels: List[str] = []
        for label in item.get("start_image_labels") or []:
            if label and label not in attachment_labels:
                attachment_labels.append(label)
        for label in item.get("end_image_labels") or []:
            if label and label not in attachment_labels:
                attachment_labels.append(label)
        media_suffix = ""
        if attachment_labels:
            media_suffix = " [" + ", ".join(attachment_labels) + "]"

        lines.append(
            f"{index:>3}  {task_display:>6}  {str(repeats):>3}  {str(length):>6}  {str(steps):>5}  {start_flag:>5}  {end_flag:>3}  {prompt_text}"
            + media_suffix
        )

    lines.append("")
    lines.append(f"Total queued entries (excluding active): {len(queue) - 1}")
    return "\n".join(lines)


def build_queue_snapshot(queue: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "queue_summary": generate_queue_summary(queue),
        "queue_length": max(len(queue) - 1, 0),
    }


@dataclass
class QueueStateTracker:
    """
    Maintain the latest queue snapshot for headless controllers.

    The tracker stores a shallow copy of the queue alongside derived metrics so
    callers can surface consistent telemetry without relying on `wgp` globals.
    """

    _queue: List[Dict[str, Any]] = None
    _queue_summary: str = "Queue is empty."
    _queue_length: int = 0

    def __post_init__(self) -> None:
        if self._queue is None:
            self._queue = []

    def update(self, queue: List[Dict[str, Any]]) -> Dict[str, Any]:
        snapshot = list(queue)
        self._queue = snapshot
        metrics = build_queue_snapshot(snapshot)
        self._queue_summary = metrics["queue_summary"]
        self._queue_length = metrics["queue_length"]
        return metrics

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self._queue)

    def metrics(self) -> Dict[str, Any]:
        return {
            "queue_summary": self._queue_summary,
            "queue_length": self._queue_length,
        }


_default_tracker = QueueStateTracker()


def get_default_tracker() -> QueueStateTracker:
    return _default_tracker


def update_queue_tracking(
    queue: List[Dict[str, Any]],
    tracker: Optional[QueueStateTracker] = None,
) -> Dict[str, Any]:
    active_tracker = tracker or _default_tracker
    return active_tracker.update(queue)


QUEUE_COUNTER_DEFAULTS = {
    "extra_windows": 0,
    "total_windows": 1,
    "window_no": 1,
    "extra_orders": 0,
    "repeat_no": 0,
    "total_generation": 0,
    "prompts_max": 0,
    "prompt_no": 0,
    "progress_status": "",
    "progress_args": None,
}


def reset_generation_counters(gen: Dict[str, Any], *, preserve_abort: bool = False) -> None:
    """
    Reset queue-related counters so future runs start from a clean slate.

    Callers may choose to preserve the abort flag when they still expect the
    worker thread to notice an outstanding abort request.
    """

    for key, value in QUEUE_COUNTER_DEFAULTS.items():
        gen[key] = value
    if preserve_abort:
        gen.setdefault("abort", True)
    else:
        gen["abort"] = False
        gen["in_progress"] = False
    gen.setdefault("refresh", 0)


def request_abort(gen: Dict[str, Any]) -> bool:
    """
    Signal that the active task should abort.

    Returns True when a generation was in progress and the abort flag changed.
    """

    if not gen.get("in_progress"):
        return False
    gen["abort"] = True
    gen["extra_orders"] = 0
    return True


def clear_pending_tasks(
    gen: Dict[str, Any],
    queue: List[Dict[str, Any]],
) -> bool:
    """
    Drop any queued tasks beyond the active item.

    Returns True when the queue was modified.
    """

    if not queue:
        return False
    if len(queue) > 1:
        queue.clear()
        gen["prompts_max"] = 0
        return True
    active = queue[0]
    if active is not None and active.get("id") is not None:
        queue.clear()
        gen["prompts_max"] = 0
        return True
    if active is None:
        queue.clear()
        return True
    return False
