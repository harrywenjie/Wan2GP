from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from cli.queue_state import (
    QueueStateTracker,
    build_queue_snapshot,
    clear_pending_tasks,
    generate_queue_summary as core_generate_queue_summary,
    request_abort,
    reset_generation_counters,
    update_queue_tracking,
)
from shared.utils.image import pil_to_base64_uri
from shared.utils.notifications import (
    notify_debug,
    notify_info as default_notify_info,
)
from shared.utils.process_locks import get_gen_info

PREVIEW_INPUT_KEYS: Sequence[str] = (
    "image_start",
    "video_source",
    "image_end",
    "video_guide",
    "image_guide",
    "video_mask",
    "image_mask",
    "image_refs",
)

PREVIEW_LABELS: Sequence[str] = (
    "Start Image",
    "Video Source",
    "End Image",
    "Video Guide",
    "Image Guide",
    "Video Mask",
    "Image Mask",
    "Image Reference",
)


@dataclass
class PreviewImages:
    start_data: Optional[List[Any]]
    end_data: Optional[List[Any]]
    start_labels: List[str]
    end_labels: List[str]

    def build_payload(self) -> Dict[str, Any]:
        start_base64 = _encode_preview_list(self.start_data)
        end_base64 = _encode_preview_list(self.end_data)
        return {
            "start_image_labels": list(self.start_labels),
            "end_image_labels": list(self.end_labels),
            "start_image_data_base64": start_base64,
            "end_image_data_base64": end_base64,
        }


def _normalise_input(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value.copy()
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _encode_preview_list(images: Optional[List[Any]]) -> Optional[List[Optional[str]]]:
    if images is None:
        return None
    encoded = [pil_to_base64_uri(item, format="jpeg", quality=70) for item in images]
    return encoded


def get_preview_images(inputs: Mapping[str, Any]) -> PreviewImages:
    start_data: Optional[List[Any]] = None
    end_data: Optional[List[Any]] = None
    start_labels: List[str] = []
    end_labels: List[str] = []

    for key, label in zip(PREVIEW_INPUT_KEYS, PREVIEW_LABELS):
        raw_items = _normalise_input(inputs.get(key))
        if not raw_items:
            continue
        labels = [label] * len(raw_items)
        if start_data is None:
            start_data = raw_items
            start_labels.extend(labels)
        else:
            if end_data is None:
                end_data = raw_items
            else:
                end_data.extend(raw_items)
            end_labels.extend(labels)

    if start_data and len(start_data) > 1 and end_data is None:
        end_data = start_data[1:]
        end_labels = start_labels[1:]
        start_data = start_data[:1]
        start_labels = start_labels[:1]

    return PreviewImages(
        start_data=start_data,
        end_data=end_data,
        start_labels=start_labels,
        end_labels=end_labels,
    )


def update_task_thumbnails(task: Dict[str, Any], inputs: Mapping[str, Any]) -> None:
    """
    Update the thumbnail metadata for a queue task using the provided inputs.
    """

    preview = get_preview_images(inputs)
    payload = preview.build_payload()
    task.update(payload)

    attachment_labels: List[str] = []
    for label in preview.start_labels + preview.end_labels:
        if label and label not in attachment_labels:
            attachment_labels.append(label)

    task_identifier = task.get("id")
    task_label = str(task_identifier) if task_identifier is not None else "<unknown>"
    if attachment_labels:
        notify_debug(
            f"Queue task {task_label} preview assets updated: {', '.join(attachment_labels)}"
        )
    else:
        notify_debug(f"Queue task {task_label} preview assets cleared.")


def generate_queue_summary(queue: Iterable[Dict[str, Any]]) -> str:
    """Return a human-readable summary of queued tasks."""
    return core_generate_queue_summary(list(queue))


def update_queue_data(
    queue: List[Dict[str, Any]],
    *,
    tracker: Optional[QueueStateTracker] = None,
    audio_tracks: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Update queue tracking metadata and return a lightweight snapshot.

    Mirrors the legacy helper from `wgp.py` but lives alongside other CLI queue
    utilities so orchestration no longer depends on the legacy module.
    """

    update_queue_tracking(queue, tracker, audio_tracks=audio_tracks)
    return dict(build_queue_snapshot(queue, audio_tracks=audio_tracks))


def clear_queue_action(
    state: Dict[str, Any],
    *,
    queue: Optional[List[Dict[str, Any]]] = None,
    lock: Optional[Any] = None,
    tracker: Optional[QueueStateTracker] = None,
    interrupt_callback: Optional[Callable[[], None]] = None,
    notify_info: Callable[[str], None] = default_notify_info,
) -> Dict[str, Any]:
    """
    Abort the active task (when running) and clear any pending queue entries.

    The helper accepts optional orchestration primitives so both the CLI queue
    controller and the legacy `wgp` callers can share the implementation.
    """

    gen = get_gen_info(state)
    active_queue: List[Dict[str, Any]] = queue or gen.get("queue", [])
    lock_ctx = lock if lock is not None else nullcontext()

    with lock_ctx:
        aborted_current = request_abort(gen)
        if aborted_current and interrupt_callback is not None:
            try:
                interrupt_callback()
            except Exception:
                # Best-effort interrupt; downstream cleanup still runs.
                pass
        cleared_pending = clear_pending_tasks(gen, active_queue)
        metrics = update_queue_data(
            active_queue,
            tracker=tracker,
            audio_tracks=gen.get("audio_tracks"),
        )
        if aborted_current or cleared_pending:
            reset_generation_counters(gen, preserve_abort=aborted_current)

    if aborted_current and cleared_pending:
        notify_info("Queue cleared and current generation aborted.")
    elif aborted_current:
        notify_info("Current generation aborted.")
    elif cleared_pending:
        notify_info("Queue cleared.")
    else:
        notify_info("Queue is already empty or only contains the active task (which wasn't aborted now).")

    result = dict(metrics)
    result["aborted"] = aborted_current
    result["cleared"] = cleared_pending
    return result
