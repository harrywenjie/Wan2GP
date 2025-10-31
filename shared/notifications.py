from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


class GenerationNotifier:
    """
    Interface for generation progress hooks.

    CLI callers can provide light-weight implementations while legacy
    wgp flows can proxy the existing helpers.
    """

    def reset_progress(self, state: Dict[str, Any]) -> None:
        """Reset progress counters when a task completes or aborts."""

    def refresh_preview(self, task: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Update task previews when new reference assets land."""

    def notify_video_ready(self, video_path: Optional[str] = None) -> None:
        """Emit a signal when a single video finishes encoding."""

    def notify_generation_complete(self) -> None:
        """Emit a signal when a batch of prompts finishes."""


@dataclass
class LegacyGenerationNotifier(GenerationNotifier):
    """
    Bridges the legacy wgp helpers to the new notifier contract.

    Accepts callables so future refactors can feed alternative behaviour
    (e.g., no-op implementations for pure CLI logging).
    """

    clear_status_fn: Callable[[Dict[str, Any]], None]
    update_task_thumbnails_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]]
    notification_sound_module: Any
    server_config: Dict[str, Any]
    single_default_enabled: int = 0
    batch_default_enabled: int = 1

    def reset_progress(self, state: Dict[str, Any]) -> None:
        self.clear_status_fn(state)

    def refresh_preview(self, task: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        if self.update_task_thumbnails_fn is None:
            return
        self.update_task_thumbnails_fn(task, inputs)

    def notify_video_ready(self, video_path: Optional[str] = None) -> None:
        self._notify_with_sound(video_path=video_path, default_enabled=self.single_default_enabled)

    def notify_generation_complete(self) -> None:
        self._notify_with_sound(video_path=None, default_enabled=self.batch_default_enabled)

    def _notify_with_sound(self, *, video_path: Optional[str], default_enabled: int) -> None:
        module = self.notification_sound_module
        if module is None:
            return
        try:
            enabled = self.server_config.get("notification_sound_enabled", default_enabled)
            if not enabled:
                return
            volume = self.server_config.get("notification_sound_volume", 50)
            module.notify_video_completion(video_path=video_path, volume=volume)
        except Exception as exc:
            print(f"Warning: notifier sound hook failed ({exc})")


def create_legacy_notifier(
    *,
    clear_status_fn: Callable[[Dict[str, Any]], None],
    update_task_thumbnails_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]],
    notification_sound_module: Any,
    server_config: Dict[str, Any],
) -> LegacyGenerationNotifier:
    return LegacyGenerationNotifier(
        clear_status_fn=clear_status_fn,
        update_task_thumbnails_fn=update_task_thumbnails_fn,
        notification_sound_module=notification_sound_module,
        server_config=server_config,
    )
