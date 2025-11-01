import unittest
from pathlib import Path
from typing import Any, Dict, Optional

from cli.queue_controller import QueueController
from cli.queue_state import QueueStateTracker, generate_queue_summary, normalize_audio_tracks


class _StubTaskInputs:
    def prepare_inputs_dict(
        self,
        section: str,
        inputs: Dict[str, Any],
        *,
        model_type: Optional[str] = None,
        model_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        if section != "metadata":
            raise AssertionError("Unexpected section requested in stub")
        return {}


class _StubProductionManager:
    def __init__(self) -> None:
        self.queue_tracker = QueueStateTracker()
        self._task_inputs = _StubTaskInputs()

    def task_inputs(self) -> _StubTaskInputs:
        return self._task_inputs


class QueueAudioMetadataTests(unittest.TestCase):
    def test_normalize_audio_tracks_converts_entries(self) -> None:
        raw = [
            {
                "path": Path("/tmp/audio.wav"),
                "sample_rate": "48000",
                "duration": "1.5",
                "language": "eng",
                "channels": "2",
            },
            {"path": None},
        ]

        normalised = normalize_audio_tracks(raw)
        self.assertEqual(len(normalised), 1)
        entry = normalised[0]
        self.assertEqual(entry["path"], "/tmp/audio.wav")
        self.assertEqual(entry["sample_rate"], 48000)
        self.assertAlmostEqual(entry["duration_s"], 1.5)
        self.assertEqual(entry["channels"], 2)
        self.assertEqual(entry["language"], "eng")

    def test_queue_summary_includes_audio_section(self) -> None:
        audio = [
            {
                "path": "/tmp/audio.wav",
                "sample_rate": 44100,
                "duration_s": 2.0,
                "language": "eng",
                "channels": 1,
            }
        ]

        summary = generate_queue_summary([], audio_tracks=audio)
        self.assertIn("Audio tracks:", summary)
        self.assertIn("path=/tmp/audio.wav", summary)
        self.assertIn("duration=2.00s", summary)

    def test_queue_metrics_surface_audio_tracks(self) -> None:
        state: Dict[str, Any] = {"gen": {"queue": [], "audio_tracks": [
            {
                "path": "/tmp/audio.wav",
                "sample_rate": 22050,
                "duration_s": 3.25,
                "language": "eng",
                "channels": 1,
            }
        ]}}

        controller = QueueController(manager=_StubProductionManager(), state=state)
        controller._update_queue_tracking(state["gen"].get("queue", []))  # type: ignore[arg-type]

        metrics = controller.queue_metrics()
        audio_tracks = metrics.get("audio_tracks")
        self.assertIsInstance(audio_tracks, list)
        self.assertEqual(audio_tracks[0]["path"], "/tmp/audio.wav")
        self.assertAlmostEqual(audio_tracks[0]["duration_s"], 3.25)
        summary = metrics.get("queue_summary")
        self.assertIn("Audio tracks:", summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
