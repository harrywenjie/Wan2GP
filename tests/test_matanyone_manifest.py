from pathlib import Path
from typing import List
from unittest import TestCase

from cli.manifest import ArtifactCapture, build_matanyone_artifacts
from core.io.media import AudioSaveConfig, VideoSaveConfig


class MatAnyOneManifestTests(TestCase):
    def _make_capture(self, path: str) -> ArtifactCapture:
        config = VideoSaveConfig(
            fps=24,
            codec_type="libx264_8",
            container="mp4",
        )
        return ArtifactCapture(path=Path(path), kind="video", config=config)

    def test_build_artifacts_with_manifest_captures(self) -> None:
        captures: List[ArtifactCapture] = [
            self._make_capture("/tmp/foreground.mp4"),
            self._make_capture("/tmp/foreground_alpha.mp4"),
            ArtifactCapture(path=Path("/tmp/foreground.zip"), kind="mask_archive", config=None),
        ]

        artifacts = build_matanyone_artifacts(
            foreground_path=Path("/tmp/foreground.mp4"),
            alpha_path=Path("/tmp/foreground_alpha.mp4"),
            rgba_zip_path=Path("/tmp/foreground.zip"),
            frames_processed=48,
            fps=24.0,
            metadata_mode="json",
            captures=captures,
            codec="libx264_8",
            container="mp4",
        )

        self.assertEqual(len(artifacts), 3)
        foreground, alpha, rgba = artifacts
        self.assertEqual(foreground["role"], "mask_foreground")
        self.assertEqual(foreground["codec"], "libx264_8")
        self.assertEqual(foreground["container"], "mp4")
        self.assertAlmostEqual(foreground["duration_s"], 2.0)
        self.assertEqual(foreground["metadata_sidecar"], "/tmp/foreground.json")

        self.assertEqual(alpha["role"], "mask_alpha")
        self.assertEqual(alpha["codec"], "libx264_8")
        self.assertEqual(alpha["container"], "mp4")
        self.assertEqual(alpha["metadata_sidecar"], "/tmp/foreground_alpha.json")

        self.assertEqual(rgba["role"], "rgba_archive")
        self.assertEqual(rgba["codec"], None)
        self.assertEqual(rgba["container"], "zip")
        self.assertIsNone(rgba["metadata_sidecar"])

    def test_foreground_temp_capture_maps_to_final_path(self) -> None:
        captures = [
            self._make_capture("/tmp/session_tmp.mp4"),
            self._make_capture("/tmp/session_alpha.mp4"),
        ]

        artifacts = build_matanyone_artifacts(
            foreground_path=Path("/tmp/session.mp4"),
            alpha_path=Path("/tmp/session_alpha.mp4"),
            rgba_zip_path=None,
            frames_processed=24,
            fps=24.0,
            metadata_mode="metadata",
            captures=captures,
            codec=None,
            container=None,
        )

        self.assertEqual(len(artifacts), 2)
        foreground, alpha = artifacts
        self.assertEqual(foreground["role"], "mask_foreground")
        self.assertEqual(foreground["codec"], "libx264_8")
        self.assertEqual(foreground["container"], "mp4")
        self.assertIsNone(foreground["metadata_sidecar"])

        self.assertEqual(alpha["role"], "mask_alpha")

    def test_audio_metadata_enriches_entries(self) -> None:
        audio_path = Path("/tmp/session_audio.wav")
        audio_capture = ArtifactCapture(
            path=audio_path,
            kind="audio",
            config=AudioSaveConfig(sample_rate=16000, format="wav", subtype="PCM_16"),
        )

        artifacts = build_matanyone_artifacts(
            foreground_path=Path("/tmp/session.mp4"),
            alpha_path=Path("/tmp/session_alpha.mp4"),
            rgba_zip_path=None,
            frames_processed=24,
            fps=24.0,
            metadata_mode="json",
            captures=[audio_capture],
            codec=None,
            container=None,
            audio_metadata=[
                {
                    "path": str(audio_path),
                    "sample_rate": 22050,
                    "duration": 1.5,
                    "language": "eng",
                    "channels": 1,
                }
            ],
        )

        audio_entries = [artifact for artifact in artifacts if artifact["role"] == "audio"]
        self.assertEqual(len(audio_entries), 1)
        audio_entry = audio_entries[0]
        self.assertEqual(audio_entry["path"], str(audio_path.resolve()))
        self.assertEqual(audio_entry["sample_rate"], 22050)
        self.assertAlmostEqual(audio_entry["duration_s"], 1.5)
        self.assertEqual(audio_entry["language"], "eng")
        self.assertEqual(audio_entry["channels"], 1)
        self.assertEqual(audio_entry["container"], "wav")
        self.assertEqual(audio_entry["codec"], "PCM_16")
        self.assertEqual(audio_entry["metadata_sidecar"], str(audio_path.with_suffix(".json").resolve()))
