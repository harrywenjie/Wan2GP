import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import TestCase, mock

import numpy as np

from core.io.media import (
    AudioSaveConfig,
    ImageSaveConfig,
    MaskSaveConfig,
    MediaPersistenceContext,
    VideoSaveConfig,
)
from preprocessing.matanyone.app import MatAnyOneRequest, _save_outputs


class RecordingContext(MediaPersistenceContext):
    """MediaPersistenceContext subclass that records persistence requests."""

    def __init__(
        self,
        *,
        container: str = "mkv",
        codec: str = "libx265",
        save_masks: bool = True,
    ) -> None:
        super().__init__(
            video_template=VideoSaveConfig(codec_type=codec, container=container),
            image_template=ImageSaveConfig(),
            audio_template=AudioSaveConfig(),
            mask_template=MaskSaveConfig(),
            save_debug_masks=save_masks,
        )
        self.video_calls = []
        self.mask_calls = []

    @staticmethod
    def _ensure_suffix(path: str, container: str) -> str:
        suffix = container if container.startswith(".") else f".{container}"
        return path if path.endswith(suffix) else f"{path}{suffix}"

    def save_video(
        self,
        data: Any,
        target_path: Optional[str],
        *,
        logger=None,
        config=None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        record = {
            "target_path": target_path,
            "overrides": dict(overrides or {}),
        }
        self.video_calls.append(record)
        return self._ensure_suffix(str(target_path), self.video_template.container)

    def save_mask_archive(
        self,
        frames: Any,
        target_path: str,
        *,
        logger=None,
        config=None,
        overrides: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[str]:
        self.mask_calls.append({"target_path": target_path, "force": force})
        return super().save_mask_archive(
            frames,
            target_path,
            logger=logger,
            config=config,
            overrides=overrides,
            force=force,
        )


class MatAnyOnePersistenceTests(TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.output_dir = Path(self._tmpdir.name)

    def _make_request(
        self,
        *,
        context: RecordingContext,
        codec: str = "libx265",
        mask_type: str = "wangp",
    ) -> MatAnyOneRequest:
        return MatAnyOneRequest(
            input_path=self.output_dir / "sample.mp4",
            template_mask_path=self.output_dir / "template.png",
            output_dir=self.output_dir,
            codec=codec,
            metadata_mode="json",
            mask_type=mask_type,
            media_context=context,
        )

    def test_save_outputs_uses_media_context_overrides(self) -> None:
        context = RecordingContext(container="mkv", codec="libx265", save_masks=True)
        request = self._make_request(context=context, codec="libx265")

        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]
        alpha_frames = [np.zeros((2, 2, 1), dtype=np.uint8) for _ in range(2)]

        result = _save_outputs(
            request=request,
            frames=frames,
            alpha_frames=alpha_frames,
            fps=24.0,
            audio_tracks=[],
            audio_metadata=None,
            foreground_suffix="",
            alpha_suffix="_alpha",
            rgba_frames=None,
        )

        self.assertEqual(len(context.video_calls), 2, "foreground and alpha videos should be persisted")
        for call in context.video_calls:
            self.assertEqual(call["overrides"].get("fps"), 24.0)
            self.assertEqual(call["overrides"].get("codec_type"), "libx265")

        self.assertEqual(result.metadata["codec"], "libx265")
        self.assertEqual(result.metadata["container"], "mkv")
        self.assertEqual(result.foreground_path.suffix, ".mkv")
        self.assertEqual(result.alpha_path.suffix, ".mkv")

    def test_mask_archive_respects_save_masks_toggle(self) -> None:
        context = RecordingContext(container="mp4", codec="libx264_8", save_masks=False)
        request = self._make_request(context=context, mask_type="alpha")

        frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
        alpha_frames = [np.zeros((2, 2, 1), dtype=np.uint8)]
        rgba_frames = [np.zeros((2, 2, 4), dtype=np.uint8)]

        result = _save_outputs(
            request=request,
            frames=frames,
            alpha_frames=alpha_frames,
            fps=12.0,
            audio_tracks=[],
            audio_metadata=None,
            foreground_suffix="_RGBA",
            alpha_suffix="_alpha",
            rgba_frames=rgba_frames,
        )

        self.assertEqual(len(context.mask_calls), 1, "mask archive should be attempted exactly once")
        self.assertFalse(context.save_debug_masks)
        self.assertFalse(context.mask_calls[0]["force"])
        self.assertIsNone(result.rgba_zip_path)
        self.assertFalse(list(self.output_dir.glob("*.zip")), "no zip files should be written when save_masks is disabled")

    @mock.patch("models.wan.alpha.utils.write_zip_file")
    def test_mask_archive_emitted_when_enabled(self, mock_write_zip_file: mock.MagicMock) -> None:
        context = RecordingContext(container="mp4", codec="libx264_8", save_masks=True)
        request = self._make_request(context=context, mask_type="alpha")

        frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
        alpha_frames = [np.zeros((2, 2, 1), dtype=np.uint8)]
        rgba_frames = [np.zeros((2, 2, 4), dtype=np.uint8)]

        result = _save_outputs(
            request=request,
            frames=frames,
            alpha_frames=alpha_frames,
            fps=10.0,
            audio_tracks=[],
            audio_metadata=None,
            foreground_suffix="_RGBA",
            alpha_suffix="_alpha",
            rgba_frames=rgba_frames,
        )

        self.assertEqual(len(context.mask_calls), 1)
        self.assertTrue(context.save_debug_masks)
        self.assertFalse(context.mask_calls[0]["force"])
        expected_zip = result.foreground_path.with_suffix(".zip")
        self.assertEqual(result.rgba_zip_path, expected_zip)
        mock_write_zip_file.assert_called_once_with(str(expected_zip), rgba_frames)

