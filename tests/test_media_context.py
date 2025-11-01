import unittest
from unittest.mock import patch

import numpy as np
import torch

import core.io.media as media
from core.io.media import (
    AudioSaveConfig,
    ImageSaveConfig,
    MaskSaveConfig,
    MediaPersistenceContext,
    VideoSaveConfig,
)


class MediaPersistenceContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = MediaPersistenceContext(
            video_template=VideoSaveConfig(),
            image_template=ImageSaveConfig(),
            audio_template=AudioSaveConfig(retry=2),
            mask_template=MaskSaveConfig(retry=1),
            save_debug_masks=False,
        )

    def test_save_audio_tensor_uses_soundfile(self) -> None:
        if media.sf is None:  # pragma: no cover - guard for optional dependency
            self.skipTest("soundfile dependency missing")

        tensor = torch.zeros(4, dtype=torch.float32)
        with patch("core.io.media.sf.write") as mock_write:
            result = self.context.save_audio(tensor, "test.wav", sample_rate=22050)

        self.assertEqual(result, "test.wav")
        mock_write.assert_called_once()
        call_args = mock_write.call_args[0]
        self.assertEqual(call_args[0], "test.wav")
        self.assertTrue(isinstance(call_args[1], np.ndarray))
        self.assertEqual(call_args[2], 22050)

    def test_save_audio_requires_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            self.context.save_audio(np.zeros(8), "test.wav")

    @patch("models.wan.alpha.utils.write_zip_file")
    def test_save_mask_archive_respects_toggle(self, mock_write) -> None:
        result = self.context.save_mask_archive(["frame"], "mask.zip")
        self.assertIsNone(result)
        mock_write.assert_not_called()

        result_forced = self.context.save_mask_archive(["frame"], "mask.zip", force=True)
        self.assertEqual(result_forced, "mask.zip")
        mock_write.assert_called_once_with("mask.zip", ["frame"])

    @patch("models.wan.alpha.utils.write_zip_file")
    def test_save_mask_archive_honours_enabled_flag(self, mock_write) -> None:
        enabled_context = MediaPersistenceContext(
            video_template=VideoSaveConfig(),
            image_template=ImageSaveConfig(),
            audio_template=AudioSaveConfig(),
            mask_template=MaskSaveConfig(),
            save_debug_masks=True,
        )
        result = enabled_context.save_mask_archive(["frame"], "mask.zip")
        self.assertEqual(result, "mask.zip")
        mock_write.assert_called_once_with("mask.zip", ["frame"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
