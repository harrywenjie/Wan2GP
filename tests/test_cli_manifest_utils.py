import hashlib
import json
import unittest
from pathlib import Path

from cli.generate import _build_manifest_artifacts
from cli.manifest import ArtifactCapture, canonicalize_structure, compute_adapter_hashes
from core.io.media import AudioSaveConfig, VideoSaveConfig


class ManifestAssemblyTests(unittest.TestCase):
    def test_build_manifest_artifacts_merges_captures_and_settings(self) -> None:
        video_config = VideoSaveConfig(fps=24, codec_type="libx264_8", container="mp4")
        audio_config = AudioSaveConfig(sample_rate=44100, subtype="PCM_16", format="wav")
        captures = [
            ArtifactCapture(path=Path("/tmp/output.mp4"), kind="video", config=video_config),
            ArtifactCapture(path=Path("/tmp/audio.wav"), kind="audio", config=audio_config),
            ArtifactCapture(path=Path("/tmp/output_mask.zip"), kind="mask_archive", config=None),
        ]

        artifacts = _build_manifest_artifacts(
            video_paths=["/tmp/output.mp4"],
            video_settings=[{"video_length": 48, "metadata_mode": "json"}],
            audio_paths=["/tmp/audio.wav"],
            audio_settings=[{"duration_s": "2.5", "metadata_mode": "metadata"}],
            captures=captures,
        )

        self.assertEqual(len(artifacts), 3)

        video_entry = artifacts[0]
        self.assertEqual(video_entry["role"], "foreground")
        self.assertEqual(video_entry["container"], "mp4")
        self.assertEqual(video_entry["codec"], "libx264_8")
        self.assertEqual(video_entry["frames"], 48)
        self.assertAlmostEqual(video_entry["duration_s"], 2.0)
        self.assertTrue(video_entry["metadata_sidecar"].endswith("output.json"))

        audio_entry = artifacts[1]
        self.assertEqual(audio_entry["role"], "audio")
        self.assertEqual(audio_entry["container"], "wav")
        self.assertEqual(audio_entry["codec"], "PCM_16")
        self.assertEqual(audio_entry["duration_s"], 2.5)
        self.assertIsNone(audio_entry["metadata_sidecar"])

        mask_entry = artifacts[2]
        self.assertEqual(mask_entry["role"], "mask_archive")
        self.assertTrue(mask_entry["path"].endswith("output_mask.zip"))

    def test_compute_adapter_hashes_canonicalises_payloads(self) -> None:
        payload = {"path": Path("/weights/model.safetensors"), "weights": [1, 2, 3]}
        hashes = compute_adapter_hashes({"lora": payload})

        canonical = canonicalize_structure(payload)
        serialised = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        expected_digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()

        self.assertIn("lora", hashes)
        entry = hashes["lora"]
        self.assertEqual(entry["sha256"], expected_digest)
        self.assertEqual(entry["source_bytes"], len(serialised))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
