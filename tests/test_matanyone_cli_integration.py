import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import TestCase, mock

from cli import matanyone
from core.io.media import AudioSaveConfig, ImageSaveConfig, MaskSaveConfig, MediaPersistenceContext, VideoSaveConfig
from core.production_manager import MetadataState
from preprocessing.matanyone.app import MatAnyOneRequest, MatAnyOneResult


class StubMediaContext(MediaPersistenceContext):
    """Test double that records audio saves without touching codecs."""

    def __init__(self) -> None:
        super().__init__(
            video_template=VideoSaveConfig(container="mp4"),
            image_template=ImageSaveConfig(),
            audio_template=AudioSaveConfig(sample_rate=16000, format="wav", subtype="PCM_16"),
            mask_template=MaskSaveConfig(),
        )
        self.saved_audio: List[str] = []

    def save_video(
        self,
        data: Any,
        target_path: str,
        *,
        logger: Any = None,
        config: Any = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        return target_path

    def save_audio(
        self,
        data: Any,
        target_path: str,
        *,
        sample_rate: Any = None,
        logger: Any = None,
        config: Any = None,
    ) -> str:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(b"stub")
        self.saved_audio.append(target_path)
        return target_path

    def save_mask_archive(
        self,
        frames: Any,
        target_path: str,
        *,
        logger: Any = None,
        config: Any = None,
        overrides: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> str:
        return target_path


class MatAnyOneCliIntegrationTests(TestCase):
    def test_manifest_emission_includes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / "input.mp4"
            mask_path = work_dir / "mask.png"
            manifest_path = work_dir / "manifest.jsonl"
            output_dir = work_dir / "outputs"

            input_path.write_bytes(b"\x00")
            mask_path.write_bytes(b"\x00")

            result_metadata: Dict[str, object] = {"codec": "libx264_8", "container": "mp4"}
            stub_result = MatAnyOneResult(
                foreground_path=output_dir / "mask_foreground.mp4",
                alpha_path=output_dir / "mask_alpha.mp4",
                rgba_zip_path=output_dir / "mask_rgba.zip",
                frames_processed=12,
                fps=6.0,
                metadata=result_metadata,
            )

            stub_result.foreground_path.parent.mkdir(parents=True, exist_ok=True)
            stub_result.foreground_path.write_bytes(b"\x00")
            stub_result.alpha_path.write_bytes(b"\x00")
            if stub_result.rgba_zip_path is not None:
                stub_result.rgba_zip_path.write_bytes(b"\x00")

            metadata_state = MetadataState(choice="json", configs={})

            with mock.patch(
                "cli.matanyone._resolve_runtime_contexts",
                return_value=(metadata_state, None),
            ) as resolve_mock, mock.patch(
                "cli.matanyone.generate_masks",
                return_value=stub_result,
            ) as generate_mock:
                exit_code = matanyone.main(
                    [
                        "--input",
                        str(input_path),
                        "--template-mask",
                        str(mask_path),
                        "--output-dir",
                        str(output_dir),
                        "--manifest-path",
                        str(manifest_path),
                        "--codec",
                        "libx264_8",
                        "--metadata-mode",
                        "json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            resolve_mock.assert_called_once()
            generate_mock.assert_called_once()

            manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(manifest_lines), 1)

            entry = json.loads(manifest_lines[0])
            self.assertEqual(entry["status"], "success")
            self.assertEqual(entry["metadata_mode"], "json")
            self.assertEqual(entry["output_dir"], str(output_dir.resolve()))

            artifacts = entry["artifacts"]
            roles = {artifact["role"] for artifact in artifacts}
            self.assertSetEqual(roles, {"mask_foreground", "mask_alpha", "rgba_archive"})

            foreground = next(artifact for artifact in artifacts if artifact["role"] == "mask_foreground")
            alpha = next(artifact for artifact in artifacts if artifact["role"] == "mask_alpha")
            rgba = next(artifact for artifact in artifacts if artifact["role"] == "rgba_archive")

            self.assertEqual(foreground["path"], str(stub_result.foreground_path.resolve()))
            self.assertEqual(foreground["container"], "mp4")
            self.assertEqual(foreground["codec"], "libx264_8")
            self.assertAlmostEqual(foreground["duration_s"], 2.0)
            self.assertEqual(
                foreground["metadata_sidecar"],
                str(stub_result.foreground_path.with_suffix(".json").resolve()),
            )

            self.assertEqual(alpha["path"], str(stub_result.alpha_path.resolve()))
            self.assertEqual(alpha["container"], "mp4")
            self.assertEqual(alpha["metadata_sidecar"], str(stub_result.alpha_path.with_suffix(".json").resolve()))

            self.assertEqual(rgba["path"], str(stub_result.rgba_zip_path.resolve()))
            self.assertEqual(rgba["container"], "zip")
            self.assertIsNone(rgba["metadata_sidecar"])

    def test_manifest_emission_includes_audio_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            input_path = work_dir / "input.mp4"
            mask_path = work_dir / "mask.png"
            manifest_path = work_dir / "manifest.jsonl"
            output_dir = work_dir / "outputs"

            input_path.write_bytes(b"\x00")
            mask_path.write_bytes(b"\x00")

            result_metadata: Dict[str, object] = {"codec": "libx264_8", "container": "mp4", "attach_audio": True}
            stub_result = MatAnyOneResult(
                foreground_path=output_dir / "mask_foreground.mp4",
                alpha_path=output_dir / "mask_alpha.mp4",
                rgba_zip_path=output_dir / "mask_rgba.zip",
                frames_processed=8,
                fps=4.0,
                metadata=result_metadata,
            )

            stub_result.foreground_path.parent.mkdir(parents=True, exist_ok=True)
            stub_result.foreground_path.write_bytes(b"\x00")
            stub_result.alpha_path.write_bytes(b"\x00")
            if stub_result.rgba_zip_path is not None:
                stub_result.rgba_zip_path.write_bytes(b"\x00")

            metadata_state = MetadataState(choice="json", configs={})
            context = StubMediaContext()
            audio_output = output_dir / "stub_audio.wav"

            def fake_generate(state: Dict[str, object], request: MatAnyOneRequest) -> MatAnyOneResult:
                if request.media_context is not None:
                    request.media_context.save_audio(
                        [0.0, 0.0],
                        str(audio_output),
                        overrides={"sample_rate": 16000},
                    )
                return stub_result

            with mock.patch(
                "cli.matanyone._resolve_runtime_contexts",
                return_value=(metadata_state, context),
            ), mock.patch(
                "cli.matanyone.generate_masks",
                side_effect=fake_generate,
            ):
                exit_code = matanyone.main(
                    [
                        "--input",
                        str(input_path),
                        "--template-mask",
                        str(mask_path),
                        "--output-dir",
                        str(output_dir),
                        "--manifest-path",
                        str(manifest_path),
                        "--codec",
                        "libx264_8",
                        "--metadata-mode",
                        "json",
                    ]
                )

            self.assertEqual(exit_code, 0)

            manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(manifest_lines), 1)

            entry = json.loads(manifest_lines[0])
            roles = {artifact["role"] for artifact in entry["artifacts"]}
            self.assertIn("audio", roles)

            audio_entry = next(artifact for artifact in entry["artifacts"] if artifact["role"] == "audio")
            self.assertEqual(audio_entry["path"], str(audio_output.resolve()))
            self.assertEqual(audio_entry["container"], "wav")
            self.assertEqual(audio_entry["codec"], "PCM_16")
            self.assertEqual(
                audio_entry["metadata_sidecar"],
                str(audio_output.with_suffix(".json").resolve()),
            )
