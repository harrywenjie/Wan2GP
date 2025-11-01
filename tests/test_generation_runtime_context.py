import unittest

from core.production_manager import GenerationRuntime


class _StubWGP:
    def __init__(self, state):
        self.server_config = {}
        self.state = state

    def generate_video(self, task, send_cmd, **kwargs):
        media_context = kwargs.get("media_context")
        if media_context is None:
            raise RuntimeError("media_context missing")
        self.state["gen"].setdefault("file_list", []).append("artifact.mp4")


class GenerationRuntimeMediaContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {"gen": {"queue": [], "file_list": [], "audio_file_list": []}}
        self.wgp = _StubWGP(self.state)

    def test_run_raises_when_media_context_missing(self) -> None:
        runtime = GenerationRuntime(
            wgp=self.wgp,
            state=self.state,
        )
        params = {"prompt": "demo"}
        with self.assertRaisesRegex(RuntimeError, "media_context"):
            runtime.run(params, lambda *_args: None)

    def test_run_succeeds_with_media_context(self) -> None:
        runtime = GenerationRuntime(
            wgp=self.wgp,
            state=self.state,
            media_context=object(),
        )
        params = {"prompt": "demo"}
        outputs = runtime.run(params, lambda *_args: None)
        self.assertEqual(outputs, ["artifact.mp4"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
