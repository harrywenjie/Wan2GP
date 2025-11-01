import unittest
from typing import Any, Dict, Optional
from unittest.mock import patch

from cli.queue_controller import QueueController
from cli.queue_state import QueueStateTracker
from cli.queue_utils import PreviewImages


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
        prompt = inputs.get("prompt", "")
        return {
            "prompt": prompt,
            "enhanced_prompt": "\n".join(["enhanced prompt"]),
            "repeat_generation": 2,
            "video_length": 16,
            "num_inference_steps": 28,
            "adapter_payloads": {
                "prompt_enhancer": {
                    "enhanced_prompts": ["enhanced prompt"],
                    "context": {"seed": 42, "source_prompts": [prompt]},
                }
            },
        }


class _StubProductionManager:
    def __init__(self) -> None:
        self._task_inputs = _StubTaskInputs()
        self.queue_tracker = QueueStateTracker()

    def task_inputs(self) -> _StubTaskInputs:
        return self._task_inputs


class QueuePromptPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state: Dict[str, Any] = {"gen": {}}
        self.manager = _StubProductionManager()

    @patch("cli.queue_controller.get_preview_images")
    def test_enhanced_prompt_payload_propagates(self, mock_preview) -> None:
        mock_preview.return_value = PreviewImages(None, None, [], [])
        controller = QueueController(
            manager=self.manager,
            state=self.state,
        )

        params = {"prompt": "base prompt", "model_type": "t2v"}
        entry = controller.enqueue_task(params)

        self.assertIn("adapter_payloads", entry)
        payload = entry["adapter_payloads"]["prompt_enhancer"]
        self.assertEqual(payload["enhanced_prompts"], ["enhanced prompt"])

        metadata = entry["metadata"]
        self.assertEqual(metadata["enhanced_prompt"], "enhanced prompt")
        self.assertEqual(
            metadata["adapter_payloads"]["prompt_enhancer"]["enhanced_prompts"],
            ["enhanced prompt"],
        )

        queue_snapshot = self.manager.queue_tracker.snapshot()
        self.assertEqual(len(queue_snapshot), 1)
        self.assertEqual(
            queue_snapshot[0]["metadata"]["adapter_payloads"]["prompt_enhancer"]["enhanced_prompts"],
            ["enhanced prompt"],
        )
        self.assertEqual(queue_snapshot[0]["repeats"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
