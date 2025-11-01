from __future__ import annotations

import unittest
from typing import Any, Dict, List, Sequence

from core.prompt_enhancer.bridge import (
    PromptEnhancerBridge,
    PromptEnhancerContext,
    PromptEnhancerSpec,
)


class StubWgp:
    def __init__(self) -> None:
        self.server_config: Dict[str, Any] = {"enhancer_enabled": 1}
        self.setup_calls: List[Dict[str, Any]] = []
        self.reset_calls = 0
        self.process_calls: List[Dict[str, Any]] = []

    def setup_prompt_enhancer(self, pipe, kwargs):
        self.setup_calls.append({"pipe": dict(pipe), "kwargs": dict(kwargs)})

    def reset_prompt_enhancer(self):
        self.reset_calls += 1

    def process_prompt_enhancer(
        self,
        prompt_enhancer: str,
        prompts: Sequence[str],
        image_start,
        image_refs,
        is_image: bool,
        audio_only: bool,
        seed: int,
    ):
        self.process_calls.append(
            {
                "mode": prompt_enhancer,
                "prompts": list(prompts),
                "image_start": image_start,
                "image_refs": image_refs,
                "is_image": is_image,
                "audio_only": audio_only,
                "seed": seed,
            }
        )
        return ["enhanced prompt"]


class PromptEnhancerBridgeTests(unittest.TestCase):
    def test_prime_caches_per_server_config(self) -> None:
        stub = StubWgp()
        bridge = PromptEnhancerBridge(stub)
        spec = PromptEnhancerSpec(prompt_enhancer="IT", pipe={}, kwargs={})

        bridge.prime(spec)
        bridge.prime(spec)

        self.assertEqual(len(stub.setup_calls), 1)
        stub.server_config["enhancer_mode"] = 2
        bridge.prime(spec)
        self.assertEqual(len(stub.setup_calls), 2)

    def test_reset_triggers_underlying_and_clears_state(self) -> None:
        stub = StubWgp()
        bridge = PromptEnhancerBridge(stub)
        spec = PromptEnhancerSpec(prompt_enhancer="IT", pipe={}, kwargs={})
        bridge.prime(spec)

        bridge.reset()

        self.assertEqual(stub.reset_calls, 1)
        self.assertEqual(bridge.snapshot_state()["primed_configs"], [])

    def test_enhance_delegates_to_wgp(self) -> None:
        stub = StubWgp()
        bridge = PromptEnhancerBridge(stub)
        context = PromptEnhancerContext(
            prompts=["city skyline"],
            image_start=("img-1",),
            image_refs=("img-2",),
            is_image=False,
            audio_only=False,
            seed=123,
        )

        result = bridge.enhance("IT", context)

        self.assertEqual(result, ["enhanced prompt"])
        self.assertTrue(stub.process_calls)


if __name__ == "__main__":
    unittest.main()
