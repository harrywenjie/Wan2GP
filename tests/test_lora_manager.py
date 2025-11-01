from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Sequence

from core.lora.manager import LoRAInjectionManager


class StubWgp:
    def __init__(self, lora_dir: Path) -> None:
        self.lora_dir = lora_dir
        self.server_config = {"enhancer_enabled": 0, "lora_dir": str(lora_dir)}
        self.setup_calls: List[tuple[Any, str]] = []
        self.extract_calls: List[Sequence[str]] = []

    def get_lora_dir(self, model_type: str) -> str:
        return str(self.lora_dir / model_type)

    def setup_loras(self, model_type, transformer, lora_dir, preselected_preset, split_map):
        self.setup_calls.append((model_type, preselected_preset))
        base_dir = Path(self.get_lora_dir(model_type))
        base_dir.mkdir(parents=True, exist_ok=True)
        files = ["alpha.safetensors", "beta.safetensors"]
        presets = ["cinematic.lset", "noir.json"]
        return files, presets, ["alpha.safetensors"], "1.0", "", preselected_preset

    def extract_preset(self, model_type, preset_name, available_loras):
        self.extract_calls.append((model_type, preset_name, tuple(available_loras)))
        return ["alpha.safetensors"], "0.5,1.0", "night city", True, ""


class LoRAManagerTests(unittest.TestCase):
    def _build_manager(self) -> tuple[LoRAInjectionManager, Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        manager = LoRAInjectionManager(StubWgp(root))
        return manager, root

    def test_hydrate_caches_results(self) -> None:
        manager, _ = self._build_manager()
        hydration_first = manager.hydrate("wan")
        hydration_second = manager.hydrate("wan")

        self.assertEqual(hydration_first.library.loras, hydration_second.library.loras)
        self.assertEqual(manager._wgp.setup_calls, [("wan", "")])

    def test_hydrate_respects_refresh(self) -> None:
        manager, _ = self._build_manager()
        manager.hydrate("wan")
        manager.hydrate("wan", refresh=True)

        self.assertEqual(manager._wgp.setup_calls, [("wan", ""), ("wan", "")])

    def test_cache_invalidated_on_server_config_change(self) -> None:
        manager, _ = self._build_manager()
        manager.hydrate("wan")
        manager._wgp.server_config["new_flag"] = 1
        manager.hydrate("wan")

        self.assertEqual(manager._wgp.setup_calls, [("wan", ""), ("wan", "")])

    def test_resolve_preset_lset(self) -> None:
        manager, _ = self._build_manager()
        manager.hydrate("wan")
        result = manager.resolve_preset("wan", "cinematic.lset")

        self.assertEqual(result.names, ("alpha.safetensors",))
        self.assertEqual(result.multipliers, "0.5,1.0")
        self.assertEqual(result.prompt, "night city")
        self.assertTrue(result.prompt_is_full)
        self.assertTrue(manager._wgp.extract_calls)

    def test_resolve_preset_json(self) -> None:
        manager, _ = self._build_manager()
        hydration = manager.hydrate("wan")
        library_dir = hydration.library.lora_dir
        if library_dir is None:
            self.fail("expected LoRA directory to be discovered")
        preset_path = library_dir / "custom.json"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        preset_path.write_text(
            '{"activated_loras": ["alpha.safetensors"], "loras_multipliers": "0.1", "prompt": "city", "full_prompt": false}',
            encoding="utf-8",
        )

        result = manager.resolve_preset("wan", "custom.json")

        self.assertEqual(result.names, ("alpha.safetensors",))
        self.assertEqual(result.multipliers, "0.1")
        self.assertEqual(result.prompt, "city")
        self.assertFalse(result.prompt_is_full)


if __name__ == "__main__":
    unittest.main()
