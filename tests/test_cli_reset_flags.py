import unittest

from cli.generate import reset_adapter_caches


class _Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls += 1


class _StubManager:
    def __init__(self) -> None:
        self._lora = _Recorder()
        self._prompt = _Recorder()

    def lora_manager(self) -> _Recorder:
        return self._lora

    def prompt_enhancer(self) -> _Recorder:
        return self._prompt


class _NullLogger:
    def info(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        pass


class ResetAdapterCachesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = _NullLogger()

    def test_no_resets_when_flags_disabled(self) -> None:
        manager = _StubManager()
        reset_adapter_caches(
            manager,
            reset_lora=False,
            reset_prompt_enhancer=False,
            logger=self.logger,
        )
        self.assertEqual(manager.lora_manager().calls, 0)
        self.assertEqual(manager.prompt_enhancer().calls, 0)

    def test_lora_reset_flag(self) -> None:
        manager = _StubManager()
        reset_adapter_caches(
            manager,
            reset_lora=True,
            reset_prompt_enhancer=False,
            logger=self.logger,
        )
        self.assertEqual(manager.lora_manager().calls, 1)
        self.assertEqual(manager.prompt_enhancer().calls, 0)

    def test_prompt_enhancer_reset_flag(self) -> None:
        manager = _StubManager()
        reset_adapter_caches(
            manager,
            reset_lora=False,
            reset_prompt_enhancer=True,
            logger=self.logger,
        )
        self.assertEqual(manager.lora_manager().calls, 0)
        self.assertEqual(manager.prompt_enhancer().calls, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
