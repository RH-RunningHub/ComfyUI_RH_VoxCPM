import sys
import types
import unittest

import numpy as np


class _ProgressBar:
    def __init__(self, total):
        self.total = total

    def update(self, *_args, **_kwargs):
        return None

    def update_absolute(self, *_args, **_kwargs):
        return None


comfy_module = types.ModuleType("comfy")
comfy_utils_module = types.ModuleType("comfy.utils")
comfy_utils_module.ProgressBar = _ProgressBar
comfy_module.utils = comfy_utils_module
sys.modules.setdefault("comfy", comfy_module)
sys.modules.setdefault("comfy.utils", comfy_utils_module)

folder_paths_module = types.ModuleType("folder_paths")
folder_paths_module.models_dir = "/tmp/models"
folder_paths_module.add_model_folder_path = lambda *_args, **_kwargs: None
folder_paths_module.get_folder_paths = lambda *_args, **_kwargs: []
sys.modules.setdefault("folder_paths", folder_paths_module)

torchaudio_module = types.ModuleType("torchaudio")
torchaudio_module.save = lambda *_args, **_kwargs: None
sys.modules.setdefault("torchaudio", torchaudio_module)

soundfile_module = types.ModuleType("soundfile")
soundfile_module.write = lambda *_args, **_kwargs: None
sys.modules.setdefault("soundfile", soundfile_module)

from nodes.generate import (  # noqa: E402
    RunningHubVoxCPMGenerate,
    _generate_with_quality_guard,
)


class _RetryModel:
    def __init__(self):
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        seconds = 1.0 if self.calls == 1 else 3.0
        return np.full(int(48000 * seconds), 0.05, dtype=np.float32)


class GenerateRegressionTests(unittest.TestCase):
    def test_quality_guard_retries_premature_eos(self):
        model = _RetryModel()
        result = _generate_with_quality_guard(
            model,
            {"text": "这是二十个中文字用于测试短音频重试"},
            target_text="这是二十个中文字用于测试短音频重试",
            sample_rate=48000,
            retry_badcase=True,
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(result.size, 3 * 48000)

    def test_ultimate_clone_requires_exact_manual_transcript(self):
        node = RunningHubVoxCPMGenerate()
        model = {
            "model": _RetryModel(),
            "sample_rate": 48000,
            "architecture": "voxcpm2",
        }
        with self.assertRaisesRegex(ValueError, "reference_audio_text"):
            node.generate(
                model,
                "",
                "target text",
                2.0,
                10,
                0,
                reference_audio={"waveform": None, "sample_rate": 16000},
                ultimate_clone=True,
                reference_audio_text="",
            )


if __name__ == "__main__":
    unittest.main()
