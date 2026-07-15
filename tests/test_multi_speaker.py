import sys
import types
import unittest

import numpy as np
import torch


class _ProgressBar:
    def __init__(self, total):
        self.total = total

    def update_absolute(self, *_args, **_kwargs):
        return None


comfy_module = types.ModuleType("comfy")
comfy_utils_module = types.ModuleType("comfy.utils")
comfy_utils_module.ProgressBar = _ProgressBar
comfy_module.utils = comfy_utils_module
sys.modules.setdefault("comfy", comfy_module)
sys.modules.setdefault("comfy.utils", comfy_utils_module)

generate_module = types.ModuleType("nodes.generate")
generate_module._denoise_audio = lambda path: path
generate_module._safe_save_wav = lambda *_args, **_kwargs: None
generate_module._sanitize_control = lambda text: " ".join(str(text or "").split())


def _quality_guard(model, kwargs, **_context):
    return model.generate(**kwargs)


generate_module._generate_with_quality_guard = _quality_guard
sys.modules.setdefault("nodes.generate", generate_module)

from nodes.multi_speaker import (  # noqa: E402
    RunningHubVoxCPMMultiSpeaker,
    RunningHubVoxCPMMultiSpeakerListReference,
)


class _FakeVoxCPM:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.full(48000, 0.05, dtype=np.float32)


class MultiSpeakerRegressionTests(unittest.TestCase):
    def _model(self):
        engine = _FakeVoxCPM()
        return engine, {
            "model": engine,
            "sample_rate": 48000,
            "architecture": "voxcpm2",
        }

    def test_generates_each_dialogue_segment_independently(self):
        engine, model = self._model()
        node = RunningHubVoxCPMMultiSpeaker()
        result = node.generate(
            model,
            "[spk1]第一句[spk2]第二句[spk1]第三句",
            2.0,
            10,
            0,
        )
        self.assertEqual([call["text"] for call in engine.calls], ["第一句", "第二句", "第三句"])
        self.assertEqual(result[0]["waveform"].shape[-1], 3 * 48000)

    def test_v2_reference_without_manual_text_never_uses_prompt_text(self):
        engine, model = self._model()
        node = RunningHubVoxCPMMultiSpeaker()
        reference_audio = {
            "waveform": torch.zeros((1, 1, 160), dtype=torch.float32),
            "sample_rate": 16000,
        }
        node.generate(
            model,
            "[spk1]安全参考模式",
            2.0,
            10,
            0,
            audio_1=reference_audio,
            control_1="",
        )
        self.assertIn("reference_wav_path", engine.calls[0])
        self.assertNotIn("prompt_text", engine.calls[0])
        self.assertNotIn("prompt_wav_path", engine.calls[0])

    def test_new_transcript_widgets_are_appended_for_compatibility(self):
        fixed_optional = list(
            RunningHubVoxCPMMultiSpeaker.INPUT_TYPES()["optional"]
        )
        self.assertGreater(
            fixed_optional.index("reference_text_1"),
            fixed_optional.index("retry_badcase"),
        )

        dynamic_optional = list(
            RunningHubVoxCPMMultiSpeakerListReference.INPUT_TYPES()["optional"]
        )
        self.assertGreater(
            dynamic_optional.index("reference_texts"),
            dynamic_optional.index("retry_badcase"),
        )


if __name__ == "__main__":
    unittest.main()
