import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodes.validation import (
    GeneratedAudioQualityError,
    resolve_allowed_file,
    safe_child_directory,
    safe_name_component,
    validate_generated_audio,
    validate_path_component,
)


class PathValidationTests(unittest.TestCase):
    def test_output_name_rejects_traversal_and_absolute_paths(self):
        for value in ("../escape", "nested/name", "/tmp/escape", "..\\escape"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_name_component(value, "default")

    def test_existing_name_preserves_spaces_but_rejects_separators(self):
        self.assertEqual(validate_path_component("voice one.safetensors"), "voice one.safetensors")
        with self.assertRaises(ValueError):
            validate_path_component("../../voice.safetensors")

    def test_safe_child_stays_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = safe_child_directory(tmp, "dataset one", "dataset")
            self.assertEqual(child.parent, Path(tmp).resolve())
            self.assertEqual(child.name, "dataset_one")

    def test_allowed_file_must_remain_under_configured_root(self):
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            good = Path(allowed) / "train.jsonl"
            bad = Path(outside) / "train.jsonl"
            good.write_text("{}\n", encoding="utf-8")
            bad.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                resolve_allowed_file(good, [allowed], suffixes=(".jsonl",)),
                good.resolve(),
            )
            with self.assertRaises(ValueError):
                resolve_allowed_file(bad, [allowed], suffixes=(".jsonl",))


class GeneratedAudioValidationTests(unittest.TestCase):
    SAMPLE_RATE = 48000

    @staticmethod
    def _tone(seconds, sample_rate=48000):
        samples = int(seconds * sample_rate)
        t = np.arange(samples, dtype=np.float32) / sample_rate
        return 0.05 * np.sin(2 * np.pi * 220 * t)

    def test_rejects_task_regression_short_audio(self):
        # Task 2077335226551599105 returned 13.12s for 184 CJK characters.
        text = "这是用于验证异常短音频检测的文本" * 12
        self.assertGreaterEqual(len(text), 184)
        with self.assertRaises(GeneratedAudioQualityError):
            validate_generated_audio(
                self._tone(13.12), self.SAMPLE_RATE, text[:184]
            )

    def test_accepts_plausible_duration(self):
        text = "这是一段正常语速的中文语音测试文本" * 5
        wav = self._tone(max(2.0, len(text) / 5.0))
        result = validate_generated_audio(wav, self.SAMPLE_RATE, text)
        self.assertEqual(result.dtype, np.float32)

    def test_rejects_silence_and_non_finite_samples(self):
        with self.assertRaises(GeneratedAudioQualityError):
            validate_generated_audio(np.zeros(48000), self.SAMPLE_RATE, "测试")
        with self.assertRaises(GeneratedAudioQualityError):
            validate_generated_audio(
                np.array([0.1, np.nan], dtype=np.float32),
                self.SAMPLE_RATE,
                "test",
            )


if __name__ == "__main__":
    unittest.main()
