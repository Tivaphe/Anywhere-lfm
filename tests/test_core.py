import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import DEFAULT_SETTINGS, load_settings, save_settings
from core.generate import as_text_messages, build_sampling_kwargs
from core.models import is_gguf_model, is_vl_model


class SamplingKwargsTests(unittest.TestCase):
    def test_official_liquidai_defaults_use_min_p_not_top_p(self):
        kwargs = build_sampling_kwargs(DEFAULT_SETTINGS)
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["min_p"], 0.15)
        self.assertEqual(kwargs["repetition_penalty"], 1.05)
        self.assertNotIn("top_p", kwargs)

    def test_zero_temperature_disables_sampling(self):
        kwargs = build_sampling_kwargs({"temperature": 0, "min_p": 0.15})
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("min_p", kwargs)
        self.assertNotIn("temperature", kwargs)


class MessageNormalizationTests(unittest.TestCase):
    def test_multimodal_parts_become_text(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Que vois-tu ?"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            }
        ]
        normalized = as_text_messages(history)
        self.assertEqual(normalized[0]["content"], "Que vois-tu ?")


class ModelKindTests(unittest.TestCase):
    def test_kind_detection(self):
        self.assertTrue(is_vl_model("LiquidAI/LFM2-VL-1.6B"))
        self.assertFalse(is_vl_model("LiquidAI/LFM2-1.2B"))
        self.assertTrue(is_gguf_model("LiquidAI/LFM2-1.2B-GGUF"))
        self.assertFalse(is_gguf_model("LiquidAI/LFM2-1.2B"))


class SettingsPersistenceTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            save_settings({"temperature": 0.5, "min_p": 0.2}, path)
            loaded = load_settings(path)
            self.assertEqual(loaded["temperature"], 0.5)
            self.assertEqual(loaded["min_p"], 0.2)
            self.assertEqual(loaded["system_prompt"], DEFAULT_SETTINGS["system_prompt"])
            with open(path, encoding="utf-8") as handle:
                json.load(handle)


if __name__ == "__main__":
    unittest.main()
