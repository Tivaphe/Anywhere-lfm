import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.catalog import (
    classify_kind,
    entry_from_payload,
    is_allowed_model,
    is_excluded_repo,
    parse_model_ref,
    pick_gguf_file,
)
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
        self.assertTrue(is_gguf_model("LiquidAI/LFM2.5-2.6B-GGUF:LFM2.5-2.6B-Q4_K_M.gguf"))
        self.assertFalse(is_vl_model("LiquidAI/LFM2-VL-1.6B-GGUF"))


class CatalogFilterTests(unittest.TestCase):
    def test_allows_new_lfm_families_without_hardcoding(self):
        self.assertTrue(is_allowed_model("LiquidAI/LFM2.5-1.2B-Instruct"))
        self.assertTrue(is_allowed_model("LiquidAI/LFM3-8B-GGUF:Q4_K_M.gguf"))
        self.assertFalse(is_allowed_model("LiquidAI/LeapBundles"))
        self.assertFalse(is_allowed_model("openai/gpt-4"))

    def test_excludes_export_formats_and_bases(self):
        self.assertTrue(is_excluded_repo("LiquidAI/LFM2.5-2.6B-MLX", ["mlx"], "mlx"))
        self.assertTrue(is_excluded_repo("LiquidAI/LFM2.5-350M-Base"))
        self.assertTrue(is_excluded_repo("LiquidAI/LFM2.5-Encoders"))
        self.assertFalse(is_excluded_repo("LiquidAI/LFM2.5-2.6B-GGUF", ["gguf"], "gguf"))

    def test_classify_gguf_from_siblings(self):
        kind = classify_kind(
            "LiquidAI/LFM2.5-Mystery",
            tags=[],
            files=["weights/LFM2.5-Mystery-Q4_K_M.gguf"],
        )
        self.assertEqual(kind, "gguf")

    def test_pick_preferred_quant(self):
        files = [
            "LFM2.5-2.6B-F16.gguf",
            "LFM2.5-2.6B-Q8_0.gguf",
            "LFM2.5-2.6B-Q4_K_M.gguf",
            "LFM2.5-2.6B-Q5_K_M.gguf",
        ]
        self.assertEqual(pick_gguf_file(files), "LFM2.5-2.6B-Q4_K_M.gguf")
        self.assertEqual(pick_gguf_file(files, "Q8_0"), "LFM2.5-2.6B-Q8_0.gguf")

    def test_parse_ref(self):
        repo, spec = parse_model_ref("LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M")
        self.assertEqual(repo, "LiquidAI/LFM2.5-2.6B-GGUF")
        self.assertEqual(spec, "Q4_K_M")

    def test_entry_from_payload_skips_mlx(self):
        entry = entry_from_payload(
            {
                "id": "LiquidAI/LFM2.5-2.6B-MLX-8bit",
                "tags": ["mlx"],
                "library_name": "mlx",
            }
        )
        self.assertIsNone(entry)

    def test_entry_from_payload_keeps_gguf(self):
        entry = entry_from_payload(
            {
                "id": "LiquidAI/LFM2.5-2.6B-GGUF",
                "tags": ["gguf", "text-generation"],
                "library_name": "gguf",
                "siblings": [
                    {"rfilename": "LFM2.5-2.6B-Q4_K_M.gguf"},
                    {"rfilename": "README.md"},
                ],
            }
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.kind, "gguf")
        self.assertEqual(entry.gguf_files, ["LFM2.5-2.6B-Q4_K_M.gguf"])


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
