import tempfile
import unittest
from pathlib import Path

from scripts.sc117_model import normalize_tier, sc117_model_paths, _context_tokens


class SC117ModelTests(unittest.TestCase):
    def test_normalize_supported_tiers(self):
        for tier in ("Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"):
            self.assertEqual(normalize_tier(tier.lower()), tier)

    def test_reject_unknown_tier(self):
        with self.assertRaises(ValueError):
            normalize_tier("Q8_0")

    def test_paths_are_namespaced_for_sc117(self):
        with tempfile.TemporaryDirectory() as td:
            paths = sc117_model_paths(Path(td), "Q2_0")
            self.assertEqual(paths["model_dir"], Path(td) / "models" / "sc117" / "Q2_0")
            self.assertEqual(paths["pack_dir"], Path(td) / "packs" / "sc117_q2_0")
            self.assertEqual(paths["mtp_dir"], Path(td) / "mtp" / "rt")
            self.assertIn("abliterated-Q2_0-00001-of-00002.gguf", paths["shard1"].name)
            self.assertIn("abliterated-Q2_0-00002-of-00002.gguf", paths["shard2"].name)

    def test_context_parser(self):
        self.assertEqual(_context_tokens("32k"), 32768)
        self.assertEqual(_context_tokens("8K"), 8192)
        self.assertEqual(_context_tokens("4096"), 4096)


if __name__ == "__main__":
    unittest.main()
