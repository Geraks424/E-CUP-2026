from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baseline" / "quality-baseline-submit"
for path in (REPO_ROOT, BASELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.pack_quality_submit import ALLOWLIST, pack_submit  # noqa: E402
from scripts.validate_quality_submission import validate_submission  # noqa: E402


class PackSubmitTest(unittest.TestCase):
    def test_allowlist_excludes_embed_cache_and_includes_metadata_root(self):
        self.assertNotIn("src/embed_cache.py", ALLOWLIST)
        self.assertIn("metadata.json", ALLOWLIST)
        self.assertIn("src/__init__.py", ALLOWLIST)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "submit.zip"
            size, digest = pack_submit(BASELINE_ROOT, out)
            self.assertGreater(size, 0)
            self.assertEqual(len(digest), 64)
            names = zipfile.ZipFile(out).namelist()
            self.assertEqual(sorted(names), sorted(ALLOWLIST))
            self.assertNotIn("src/embed_cache.py", names)
            self.assertIn("metadata.json", names)


class RunPipelineGlueTest(unittest.TestCase):
    def test_rules_mode_writes_valid_csv_without_llm(self):
        run_mod = importlib.import_module("run")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            (images / "1").mkdir(parents=True)
            (images / "2").mkdir()
            input_csv = root / "input.csv"
            output_csv = root / "submit.csv"
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "name": ["БАД комплекс", "Газовая горелка"],
                    "description": [
                        "Биологически активная добавка, 60 капсул",
                        "Горелка туристическая без баллона",
                    ],
                    "category": ["БАД", "Легковоспламеняющиеся"],
                }
            ).to_csv(input_csv, index=False)

            dummy_logreg = MagicMock()
            dummy_logreg.predict.return_value = (
                np.array([0.9, 0.1]),
                np.array([1, 0]),
            )
            dummy_bad = MagicMock()
            dummy_flame = MagicMock()

            argv = [
                "run.py",
                "--test_data_path",
                str(input_csv),
                "--output_path",
                str(output_csv),
                "--comments_mode",
                "rules",
                "--embed_batch",
                "1",
                "--llm_batch",
                "1",
                "--pixel_preset",
                "S",
            ]
            embeddings = np.zeros((2, 8), dtype=np.float32)
            with (
                patch.object(sys, "argv", argv),
                patch.object(run_mod, "embed_data_cuda", return_value=embeddings) as embed,
                patch.object(run_mod, "generate_comments_cuda") as llm,
                patch.object(run_mod.ProductQualityPredictor, "load", return_value=dummy_logreg),
                patch.object(run_mod.BadQualityClassifier, "load", return_value=dummy_bad),
                patch.object(run_mod.FlammableQualityClassifier, "load", return_value=dummy_flame),
                patch.object(
                    run_mod,
                    "predict_bad_rows",
                    return_value=(np.array([0.8]), np.array([1])),
                ),
                patch.object(
                    run_mod,
                    "predict_flammable_rows",
                    return_value=(np.array([0.2]), np.array([0])),
                ),
            ):
                run_mod.main()

            embed.assert_called_once()
            llm.assert_not_called()
            self.assertTrue(output_csv.is_file())
            errors = validate_submission(input_csv, output_csv)
            self.assertEqual(errors, [])
            out = pd.read_csv(output_csv)
            self.assertEqual(list(out.columns), ["id", "result"])
            self.assertTrue(str(out.loc[out["id"] == 1, "result"].iloc[0]).endswith("<вердикт>не бан"))
            self.assertTrue(str(out.loc[out["id"] == 2, "result"].iloc[0]).endswith("<вердикт>бан"))


if __name__ == "__main__":
    unittest.main()
