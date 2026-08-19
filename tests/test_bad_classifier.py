from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from src.bad_classifier import BadQualityClassifier  # noqa: E402


class BadClassifierTest(unittest.TestCase):
    def test_fit_predict_and_serialization(self):
        texts = [
            "бад биологически активная добавка капсулы",
            "dietary supplement капсулы",
            "прямая маркировка бад таблетки",
            "биологически активная добавка к пище",
            "спортивное питание протеин",
            "аминокислоты bcaa для спорта",
            "таблетница без маркировки",
            "товар не является бад",
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        model = BadQualityClassifier(
            max_word_features=200,
            max_char_features=300,
            random_state=7,
        ).fit(texts, labels)

        probabilities = model.predict_proba(texts)
        self.assertEqual(probabilities.shape, (8,))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            loaded = BadQualityClassifier.load(path)
            np.testing.assert_allclose(loaded.predict_proba(texts), probabilities)


if __name__ == "__main__":
    unittest.main()

