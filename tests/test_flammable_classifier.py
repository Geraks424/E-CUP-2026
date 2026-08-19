from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from src.flammable_classifier import (  # noqa: E402
    FlammableQualityClassifier,
    flammable_row_mask,
    predict_flammable_rows,
)
from src.rules import FLAMMABLE_CATEGORY  # noqa: E402


class FlammableClassifierTest(unittest.TestCase):
    def test_fit_predict_and_serialization(self):
        texts = [
            "спички длительного горения",
            "зажигалка карманная",
            "газовый баллон 650 г",
            "газовая плита без баллона",
            "мангал складной",
            "фильтр с активированным углем",
            "фонарик со встроенной зажигалкой",
            "туристическая газовая плита",
        ]
        labels = [1, 1, 1, 0, 0, 0, 0, 0]
        embeddings = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)
        model = FlammableQualityClassifier(random_state=7, c=1.0).fit(embeddings, texts, labels)

        probabilities = model.predict_proba(embeddings, texts)
        self.assertEqual(probabilities.shape, (8,))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            loaded = FlammableQualityClassifier.load(path)
            np.testing.assert_allclose(loaded.predict_proba(embeddings, texts), probabilities)

    def test_row_masking_without_cuda(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["a", "b", "c"],
                "description": ["d", "e", "f"],
                "category": ["БАД", FLAMMABLE_CATEGORY, "БАД"],
                "label": [1, 1, 0],
            }
        )
        mask = flammable_row_mask(df)
        self.assertEqual(mask.tolist(), [False, True, False])

        texts = [
            "спички",
            "зажигалка",
            "бад",
            "газовый баллон",
            "мангал",
            "уголь",
            "фонарик",
            "плита",
        ]
        labels = [1, 1, 1, 0, 0, 0, 0, 0]
        embeddings = np.random.default_rng(1).normal(size=(8, 8)).astype(np.float32)
        model = FlammableQualityClassifier(threshold=0.5, random_state=3).fit(embeddings, texts, labels)

        flame_df = df.loc[mask].copy()
        flame_embeddings = np.vstack([embeddings[1]])
        probs, preds = predict_flammable_rows(model, flame_df, flame_embeddings)
        self.assertEqual(probs.shape, (1,))
        self.assertEqual(preds.shape, (1,))


if __name__ == "__main__":
    unittest.main()
