"""Text + official-rule classifier for the ``БАД`` category."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from src.rules import BAD_CATEGORY, bad_rule_features, normalize_product_text


def build_bad_texts(df: pd.DataFrame) -> list[str]:
    """Build normalized title + description strings without leaking labels."""

    names = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    descriptions = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    return [
        normalize_product_text(f"Название: {name}\nОписание: {description}")
        for name, description in zip(names, descriptions)
    ]


class BadRuleFeatureTransformer(BaseEstimator, TransformerMixin):
    """Expose deterministic rule matches as sparse numeric model features."""

    FEATURE_NAMES = (
        "direct_bad_marker",
        "explicit_not_bad",
        "sports_nutrition",
        "missing_bad_marker",
        "rule_score",
    )

    def fit(self, X: Iterable[str], y=None):  # noqa: N803 - sklearn convention
        return self

    def transform(self, X: Iterable[str]):  # noqa: N803 - sklearn convention
        values = np.asarray([bad_rule_features(text) for text in X], dtype=np.float32)
        return sparse.csr_matrix(values)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.FEATURE_NAMES, dtype=object)


class BadQualityClassifier:
    """Serializable classifier with an independently tuned decision threshold."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        random_state: int = 42,
        c: float = 2.0,
        max_word_features: int = 60_000,
        max_char_features: int = 80_000,
    ) -> None:
        self.threshold = float(threshold)
        self.random_state = int(random_state)
        self.c = float(c)
        self.max_word_features = int(max_word_features)
        self.max_char_features = int(max_char_features)
        self.features = self._make_features()
        self.model = LogisticRegression(
            C=self.c,
            class_weight="balanced",
            max_iter=1_000,
            random_state=self.random_state,
            solver="liblinear",
        )

    def _make_features(self) -> FeatureUnion:
        return FeatureUnion(
            (
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.995,
                        max_features=self.max_word_features,
                        sublinear_tf=True,
                        strip_accents=None,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=self.max_char_features,
                        sublinear_tf=True,
                    ),
                ),
                ("rules", BadRuleFeatureTransformer()),
            )
        )

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "BadQualityClassifier":
        text_list = list(texts)
        y = np.asarray(list(labels), dtype=np.int8)
        matrix = self.features.fit_transform(text_list, y)
        self.model.fit(matrix, y)
        return self

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        matrix = self.features.transform(list(texts))
        return self.model.predict_proba(matrix)[:, 1]

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(np.int8)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)

    @staticmethod
    def load(path: str | Path) -> "BadQualityClassifier":
        model = joblib.load(path)
        if not isinstance(model, BadQualityClassifier):
            raise TypeError(f"unexpected БАД classifier type: {type(model)!r}")
        return model


def predict_bad_rows(model: BadQualityClassifier, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict only rows in the official ``БАД`` category."""

    mask = df["category"].astype(str).eq(BAD_CATEGORY).to_numpy()
    texts = build_bad_texts(df.loc[mask])
    probabilities = model.predict_proba(texts)
    predictions = (probabilities >= model.threshold).astype(np.int8)
    return probabilities, predictions

