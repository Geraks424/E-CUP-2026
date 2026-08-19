"""Multimodal embedding + official-rule classifier for flammable products."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from src.rules import FLAMMABLE_CATEGORY, combine_product_text, flammable_rule_features


def build_flammable_texts(df: pd.DataFrame) -> list[str]:
    """Build normalized name + description strings without leaking labels."""

    names = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
    descriptions = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
    return [
        combine_product_text(name, description)
        for name, description in zip(names, descriptions)
    ]


class FlammableRuleFeatureTransformer(BaseEstimator, TransformerMixin):
    """Expose deterministic flammable rule matches as sparse numeric features."""

    FEATURE_NAMES = (
        "absent_content_or_not_in_kit",
        "component_only",
        "built_in_source",
        "matches",
        "lighter",
        "gas_container",
        "fuel",
        "coal_or_wood",
        "explicit_flammable",
        "device",
        "built_in_ignition",
        "rule_score",
    )

    def fit(self, X: Iterable[str], y=None):  # noqa: N803 - sklearn convention
        return self

    def transform(self, X: Iterable[str]):  # noqa: N803 - sklearn convention
        values = np.asarray([flammable_rule_features(text) for text in X], dtype=np.float32)
        return sparse.csr_matrix(values)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.FEATURE_NAMES, dtype=object)


class EmbeddingArrayTransformer(BaseEstimator, TransformerMixin):
    """Pass precomputed dense embeddings through the sklearn feature union."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def fit(self, X=None, y=None):  # noqa: N803 - sklearn convention
        return self

    def transform(self, X=None):  # noqa: N803 - sklearn convention
        return sparse.csr_matrix(self.embeddings)


class FlammableQualityClassifier:
    """Serializable classifier with an independently tuned decision threshold."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        random_state: int = 42,
        c: float = 1.0,
        oversample_positives: bool = False,
    ) -> None:
        self.threshold = float(threshold)
        self.random_state = int(random_state)
        self.c = float(c)
        self.oversample_positives = bool(oversample_positives)
        self.model = LogisticRegression(
            C=self.c,
            class_weight="balanced",
            max_iter=1_000,
            random_state=self.random_state,
            solver="liblinear",
        )
        self._embedding_dim: int | None = None

    def _stack_features(self, embeddings: np.ndarray, texts: Sequence[str]) -> sparse.csr_matrix:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {emb.shape}")
        emb_matrix = sparse.csr_matrix(emb)
        rule_matrix = FlammableRuleFeatureTransformer().transform(texts)
        return sparse.hstack([emb_matrix, rule_matrix], format="csr")

    def _maybe_oversample(
        self,
        matrix: sparse.csr_matrix,
        labels: np.ndarray,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        if not self.oversample_positives:
            return matrix, labels
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels == 0)
        if positives.size == 0 or negatives.size == 0:
            return matrix, labels
        repeat_count = max(1, negatives.size // positives.size)
        pos_idx = np.tile(positives, repeat_count)
        combined_idx = np.concatenate([negatives, pos_idx])
        return matrix[combined_idx], labels[combined_idx]

    def fit(
        self,
        embeddings: np.ndarray,
        texts: Iterable[str],
        labels: Iterable[int],
    ) -> "FlammableQualityClassifier":
        text_list = list(texts)
        y = np.asarray(list(labels), dtype=np.int8)
        matrix = self._stack_features(embeddings, text_list)
        matrix, y = self._maybe_oversample(matrix, y)
        self._embedding_dim = int(embeddings.shape[1])
        self.model.fit(matrix, y)
        return self

    def predict_proba(self, embeddings: np.ndarray, texts: Iterable[str]) -> np.ndarray:
        matrix = self._stack_features(embeddings, list(texts))
        return self.model.predict_proba(matrix)[:, 1]

    def predict(self, embeddings: np.ndarray, texts: Iterable[str]) -> np.ndarray:
        return (self.predict_proba(embeddings, texts) >= self.threshold).astype(np.int8)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output, compress=3)

    @staticmethod
    def load(path: str | Path) -> "FlammableQualityClassifier":
        model = joblib.load(path)
        if not isinstance(model, FlammableQualityClassifier):
            raise TypeError(f"unexpected flammable classifier type: {type(model)!r}")
        return model


def flammable_row_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask for rows in the official flammable category."""

    return df["category"].astype(str).eq(FLAMMABLE_CATEGORY).to_numpy()


def predict_flammable_rows(
    model: FlammableQualityClassifier,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict only rows in the official flammable category."""

    mask = flammable_row_mask(df)
    if not mask.any():
        return np.array([], dtype=np.float64), np.array([], dtype=np.int8)
    texts = build_flammable_texts(df.loc[mask])
    emb = np.asarray(embeddings[mask], dtype=np.float32)
    probabilities = model.predict_proba(emb, texts)
    predictions = (probabilities >= model.threshold).astype(np.int8)
    return probabilities, predictions


def embedding_only_logreg_proba(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    c: float = 1.0,
) -> LogisticRegression:
    """Train embedding-only logistic regression (used for OOF baseline comparison)."""

    model = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_state,
        solver="liblinear",
    )
    model.fit(np.asarray(embeddings, dtype=np.float32), labels)
    return model
