#!/usr/bin/env python3
"""Train and validate Arseniy's text + rule classifier for the БАД category."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


_BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.bad_classifier import BadQualityClassifier, build_bad_texts  # noqa: E402
from src.rules import BAD_CATEGORY, apply_bad_rules, normalize_product_text  # noqa: E402


REQUIRED_COLUMNS = {"id", "name", "description", "category", "label"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scored = [
        (float(f1_score(labels, probabilities >= threshold, zero_division=0)), float(threshold))
        for threshold in candidates
    ]
    best_f1 = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if score == best_f1]
    threshold = min(tied, key=lambda value: abs(value - 0.5))
    return threshold, best_f1


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "confusion_matrix_labels_0_1": matrix.astype(int).tolist(),
        "positives_true": int(labels.sum()),
        "positives_pred": int(predictions.sum()),
        "rows": int(len(labels)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", type=Path, required=True)
    parser.add_argument(
        "--output_model",
        type=Path,
        default=_BASELINE_ROOT / "arseniy_bad_text_model.joblib",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/arseniy/phase3-bad.json"),
    )
    parser.add_argument(
        "--error_report",
        type=Path,
        default=Path("reports/arseniy/phase3-bad-errors.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--max_word_features", type=int, default=60_000)
    parser.add_argument("--max_char_features", type=int, default=80_000)
    args = parser.parse_args()

    if not args.data_csv.is_file():
        print(f"ERROR: data CSV not found: {args.data_csv}", file=sys.stderr)
        return 1

    frame = pd.read_csv(args.data_csv)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 1

    bad = frame.loc[frame["category"].astype(str).eq(BAD_CATEGORY)].copy().reset_index(drop=True)
    if bad.empty or bad["label"].nunique() != 2:
        print("ERROR: БАД rows with both labels are required", file=sys.stderr)
        return 1

    texts = np.asarray(build_bad_texts(bad), dtype=object)
    labels = bad["label"].astype(int).to_numpy()
    groups = bad["name"].fillna("").map(normalize_product_text).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    oof = np.zeros(len(bad), dtype=np.float64)
    fold_indices: list[np.ndarray] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(texts, labels, groups), start=1):
        print(
            f"Fold {fold}/{args.folds}: train={len(train_idx)}, valid={len(valid_idx)}",
            flush=True,
        )
        model = BadQualityClassifier(
            random_state=args.seed + fold,
            c=args.c,
            max_word_features=args.max_word_features,
            max_char_features=args.max_char_features,
        )
        model.fit(texts[train_idx], labels[train_idx])
        oof[valid_idx] = model.predict_proba(texts[valid_idx])
        fold_indices.append(valid_idx)

    threshold, _ = _best_threshold(labels, oof)
    predictions = (oof >= threshold).astype(np.int8)
    fold_metrics = [
        {"fold": fold, **_metrics(labels[idx], predictions[idx])}
        for fold, idx in enumerate(fold_indices, start=1)
    ]

    rule_decisions = [apply_bad_rules(text) for text in texts]
    rules_predictions = np.asarray([decision.label for decision in rule_decisions], dtype=np.int8)
    rules_metrics = _metrics(labels, rules_predictions)

    errors = bad.loc[predictions != labels, ["id", "name", "description", "label"]].copy()
    error_positions = np.flatnonzero(predictions != labels)
    errors["prediction"] = predictions[error_positions]
    errors["probability_quality_1"] = oof[error_positions]
    errors["distance_from_threshold"] = np.abs(oof[error_positions] - threshold)
    errors["rule_label"] = rules_predictions[error_positions]
    errors["rule_codes"] = ["|".join(rule_decisions[idx].matched_codes) for idx in error_positions]
    errors["rule_terms"] = ["|".join(rule_decisions[idx].matched_terms) for idx in error_positions]
    errors["description"] = errors["description"].fillna("").map(normalize_product_text).str.slice(0, 300)
    errors = errors.sort_values("distance_from_threshold", ascending=False)
    args.error_report.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.error_report, index=False)

    final_model = BadQualityClassifier(
        threshold=threshold,
        random_state=args.seed,
        c=args.c,
        max_word_features=args.max_word_features,
        max_char_features=args.max_char_features,
    )
    print("Training final model on all БАД rows", flush=True)
    final_model.fit(texts, labels)
    final_model.save(args.output_model)

    report = {
        "task": "quality_control",
        "owner": "Арсений Нестеренко",
        "phase": 3,
        "category": BAD_CATEGORY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": _display_path(args.data_csv),
            "sha256": _sha256(args.data_csv),
            "rows": int(len(bad)),
            "label_counts": {
                str(key): int(value)
                for key, value in bad["label"].value_counts().sort_index().items()
            },
            "missing_descriptions": int(bad["description"].isna().sum()),
        },
        "validation": {
            "method": "out-of-fold StratifiedGroupKFold grouped by normalized exact product name",
            "folds": args.folds,
            "seed": args.seed,
            "threshold_tuned_on": "OOF probabilities",
            "threshold": threshold,
            "oof": _metrics(labels, predictions),
            "per_fold_at_global_threshold": fold_metrics,
            "rules_only": rules_metrics,
        },
        "model": {
            "type": "word TF-IDF + char TF-IDF + official rule features + logistic regression",
            "c": args.c,
            "class_weight": "balanced",
            "max_word_features": args.max_word_features,
            "max_char_features": args.max_char_features,
            "artifact": _display_path(args.output_model),
            "artifact_size_bytes": int(args.output_model.stat().st_size),
            "artifact_sha256": _sha256(args.output_model),
        },
        "error_analysis": {
            "misclassified_rows": int(len(errors)),
            "report": _display_path(args.error_report),
            "ordering": "most confident OOF errors first",
        },
        "label_contract": {"0": "не качественный / бан", "1": "качественный / не бан"},
        "limitations": [
            "The Phase 3 classifier uses name and description only.",
            "Image evidence remains available to the existing multimodal baseline and Phase 4 integration.",
        ],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["validation"], ensure_ascii=False, indent=2))
    print(f"Saved model: {args.output_model}")
    print(f"Saved report: {args.report}")
    print(f"Saved error analysis: {args.error_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
