#!/usr/bin/env python3
"""Train and validate Mark's embedding + rule classifier for flammable products."""

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_ROOT = _REPO_ROOT / "baseline" / "quality-baseline-submit"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.embed_cache import load_cached_embeddings  # noqa: E402
from src.flammable_classifier import (  # noqa: E402
    FlammableQualityClassifier,
    build_flammable_texts,
    embedding_only_logreg_proba,
)
from src.rules import FLAMMABLE_CATEGORY, apply_flammable_rules, normalize_product_text  # noqa: E402

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
    parser.add_argument("--data_csv", type=Path, default=Path("D:/data.csv"))
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("local_data/flammable_embed_cache"),
    )
    parser.add_argument(
        "--output_model",
        type=Path,
        default=_BASELINE_ROOT / "mark_flammable_model.joblib",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/mark/phase4-flammable.json"),
    )
    parser.add_argument(
        "--error_report",
        type=Path,
        default=Path("reports/mark/phase4-flammable-errors.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--oversample_positives", action="store_true")
    args = parser.parse_args()

    if not args.data_csv.is_file():
        print(f"ERROR: data CSV not found: {args.data_csv}", file=sys.stderr)
        return 1

    frame = pd.read_csv(args.data_csv)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 1

    flame = frame.loc[frame["category"].astype(str).eq(FLAMMABLE_CATEGORY)].copy().reset_index(drop=True)
    if flame.empty or flame["label"].nunique() != 2:
        print("ERROR: flammable rows with both labels are required", file=sys.stderr)
        return 1

    embeddings, loaded_ids = load_cached_embeddings(flame["id"].tolist(), args.cache_dir)
    if embeddings.size == 0:
        print(
            f"ERROR: no cached embeddings in {args.cache_dir}; "
            "run scripts/cache_flammable_embeddings.py first",
            file=sys.stderr,
        )
        return 1

    flame = flame.loc[flame["id"].astype(str).isin(loaded_ids)].copy().reset_index(drop=True)
    id_to_row = {str(row_id): idx for idx, row_id in enumerate(loaded_ids)}
    order = [id_to_row[str(row_id)] for row_id in flame["id"].tolist()]
    embeddings = embeddings[order]

    texts = np.asarray(build_flammable_texts(flame), dtype=object)
    labels = flame["label"].astype(int).to_numpy()
    groups = flame["name"].fillna("").map(normalize_product_text).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    oof_full = np.zeros(len(flame), dtype=np.float64)
    oof_embed = np.zeros(len(flame), dtype=np.float64)
    oof_full_pred = np.zeros(len(flame), dtype=np.int8)
    oof_embed_pred = np.zeros(len(flame), dtype=np.int8)
    fold_indices: list[np.ndarray] = []
    per_fold_thresholds: list[float] = []
    per_fold_embed_thresholds: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(embeddings, labels, groups), start=1):
        print(
            f"Fold {fold}/{args.folds}: train={len(train_idx)}, valid={len(valid_idx)}",
            flush=True,
        )
        full_model = FlammableQualityClassifier(
            random_state=args.seed + fold,
            c=args.c,
            oversample_positives=args.oversample_positives,
        )
        full_model.fit(embeddings[train_idx], texts[train_idx], labels[train_idx])
        train_probs = full_model.predict_proba(embeddings[train_idx], texts[train_idx])
        fold_threshold, _ = _best_threshold(labels[train_idx], train_probs)
        per_fold_thresholds.append(fold_threshold)
        valid_probs = full_model.predict_proba(embeddings[valid_idx], texts[valid_idx])
        oof_full[valid_idx] = valid_probs
        oof_full_pred[valid_idx] = (valid_probs >= fold_threshold).astype(np.int8)

        embed_model = embedding_only_logreg_proba(
            embeddings[train_idx],
            labels[train_idx],
            random_state=args.seed + fold,
            c=args.c,
        )
        embed_train_probs = embed_model.predict_proba(embeddings[train_idx])[:, 1]
        embed_threshold, _ = _best_threshold(labels[train_idx], embed_train_probs)
        per_fold_embed_thresholds.append(embed_threshold)
        embed_valid_probs = embed_model.predict_proba(embeddings[valid_idx])[:, 1]
        oof_embed[valid_idx] = embed_valid_probs
        oof_embed_pred[valid_idx] = (embed_valid_probs >= embed_threshold).astype(np.int8)

        fold_indices.append(valid_idx)

    full_predictions = oof_full_pred
    embed_predictions = oof_embed_pred
    deployed_threshold = float(np.median(per_fold_thresholds))

    fold_metrics = [
        {"fold": fold, **_metrics(labels[idx], full_predictions[idx])}
        for fold, idx in enumerate(fold_indices, start=1)
    ]

    rule_decisions = [apply_flammable_rules(text) for text in texts]
    rules_predictions = np.asarray([decision.label for decision in rule_decisions], dtype=np.int8)
    rules_metrics = _metrics(labels, rules_predictions)
    embed_metrics = _metrics(labels, embed_predictions)
    full_metrics = _metrics(labels, full_predictions)

    errors = flame.loc[full_predictions != labels, ["id", "name", "description", "label"]].copy()
    error_positions = np.flatnonzero(full_predictions != labels)
    errors["prediction"] = full_predictions[error_positions]
    errors["probability_quality_1"] = oof_full[error_positions]
    errors["distance_from_threshold"] = np.abs(oof_full[error_positions] - deployed_threshold)
    errors["rule_label"] = rules_predictions[error_positions]
    errors["rule_codes"] = ["|".join(rule_decisions[idx].matched_codes) for idx in error_positions]
    errors["rule_terms"] = ["|".join(rule_decisions[idx].matched_terms) for idx in error_positions]
    errors["description"] = errors["description"].fillna("").map(normalize_product_text).str.slice(0, 300)
    errors = errors.sort_values("distance_from_threshold", ascending=False)
    args.error_report.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.error_report, index=False)

    final_model = FlammableQualityClassifier(
        threshold=deployed_threshold,
        random_state=args.seed,
        c=args.c,
        oversample_positives=args.oversample_positives,
    )
    print("Training final model on all cached flammable rows", flush=True)
    final_model.fit(embeddings, texts, labels)
    final_model.save(args.output_model)

    baseline_reference_f1 = 0.11764705882352941
    report = {
        "task": "quality_control",
        "owner": "Марк Досков",
        "phase": 4,
        "category": FLAMMABLE_CATEGORY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": _display_path(args.data_csv),
            "sha256": _sha256(args.data_csv),
            "rows_category_total": int(
                frame["category"].astype(str).eq(FLAMMABLE_CATEGORY).sum()
            ),
            "rows_with_embeddings": int(len(flame)),
            "label_counts": {
                str(key): int(value)
                for key, value in flame["label"].value_counts().sort_index().items()
            },
            "embedding_cache": _display_path(args.cache_dir),
            "missing_embeddings": int(
                frame["category"].astype(str).eq(FLAMMABLE_CATEGORY).sum() - len(flame)
            ),
        },
        "validation": {
            "method": "out-of-fold StratifiedGroupKFold grouped by normalized exact product name",
            "folds": args.folds,
            "seed": args.seed,
            "threshold_tuned_on": "train-fold probabilities only; OOF uses that fold's threshold",
            "per_fold_thresholds": per_fold_thresholds,
            "deployed_threshold_median_of_folds": deployed_threshold,
            "per_fold_embed_thresholds": per_fold_embed_thresholds,
            "oof": full_metrics,
            "per_fold_at_global_threshold": fold_metrics,
            "rules_only": rules_metrics,
            "embedding_logreg_only": embed_metrics,
            "baseline_reference": {
                "source": "reports/baseline/phase0-baseline.json (50-row subset)",
                "per_category_f1_flammable": baseline_reference_f1,
                "beats_baseline": full_metrics["f1"] > baseline_reference_f1,
                "beats_embedding_logreg_same_folds": full_metrics["f1"] > embed_metrics["f1"],
            },
        },
        "model": {
            "type": "Qwen3-VL embedding + official rule features + logistic regression",
            "c": args.c,
            "class_weight": "balanced",
            "oversample_positives": args.oversample_positives,
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
            "Training uses cached multimodal embeddings only for rows present in the embed cache.",
            "Inference in run.py reuses embeddings computed for the full submission batch.",
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
