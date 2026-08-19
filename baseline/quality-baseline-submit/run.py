import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from src.constants import (
    PIXEL_PRESETS,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_LLM_BATCH_SIZE,
    DEFAULT_PIXEL_PRESET,
)
from src.utils_data_prep import prepare_dataframe
from src.utils_logreg import ProductQualityPredictor
from src.utils_postprocess import format_results
from src.utils_embed_cuda import embed_data_cuda
from src.utils_generate_cuda import generate_comments_cuda
from src.bad_classifier import BadQualityClassifier, predict_bad_rows
from src.flammable_classifier import FlammableQualityClassifier, predict_flammable_rows
from src.rules import BAD_CATEGORY, FLAMMABLE_CATEGORY, apply_rules

# Classifier artifacts live next to run.py in the submitted archive.
_SUBMIT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = _SUBMIT_ROOT / "baseline_qwen3vl_bf16.joblib"
BAD_CLASSIFIER_PATH = _SUBMIT_ROOT / "arseniy_bad_text_model.joblib"
FLAMMABLE_CLASSIFIER_PATH = _SUBMIT_ROOT / "mark_flammable_model.joblib"

# Models path: match evaluator's SHARED_MODELS_PATH convention
_SHARED_MODELS_DIR = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
MODEL_EMBED_PATH = os.path.join(_SHARED_MODELS_DIR, "Qwen/Qwen3-VL-Embedding-2B")
MODEL_LLM_PATH = os.path.join(_SHARED_MODELS_DIR, "Qwen/Qwen3.5-4B")


def main() -> None:
    parser = argparse.ArgumentParser(description="Product quality predictor submit pipeline")
    parser.add_argument(
        "--test_data_path",
        "--test-data-path",
        "-i",
        dest="test_data_path",
        type=str,
        required=True,
        help="test data path",
    )
    parser.add_argument(
        "--output_path",
        "--output-path",
        "-o",
        dest="output_path",
        type=str,
        required=True,
        help="output file",
    )
    parser.add_argument(
        "--embed_batch",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="Embedding batch size (H100 default: 128; 8GB GPU: 1-2)",
    )
    parser.add_argument(
        "--llm_batch",
        type=int,
        default=DEFAULT_LLM_BATCH_SIZE,
        help="LLM generation batch size (H100 default: 64; 8GB GPU: 1)",
    )
    parser.add_argument(
        "--pixel_preset",
        choices=tuple(PIXEL_PRESETS.keys()),
        default=DEFAULT_PIXEL_PRESET,
        help="Vision pixel budget preset S/M/L (H100 default: M; 8GB GPU: S)",
    )
    parser.add_argument(
        "--comments_mode",
        choices=("llm", "rules"),
        default=os.environ.get("COMMENTS_MODE", "llm"),
        help="LLM explanations or deterministic rule-grounded explanations",
    )
    args = parser.parse_args()

    # Step 1: read and prepare combined text + structured image paths
    data_path = Path(args.test_data_path)
    images_path = data_path.parent / "images"
    current_df = prepare_dataframe(data_path, images_path)

    # Step 2: extract embeddings for text+images
    current_embeddings = embed_data_cuda(
        MODEL_EMBED_PATH, current_df,
        max_pixels=PIXEL_PRESETS[args.pixel_preset],
        batch_size=args.embed_batch,
    )
    current_df['embedding'] = current_embeddings.tolist()

    # Step 3: load classification models and predict
    trained_logreg = ProductQualityPredictor.load(CLASSIFIER_PATH)
    baseline_probs, baseline_preds = trained_logreg.predict(
        current_df['embedding'], current_df['category']
    )
    current_df['logreg_prob'] = baseline_probs
    current_df['pred'] = baseline_preds
    current_df['classifier_source'] = "multimodal_baseline"

    # Phase 3: replace only the БАД head.
    if BAD_CLASSIFIER_PATH.is_file():
        bad_model = BadQualityClassifier.load(BAD_CLASSIFIER_PATH)
        bad_mask = current_df['category'].astype(str).eq(BAD_CATEGORY).to_numpy()
        bad_probs, bad_preds = predict_bad_rows(bad_model, current_df)
        current_df.loc[bad_mask, 'logreg_prob'] = bad_probs
        current_df.loc[bad_mask, 'pred'] = bad_preds
        current_df.loc[bad_mask, 'classifier_source'] = "arseniy_bad_text_rules"
    else:
        print(
            f"WARNING: БАД artifact not found at {BAD_CLASSIFIER_PATH}; "
            "using the multimodal baseline for that category.",
            file=sys.stderr,
        )

    # Phase 4: replace only the flammable head when the dedicated artifact exists.
    flame_mask = current_df['category'].astype(str).eq(FLAMMABLE_CATEGORY).to_numpy()
    if FLAMMABLE_CLASSIFIER_PATH.is_file():
        flame_model = FlammableQualityClassifier.load(FLAMMABLE_CLASSIFIER_PATH)
        flame_probs, flame_preds = predict_flammable_rows(
            flame_model,
            current_df,
            current_embeddings,
        )
        current_df.loc[flame_mask, 'logreg_prob'] = flame_probs
        current_df.loc[flame_mask, 'pred'] = flame_preds
        current_df.loc[flame_mask, 'classifier_source'] = "mark_flammable_embed_rules"
    else:
        print(
            f"WARNING: flammable artifact not found at {FLAMMABLE_CLASSIFIER_PATH}; "
            "using the multimodal baseline for that category.",
            file=sys.stderr,
        )

    decisions = [apply_rules(text, category) for text, category in zip(current_df['text'], current_df['category'])]
    current_df['rule_score'] = [decision.score for decision in decisions]
    current_df['rule_label'] = [decision.label for decision in decisions]

    # Step 4: generate comments
    if args.comments_mode == "llm":
        comments = generate_comments_cuda(
            MODEL_LLM_PATH, current_df,
            batch_size=args.llm_batch,
        )
    else:
        comments = []

    # Step 5: sanitize or replace comments and enforce the Result contract.
    current_df['result'] = format_results(
        comments,
        current_df['pred'].tolist(),
        categories=current_df['category'].tolist(),
        texts=current_df['text'].tolist(),
    )

    # Step 6: finalize
    result_df = current_df[['id', 'result']]
    result_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
