#!/usr/bin/env python3
"""
Dry-run smoke for Task 2 baseline packaging — NO CUDA / NO model inference.

Produces a valid submission CSV with deterministic placeholder comments and
verdicts derived from labels (if present) or a trivial always-ban rule.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

_BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.constants import MAX_COMMENT_LEN, MIN_COMMENT_LEN  # noqa: E402
from src.utils_data_prep import prepare_dataframe  # noqa: E402
from src.utils_postprocess import format_results  # noqa: E402

CLASSIFIER_FILENAME = "baseline_qwen3vl_bf16.joblib"
DRY_RUN_PREFIX = "[DRY-RUN] "


def _deterministic_comment(row: pd.Series) -> str:
    digest = hashlib.sha256(
        f"{row['id']}|{row['category']}|{row.get('label', '')}".encode("utf-8")
    ).hexdigest()
    base = (
        f"{DRY_RUN_PREFIX}Плейсхолдер для id={row['id']}, категория «{row['category']}». "
        f"Это не инференс baseline; комментарий сгенерирован детерминированно для проверки "
        f"формата сабмита. Хеш: {digest[:16]}."
    )
    if len(base) < MIN_COMMENT_LEN:
        base = base + " " + ("x" * (MIN_COMMENT_LEN - len(base)))
    if len(base) > MAX_COMMENT_LEN:
        base = base[: MAX_COMMENT_LEN - 3].rstrip() + "..."
    return base


def _verdict_from_row(row: pd.Series) -> int:
    if "label" in row and pd.notna(row["label"]):
        return int(row["label"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run smoke for quality baseline submit")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--baseline_dir",
        type=Path,
        default=_BASELINE_ROOT,
        help="Path to baseline/quality-baseline-submit",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="Images root (default: sibling of input_csv named images/)",
    )
    args = parser.parse_args()

    if not args.input_csv.is_file():
        print(f"ERROR: input CSV not found: {args.input_csv}", file=sys.stderr)
        return 1

    classifier_path = args.baseline_dir / CLASSIFIER_FILENAME
    if not classifier_path.is_file():
        print(f"ERROR: classifier joblib missing: {classifier_path}", file=sys.stderr)
        return 1

    images_dir = args.images_dir or (args.input_csv.parent / "images")
    df_raw = pd.read_csv(args.input_csv)
    prepared = prepare_dataframe(args.input_csv, images_dir)

    missing_images = int(prepared["image_paths"].map(len).eq(0).sum())
    print(f"DRY-RUN smoke | rows={len(prepared)} | missing_images={missing_images}")
    print(f"Classifier present: {classifier_path.name} ({classifier_path.stat().st_size} bytes)")

    comments = [_deterministic_comment(row) for _, row in df_raw.iterrows()]
    verdicts = [_verdict_from_row(row) for _, row in df_raw.iterrows()]
    results = format_results(comments, verdicts)

    out_df = pd.DataFrame({"id": df_raw["id"], "result": results})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote dry-run submission: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
