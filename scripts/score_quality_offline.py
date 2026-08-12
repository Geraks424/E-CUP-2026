#!/usr/bin/env python3
"""Offline macro-averaged F1 for Task 2 quality submissions (labeled subset only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

RESULT_PATTERN = re.compile(r"^<комментарий>.+<вердикт>(бан|не бан)$", re.DOTALL)
VERDICT_TO_LABEL = {"не бан": 1, "бан": 0}


def _parse_verdict(result: str) -> int | None:
    if not isinstance(result, str):
        return None
    match = RESULT_PATTERN.match(result.strip())
    if not match:
        return None
    return VERDICT_TO_LABEL.get(match.group(1))


def score_submission(
    input_csv: Path,
    submission_csv: Path,
    mode: str = "dry_run",
) -> dict:
    input_df = pd.read_csv(input_csv)
    sub_df = pd.read_csv(submission_csv)

    if "label" not in input_df.columns:
        raise ValueError("input CSV has no label column; offline score requires labels")

    merged = input_df.merge(sub_df, on="id", how="inner", suffixes=("_true", "_pred"))
    merged["pred_label"] = merged["result"].map(_parse_verdict)
    invalid = merged["pred_label"].isna().sum()
    if invalid:
        raise ValueError(f"could not parse verdict for {invalid} rows")

    per_category: dict[str, dict] = {}
    f1_values: list[float] = []
    for category, group in merged.groupby("category"):
        y_true = group["label"].astype(int).tolist()
        y_pred = group["pred_label"].astype(int).tolist()
        cat_f1 = float(f1_score(y_true, y_pred))
        per_category[str(category)] = {
            "f1": cat_f1,
            "support": int(len(group)),
            "positives_true": int(sum(y_true)),
            "positives_pred": int(sum(y_pred)),
        }
        f1_values.append(cat_f1)

    macro_f1 = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0
    return {
        "macro_f1": macro_f1,
        "per_category_f1": {k: v["f1"] for k, v in per_category.items()},
        "per_category": per_category,
        "rows_scored": int(len(merged)),
        "verdict_mapping": {"не бан": 1, "бан": 0},
        "note": (
            "Macro F1 = mean of per-category F1 (БАД, Легковоспламеняющиеся)."
            + (
                " Dry-run scores use label-based placeholder verdicts, not model inference."
                if mode == "dry_run"
                else " Scores from real baseline model inference (logreg + LLM comments)."
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score labeled quality submission offline")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--submission_csv", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/baseline/phase0-baseline.json"),
        help="JSON report output path",
    )
    parser.add_argument(
        "--mode",
        choices=("dry_run", "model"),
        default="dry_run",
        help="Score context: dry_run placeholder vs model inference",
    )
    args = parser.parse_args()

    try:
        metrics = score_submission(args.input_csv, args.submission_csv, mode=args.mode)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = {
        "task": "quality_control",
        "phase": 0,
        "mode": args.mode,
        "status": "awaiting_gpu" if args.mode == "dry_run" else "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "submission_csv": str(args.submission_csv),
        "metrics": metrics,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
