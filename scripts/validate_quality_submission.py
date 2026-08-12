#!/usr/bin/env python3
"""Validate Task 2 quality submission CSV against input ids and format rules."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Allow importing baseline postprocess constants
_BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.constants import MAX_COMMENT_LEN, MIN_COMMENT_LEN  # noqa: E402

RESULT_PATTERN = re.compile(
    r"^<комментарий>(.+)<вердикт>(бан|не бан)$",
    re.DOTALL,
)
VALID_VERDICTS = {"бан", "не бан"}


def validate_submission(input_csv: Path, submission_csv: Path) -> list[str]:
    errors: list[str] = []

    if not submission_csv.is_file():
        return [f"submission file not found: {submission_csv}"]

    input_df = pd.read_csv(input_csv)
    sub_df = pd.read_csv(submission_csv)

    if list(sub_df.columns) != ["id", "result"]:
        errors.append(f"expected columns ['id', 'result'], got {list(sub_df.columns)}")

    if sub_df["id"].duplicated().any():
        dupes = sub_df.loc[sub_df["id"].duplicated(), "id"].tolist()
        errors.append(f"duplicate ids in submission: {dupes[:10]}")

    input_ids = set(input_df["id"].astype(sub_df["id"].dtype if len(sub_df) else object))
    sub_ids = set(sub_df["id"])
    missing = sorted(input_ids - sub_ids)
    extra = sorted(sub_ids - input_ids)
    if missing:
        errors.append(f"missing ids ({len(missing)}): {missing[:10]}")
    if extra:
        errors.append(f"unexpected ids ({len(extra)}): {extra[:10]}")

    for idx, row in sub_df.iterrows():
        result = row.get("result", "")
        if not isinstance(result, str):
            errors.append(f"id={row['id']}: result is not a string")
            continue

        match = RESULT_PATTERN.match(result.strip())
        if not match:
            errors.append(f"id={row['id']}: invalid result format/tags")
            continue

        comment, verdict = match.group(1), match.group(2)
        if verdict not in VALID_VERDICTS:
            errors.append(f"id={row['id']}: invalid verdict {verdict!r}")

        comment_len = len(comment)
        if comment_len < MIN_COMMENT_LEN or comment_len > MAX_COMMENT_LEN:
            errors.append(
                f"id={row['id']}: comment length {comment_len} "
                f"outside [{MIN_COMMENT_LEN}, {MAX_COMMENT_LEN}]"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate quality submission CSV")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--submission_csv", type=Path, required=True)
    args = parser.parse_args()

    errors = validate_submission(args.input_csv, args.submission_csv)
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: {args.submission_csv} valid for {args.input_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
