#!/usr/bin/env python3
"""CPU-only Result-stage smoke test for Arseniy's Phase 2/3/5 components."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baseline" / "quality-baseline-submit"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from src.bad_classifier import BadQualityClassifier, build_bad_texts  # noqa: E402
from src.rules import BAD_CATEGORY, apply_rules  # noqa: E402
from src.utils_postprocess import format_results  # noqa: E402


REQUIRED_COLUMNS = {"id", "name", "description", "category"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=BASELINE_ROOT / "arseniy_bad_text_model.joblib",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("local_data/arseniy_components_smoke.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        print(f"ERROR: missing columns: {missing}", file=sys.stderr)
        return 1

    texts = build_bad_texts(frame)
    predictions = np.asarray(
        [apply_rules(text, category).label for text, category in zip(texts, frame["category"])],
        dtype=np.int8,
    )

    bad_mask = frame["category"].astype(str).eq(BAD_CATEGORY).to_numpy()
    bad_model = BadQualityClassifier.load(args.model)
    predictions[bad_mask] = bad_model.predict(np.asarray(texts, dtype=object)[bad_mask])

    results = format_results(
        None,
        predictions.tolist(),
        categories=frame["category"].tolist(),
        texts=texts,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame["id"], "result": results}).to_csv(args.output_csv, index=False)
    print(f"Wrote {len(frame)} rows: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

