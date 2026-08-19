#!/usr/bin/env python3
"""Build a reproducible reference set of rule-grounded explanations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baseline" / "quality-baseline-submit"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from src.bad_classifier import build_bad_texts  # noqa: E402
from src.rules import apply_rules  # noqa: E402
from src.utils_postprocess import build_fallback_comment  # noqa: E402


TARGET_RULES = (
    ("БАД", 1, "bad_acronym"),
    ("БАД", 1, "bad_full_name"),
    ("БАД", 1, "bad_dietary_supplement"),
    ("БАД", 0, "bad_explicit_negative"),
    ("БАД", 0, "sports_nutrition"),
    ("БАД", 0, "sports_protein"),
    ("Легковоспламеняющиеся", 1, "flammable_matches"),
    ("Легковоспламеняющиеся", 1, "flammable_lighter"),
    ("Легковоспламеняющиеся", 1, "flammable_gas_container"),
    ("Легковоспламеняющиеся", 0, "flammable_absent_content"),
    ("Легковоспламеняющиеся", 0, "flammable_device"),
    ("Легковоспламеняющиеся", 0, "flammable_component_only"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", type=Path, required=True)
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("reports/arseniy/explanation-golden-set.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.data_csv)
    texts = build_bad_texts(frame)
    candidates: list[dict] = []
    for position, (row, text) in enumerate(zip(frame.to_dict("records"), texts)):
        decision = apply_rules(text, row["category"])
        if decision.label != int(row["label"]):
            continue
        for category, label, code in TARGET_RULES:
            if row["category"] == category and int(row["label"]) == label and code in decision.matched_codes:
                candidates.append(
                    {
                        "position": position,
                        "id": row["id"],
                        "name": row["name"],
                        "category": category,
                        "label": label,
                        "rule_code": code,
                        "matched_terms": "|".join(decision.matched_terms),
                        "text_length": len(text),
                        "comment": build_fallback_comment(category, text, label),
                    }
                )

    candidate_frame = pd.DataFrame(candidates)
    selected: list[pd.Series] = []
    for category, label, code in TARGET_RULES:
        matches = candidate_frame.loc[
            candidate_frame["category"].eq(category)
            & candidate_frame["label"].eq(label)
            & candidate_frame["rule_code"].eq(code)
        ]
        if matches.empty:
            print(f"WARNING: no matching labeled example for {code}", file=sys.stderr)
            continue
        selected.append(matches.sort_values(["text_length", "id"]).iloc[0])

    output = pd.DataFrame(selected).drop(columns=["position", "text_length"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(output)} reference explanations: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

