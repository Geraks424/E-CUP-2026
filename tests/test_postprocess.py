from __future__ import annotations

from pathlib import Path
import re
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baseline" / "quality-baseline-submit"
for path in (REPO_ROOT, BASELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.validate_quality_submission import validate_submission  # noqa: E402
from src.explanations import build_user_prompt  # noqa: E402
from src.utils_postprocess import format_results  # noqa: E402


RESULT_RE = re.compile(r"^<комментарий>(.+)<вердикт>(бан|не бан)$", re.DOTALL)


class PostprocessTest(unittest.TestCase):
    def test_missing_llm_comments_get_rule_grounded_fallbacks(self):
        results = format_results(
            [],
            [1, 0],
            categories=["БАД", "Легковоспламеняющиеся"],
            texts=["БАД, 60 капсул", "Газовая плита без баллона"],
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].endswith("<вердикт>не бан"))
        self.assertTrue(results[1].endswith("<вердикт>бан"))
        self.assertIn("маркировка", results[0])
        self.assertIn("отсутствует", results[1])

    def test_model_tags_are_removed(self):
        result = format_results(
            ["<think>секрет</think><комментарий>Карточка содержит прямую маркировку БАД и соответствует правилу категории.<вердикт>бан"],
            [1],
            categories=["БАД"],
            texts=["БАД"],
        )[0]
        self.assertNotIn("секрет", result)
        self.assertEqual(result.count("<комментарий>"), 1)
        self.assertEqual(result.count("<вердикт>"), 1)
        self.assertTrue(result.endswith("<вердикт>не бан"))

    def test_comment_length_is_always_within_contract(self):
        results = format_results(
            ["коротко", "слово " * 200],
            [1, 0],
            categories=["БАД", "БАД"],
            texts=["БАД", "без маркировки"],
        )
        for result in results:
            match = RESULT_RE.match(result)
            self.assertIsNotNone(match)
            self.assertGreaterEqual(len(match.group(1)), 50)
            self.assertLessEqual(len(match.group(1)), 300)

    def test_csv_validator_accepts_generated_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_csv = root / "input.csv"
            output_csv = root / "submit.csv"
            frame = pd.DataFrame(
                {
                    "id": [10, 11],
                    "name": ["БАД", "Мангал"],
                    "description": ["Биологически активная добавка", "Без угля"],
                    "category": ["БАД", "Легковоспламеняющиеся"],
                }
            )
            frame.to_csv(input_csv, index=False)
            results = format_results(
                None,
                [1, 0],
                categories=frame["category"].tolist(),
                texts=(frame["name"] + " " + frame["description"]).tolist(),
            )
            pd.DataFrame({"id": frame["id"], "result": results}).to_csv(output_csv, index=False)
            self.assertEqual(validate_submission(input_csv, output_csv), [])


class PromptTest(unittest.TestCase):
    def test_prompt_fixes_verdict_and_contains_official_bad_rules(self):
        prompt = build_user_prompt("БАД, 60 капсул", "БАД", 1)
        self.assertIn("Зафиксированный итог классификатора: качественный товар (не бан)", prompt)
        self.assertIn("dietary supplement", prompt)
        self.assertIn("не меняй итог", prompt.lower())

    def test_prompt_uses_flammable_rules(self):
        prompt = build_user_prompt("Газовая плита без баллона", "Легковоспламеняющиеся", 0)
        self.assertIn("некачественный товар (бан)", prompt)
        self.assertIn("горючее вещество или газ", prompt)


if __name__ == "__main__":
    unittest.main()
