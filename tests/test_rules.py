from __future__ import annotations

from pathlib import Path
import sys
import unittest


BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from src.rules import (  # noqa: E402
    apply_bad_rules,
    apply_flammable_rules,
    apply_rules,
    normalize_product_text,
)


class BadRulesTest(unittest.TestCase):
    def test_direct_bad_marker_is_quality_one(self):
        decision = apply_bad_rules("Биологически активная добавка к пище, 60 капсул")
        self.assertEqual(decision.label, 1)
        self.assertIn("bad_full_name", decision.matched_codes)

    def test_english_marker_is_quality_one(self):
        decision = apply_bad_rules("Dietary Supplement. 30 capsules")
        self.assertEqual(decision.label, 1)

    def test_explicit_negative_overrides_marker(self):
        decision = apply_bad_rules("Продукт не является БАД, надпись БАД приведена для сравнения")
        self.assertEqual(decision.label, 0)
        self.assertEqual(decision.matched_codes, ("bad_explicit_negative",))

    def test_sports_nutrition_is_quality_zero(self):
        decision = apply_bad_rules("BCAA, аминокислоты для спортивного питания")
        self.assertEqual(decision.label, 0)
        self.assertIn("sports_nutrition", decision.matched_codes)

    def test_missing_marker_is_quality_zero(self):
        self.assertEqual(apply_bad_rules("Таблетница на семь дней").label, 0)


class FlammableRulesTest(unittest.TestCase):
    def test_standalone_ignition_source_is_quality_one(self):
        self.assertEqual(apply_flammable_rules("Спички длительного горения").label, 1)
        self.assertEqual(apply_flammable_rules("Зажигалка карманная").label, 1)

    def test_combustible_gas_is_quality_one(self):
        decision = apply_flammable_rules("Газовый баллон 650 г")
        self.assertEqual(decision.label, 1)
        self.assertIn("flammable_gas_container", decision.matched_codes)

    def test_device_without_content_is_quality_zero(self):
        decision = apply_flammable_rules("Газовая плита с пьезоподжигом, без баллона")
        self.assertEqual(decision.label, 0)
        self.assertIn("flammable_absent_content", decision.matched_codes)

    def test_device_alone_is_quality_zero(self):
        self.assertEqual(apply_flammable_rules("Мангал складной с решеткой").label, 0)

    def test_combustible_component_is_quality_zero(self):
        decision = apply_flammable_rules("Фильтр с активированным углем")
        self.assertEqual(decision.label, 0)
        self.assertIn("flammable_component_only", decision.matched_codes)

    def test_built_in_ignition_source_is_quality_zero(self):
        decision = apply_flammable_rules("Фонарик со встроенной зажигалкой")
        self.assertEqual(decision.label, 0)
        self.assertIn("flammable_built_in_source", decision.matched_codes)

    def test_item_in_kit_is_quality_one(self):
        self.assertEqual(
            apply_flammable_rules("Подарочный набор, в комплект входит зажигалка").label,
            1,
        )


class CommonRulesTest(unittest.TestCase):
    def test_html_is_removed(self):
        self.assertEqual(normalize_product_text("<p>БАД&nbsp; 60 шт.</p>"), "бад 60 шт.")

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_rules("товар", "Неизвестная")


if __name__ == "__main__":
    unittest.main()
