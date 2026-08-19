"""Deterministic rule signals from the official E-CUP 2026 Quality rules.

The rules in this module are deliberately conservative.  They do not try to
replace the learned classifier; instead they expose auditable signals and a
preliminary decision that can be used as model features and as evidence for an
explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from typing import Iterable


BAD_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"


@dataclass(frozen=True)
class RuleDecision:
    """Auditable result of applying the official category rules."""

    category: str
    label: int
    score: float
    matched_codes: tuple[str, ...]
    matched_terms: tuple[str, ...]
    reason: str

    @property
    def high_confidence(self) -> bool:
        return abs(self.score) >= 0.8


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def normalize_product_text(value: object) -> str:
    """Normalize card text while preserving the words used by the rules."""

    if value is None:
        return ""
    text = unescape(str(value)).lower().replace("ё", "е")
    text = _HTML_TAG_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def combine_product_text(name: object, description: object) -> str:
    return normalize_product_text(f"{name or ''} {description or ''}")


def _compile(items: Iterable[tuple[str, str]]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((code, re.compile(pattern, re.IGNORECASE)) for code, pattern in items)


_BAD_DIRECT = _compile(
    (
        ("bad_acronym", r"\bбад(?:ы|а|ом|у)?\b"),
        (
            "bad_full_name",
            r"\bбиологически\s+активн\w*\s+добавк\w*\b",
        ),
        ("bad_dietary_supplement", r"\bdietary\s+supplements?\b"),
    )
)

_BAD_EXPLICIT_NEGATIVE = _compile(
    (
        (
            "bad_explicit_negative",
            r"\bне\s+(?:явля\w+\s+)?(?:бад(?:ом)?|"
            r"биологически\s+активн\w*\s+добавк\w*)\b",
        ),
    )
)

_BAD_SPORTS_NUTRITION = _compile(
    (
        ("sports_nutrition", r"\bспортивн\w*\s+питан\w*\b|\bспортпит\w*\b"),
        ("sports_amino_acids", r"\bаминокислот\w*\b"),
        ("sports_bcaa", r"\bbcaa\b|\bбцаа\b"),
        (
            "sports_l_carnitine",
            r"\bl[\s-]*carnitine\b|\bл[\s-]*карнитин\w*\b",
        ),
        ("sports_protein", r"\bпротеин\w*\b|\bprotein\b"),
    )
)


_FLAMMABLE_NO_CONTENT = _compile(
    (
        (
            "flammable_absent_content",
            r"\bбез\s+(?:газов\w*\s+)?баллон\w*\b|\bбез\s+(?:газа|топлива|"
            r"спичек|зажигалк\w*|угля)\b",
        ),
        (
            "flammable_not_in_kit",
            r"\b(?:не\s+входит|не\s+включен\w*)\s+в\s+комплект\w*\b",
        ),
    )
)

_FLAMMABLE_COMPONENT_ONLY = _compile(
    (
        (
            "flammable_component_only",
            r"\bактивированн\w*\s+угл\w*\b|\bугл\w*\s+для\s+рисован\w*\b|"
            r"\bугольн\w*\s+карандаш\w*\b",
        ),
    )
)

_FLAMMABLE_BUILT_IN = _compile(
    (
        (
            "flammable_built_in_source",
            r"\bвстроен\w*(?:\s+\w+){0,3}\s+(?:источник\w*\s+огн\w*|"
            r"зажигалк\w*|поджиг\w*)\b",
        ),
    )
)

_FLAMMABLE_SOURCE = _compile(
    (
        ("flammable_matches", r"\bспич(?:ка|ки|ек|ечн\w*)\b"),
        ("flammable_lighter", r"\bзажигалк\w*\b"),
    )
)

_FLAMMABLE_SUBSTANCE = _compile(
    (
        ("flammable_explicit", r"\bлегковоспламен\w*\b|\bгорюч\w*\b"),
        (
            "flammable_gas_container",
            r"\bгазов\w*\s+баллон\w*\b|\bбаллон\w*\s+(?:с\s+)?газ\w*\b",
        ),
        (
            "flammable_fuel",
            r"\bтоплив\w*\b|\bбензин\w*\b|\bкеросин\w*\b|\bбиоэтанол\w*\b|"
            r"\bжидкост\w*\s+для\s+розжиг\w*\b",
        ),
        (
            "flammable_coal_or_wood",
            r"\bдревесн\w*\s+угл\w*\b|\bугл\w*\s+древесн\w*\b|"
            r"\bтопливн\w*\s+брикет\w*\b|\bдров\w*\b",
        ),
    )
)

_FLAMMABLE_DEVICE = _compile(
    (
        ("flammable_device", r"\bмангал\w*\b|\bгрил\w*\b|\bгазов\w*\s+плит\w*\b|\bгорелк\w*\b"),
        ("flammable_built_in_ignition", r"\bпьезоподжиг\w*\b|\bэлектроподжиг\w*\b"),
    )
)


def _matches(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for code, pattern in patterns:
        match = pattern.search(text)
        if match:
            found.append((code, match.group(0)))
    return found


def _decision(
    category: str,
    label: int,
    score: float,
    matches: list[tuple[str, str]],
    reason: str,
) -> RuleDecision:
    return RuleDecision(
        category=category,
        label=label,
        score=score,
        matched_codes=tuple(code for code, _ in matches),
        matched_terms=tuple(term for _, term in matches),
        reason=reason,
    )


def apply_bad_rules(text: object) -> RuleDecision:
    """Apply only the supplied biological-supplement rules."""

    normalized = normalize_product_text(text)
    explicit_negative = _matches(normalized, _BAD_EXPLICIT_NEGATIVE)
    sports = _matches(normalized, _BAD_SPORTS_NUTRITION)
    direct = _matches(normalized, _BAD_DIRECT)

    if explicit_negative:
        return _decision(
            BAD_CATEGORY,
            0,
            -1.0,
            explicit_negative,
            "В карточке прямо указано, что товар не является БАД.",
        )
    if sports:
        return _decision(
            BAD_CATEGORY,
            0,
            -0.9,
            sports,
            "Карточка относит товар к спортивному питанию.",
        )
    if direct:
        return _decision(
            BAD_CATEGORY,
            1,
            1.0,
            direct,
            "В карточке есть прямая маркировка БАД.",
        )
    return _decision(
        BAD_CATEGORY,
        0,
        -0.6,
        [],
        "В тексте карточки нет прямой маркировки БАД.",
    )


def apply_flammable_rules(text: object) -> RuleDecision:
    """Apply the supplied flammability rules to product-card text."""

    normalized = normalize_product_text(text)
    absent = _matches(normalized, _FLAMMABLE_NO_CONTENT)
    component = _matches(normalized, _FLAMMABLE_COMPONENT_ONLY)
    built_in = _matches(normalized, _FLAMMABLE_BUILT_IN)
    sources = _matches(normalized, _FLAMMABLE_SOURCE)
    substances = _matches(normalized, _FLAMMABLE_SUBSTANCE)
    devices = _matches(normalized, _FLAMMABLE_DEVICE)

    if absent:
        return _decision(
            FLAMMABLE_CATEGORY,
            0,
            -1.0,
            absent,
            "В карточке указано, что воспламеняющееся содержимое отсутствует или не входит в комплект.",
        )
    if component:
        return _decision(
            FLAMMABLE_CATEGORY,
            0,
            -0.9,
            component,
            "Горючий материал указан только как компонент другого изделия.",
        )
    if built_in:
        return _decision(
            FLAMMABLE_CATEGORY,
            0,
            -0.9,
            built_in,
            "Источник воспламенения встроен в другое изделие и не является самостоятельным товаром.",
        )
    if sources or substances:
        evidence = sources + substances
        return _decision(
            FLAMMABLE_CATEGORY,
            1,
            1.0 if sources else 0.9,
            evidence,
            "Товар является источником воспламенения либо содержит горючее вещество или газ.",
        )
    if devices:
        return _decision(
            FLAMMABLE_CATEGORY,
            0,
            -0.8,
            devices,
            "Устройство предназначено для работы с огнем, но горючее содержимое в карточке не указано.",
        )
    return _decision(
        FLAMMABLE_CATEGORY,
        0,
        -0.5,
        [],
        "В тексте карточки не найден источник воспламенения или горючее вещество.",
    )


def apply_rules(text: object, category: object) -> RuleDecision:
    normalized_category = str(category).strip()
    if normalized_category == BAD_CATEGORY:
        return apply_bad_rules(text)
    if normalized_category == FLAMMABLE_CATEGORY:
        return apply_flammable_rules(text)
    raise ValueError(f"unsupported category: {category!r}")


def bad_rule_features(text: object) -> list[float]:
    """Numeric rule features consumed by the BАD text classifier."""

    normalized = normalize_product_text(text)
    direct = float(bool(_matches(normalized, _BAD_DIRECT)))
    explicit_negative = float(bool(_matches(normalized, _BAD_EXPLICIT_NEGATIVE)))
    sports = float(bool(_matches(normalized, _BAD_SPORTS_NUTRITION)))
    decision = apply_bad_rules(normalized)
    no_direct_marker = 1.0 - direct
    return [direct, explicit_negative, sports, no_direct_marker, decision.score]
