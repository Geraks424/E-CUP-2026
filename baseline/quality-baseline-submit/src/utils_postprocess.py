import re
from typing import List, Sequence

from src.constants import (
    MIN_COMMENT_LEN,
    MAX_COMMENT_LEN,
)
from src.rules import BAD_CATEGORY, FLAMMABLE_CATEGORY, apply_rules


_MODEL_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_comment(raw_comment: object) -> str:
    if not isinstance(raw_comment, str):
        return ""
    comment = re.sub(r"<think>.*?</think>", " ", raw_comment, flags=re.DOTALL | re.IGNORECASE)
    comment = _MODEL_TAG_RE.sub(" ", comment)
    return _SPACE_RE.sub(" ", comment).strip()


def _quoted_term(term: str) -> str:
    cleaned = _SPACE_RE.sub(" ", term.strip())
    return f"«{cleaned[:80]}»" if cleaned else ""


def build_fallback_comment(category: object, text: object, label: int) -> str:
    """Build a verdict-consistent explanation from rule evidence only."""

    category_text = str(category).strip()
    try:
        decision = apply_rules(text, category_text)
    except ValueError:
        decision = None

    term = _quoted_term(decision.matched_terms[0]) if decision and decision.matched_terms else ""
    codes = set(decision.matched_codes) if decision else set()

    if category_text == BAD_CATEGORY:
        if label == 1:
            if decision and decision.label == 1 and term:
                return (
                    f"Товар отнесён к БАД: в названии или описании найдена прямая маркировка {term}. "
                    "Карточка соответствует заданному правилу категории."
                )
            return (
                "Товар классифицирован как БАД по совокупности признаков названия и описания. "
                "Прямого указания, исключающего принадлежность к БАД, в тексте карточки не найдено."
            )
        if "bad_explicit_negative" in codes:
            return (
                f"Товар не относится к БАД: в карточке найдено прямое указание {term}, "
                "исключающее принадлежность товара к биологически активным добавкам."
            )
        if any(code.startswith("sports_") or code == "sports_nutrition" for code in codes):
            return (
                f"Товар не относится к БАД: признак {term} указывает на спортивное питание, "
                "которое по заданному правилу исключено из категории БАД."
            )
        return (
            "В названии и описании нет прямой маркировки «БАД» или «dietary supplement». "
            "По заданному правилу товар не относится к биологически активным добавкам."
        )

    if category_text == FLAMMABLE_CATEGORY:
        if label == 1:
            if decision and decision.label == 1 and term:
                return (
                    f"Товар отнесён к легковоспламеняющимся: признак {term} указывает на источник "
                    "воспламенения, горючее вещество или горючий газ."
                )
            return (
                "По совокупности признаков карточки товар отнесён к легковоспламеняющимся. "
                "Итоговый признак требует сопоставления текста карточки с изображениями товара."
            )
        if "flammable_absent_content" in codes or "flammable_not_in_kit" in codes:
            return (
                f"Товар не отнесён к легковоспламеняющимся: указание {term} подтверждает, "
                "что горючее содержимое отсутствует или не входит в комплект."
            )
        if "flammable_component_only" in codes:
            return (
                f"Товар не отнесён к легковоспламеняющимся: {term} обозначает компонент другого "
                "изделия, а не самостоятельное горючее содержимое."
            )
        if "flammable_built_in_source" in codes:
            return (
                f"Товар не отнесён к легковоспламеняющимся: {term} обозначает встроенный "
                "источник воспламенения, а не самостоятельный товар."
            )
        if "flammable_device" in codes or "flammable_built_in_ignition" in codes:
            return (
                f"Товар не отнесён к легковоспламеняющимся: {term} обозначает устройство для "
                "работы с огнём, но горючее содержимое в карточке не указано."
            )
        return (
            "В названии и описании не указан самостоятельный источник воспламенения, горючее "
            "вещество, горючий газ или такой предмет в комплекте товара."
        )

    return (
        "Вердикт получен классификатором по данным карточки. Категория товара не поддерживается "
        "детерминированным модулем правил и требует ручной проверки."
    )


# Patch a single comment to fit within [min_len, max_len].
def _patch_comment(raw_comment: object, fallback: str, min_len: int, max_len: int) -> str:
    comment = _clean_comment(raw_comment) or fallback

    if len(comment) < min_len:
        suffix = " Решение основано на правилах категории и данных карточки."
        comment += suffix
        if len(comment) < min_len:
            comment += "." * (min_len - len(comment))

    if len(comment) > max_len:
        trim_idx = comment.rfind(" ", 0, max_len - 1)
        comment = comment[:trim_idx] if trim_idx > 0 else comment[: max_len - 1]
        comment = comment.rstrip(" ,;:-") + "."

    return comment


# Format comments and verdicts in a single pass.
def format_results(
    raw_comments: Sequence[object] | None,
    crisp_verdicts: Sequence[object],
    categories: Sequence[object] | None = None,
    texts: Sequence[object] | None = None,
) -> List[str]:
    n = len(crisp_verdicts)

    results = []

    for i in range(n):
        label = 1 if crisp_verdicts[i] in (1, True) else 0
        verdict = "не бан" if label == 1 else "бан"
        category = categories[i] if categories is not None and i < len(categories) else ""
        text = texts[i] if texts is not None and i < len(texts) else ""
        fallback = build_fallback_comment(category, text, label)
        raw = raw_comments[i] if raw_comments is not None and i < len(raw_comments) else ""
        comment = _patch_comment(
            raw,
            fallback,
            MIN_COMMENT_LEN,
            MAX_COMMENT_LEN,
        )
        results.append(f"<комментарий>{comment}<вердикт>{verdict}")

    return results
