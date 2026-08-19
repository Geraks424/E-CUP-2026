"""Lightweight, rule-grounded prompt construction for Phase 5."""

from __future__ import annotations

from src.rules import BAD_CATEGORY, apply_rules


BAD_RULES = (
    "БАД: товар относится к категории только при прямой маркировке «БАД», "
    "«биологически активная добавка» или «dietary supplement». Спортивное питание "
    "(аминокислоты, BCAA, L-карнитин, протеин), прямое отрицание или отсутствие маркировки "
    "означают, что товар не относится к БАД."
)

FLAMMABLE_RULES = (
    "Легковоспламеняющиеся: товар относится к категории, если это самостоятельный источник "
    "огня, он содержит горючее вещество или газ, либо такой предмет входит в комплект. "
    "Устройство без горючего содержимого, встроенный источник, компонент другого изделия "
    "или предмет, не входящий в комплект, к категории не относится."
)


def build_user_prompt(text: str, category: str, prediction: int) -> str:
    """Create a prompt that cannot silently change the classifier verdict."""

    rules = BAD_RULES if category == BAD_CATEGORY else FLAMMABLE_RULES
    verdict = "качественный товар (не бан)" if prediction == 1 else "некачественный товар (бан)"
    try:
        decision = apply_rules(text, category)
    except ValueError:
        decision = None

    if decision is not None and decision.label == prediction:
        terms = ", ".join(f"«{term}»" for term in decision.matched_terms) or "явных фрагментов нет"
        evidence = f"{decision.reason} Найденные фрагменты: {terms}."
    else:
        evidence = (
            "Детерминированный сигнал правил не подтверждает итог однозначно. "
            "Не придумывай детали; опирайся только на общий смысл текста карточки."
        )

    return (
        f"Категория проверки: {category}.\n"
        f"Зафиксированный итог классификатора: {verdict}.\n"
        f"Официальные правила: {rules}\n"
        f"Трассируемый сигнал: {evidence}\n"
        f"Карточка товара:\n{text}\n\n"
        "Напиши на русском один фактический комментарий длиной от 50 до 300 символов. "
        "Не меняй итог, не упоминай модель, вероятность или процесс классификации. "
        "Не используй теги <комментарий> и <вердикт>."
    )

