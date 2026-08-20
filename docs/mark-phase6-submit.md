# Phase 6 — Сборка inference-сабмита (Марк)

**Owner:** Марк  
**Статус:** пакет для ODS готов локально; **Check / Public не прогнан** (ручной шаг вне репозитория). Фаза 6 в `TEAM_PLAN.md` не закрыта.

## Что входит в ZIP

Скрипт: `python scripts/pack_quality_submit.py`  
Выход (gitignored): `local_data/quality-baseline-submit.zip`

Allowlist (корень архива = корень сабмита):

- `metadata.json` — `odsai/ecup26-quality-baseline:1.0`, `python -u run.py`
- `run.py`
- `baseline_qwen3vl_bf16.joblib`, `arseniy_bad_text_model.joblib`, `mark_flammable_model.joblib`
- `src/` inference: classifiers, rules, explanations, utils, `__init__.py`

**Не класть в ZIP:** `src/embed_cache.py`, тесты, отчёты, `data.csv`, `images/`, кэши `.npy`, веса моделей из `/shared_models`.

Лимит архива: &lt; 5 GB. SHA-256 печатается скриптом упаковки.

## Профили GPU

| Профиль | embed batch | LLM batch | pixel | comments | LLM `max_memory` |
|---------|-------------|-----------|-------|----------|------------------|
| **ODS H100 (production defaults)** | 128 | 64 | M | `llm` | **не задавать** |
| Локальный RTX 3060 Ti (8 GB) | 1 | 1 | S | `rules` для smoke | `QUALITY_LLM_MAX_MEMORY=7GiB` только если реально грузите LLM |

На H100 **нельзя** оставлять жёсткий cap `7GiB` — Qwen3.5-4B туда не влезает как полноценный инференс. Cap включается только через env `QUALITY_LLM_MAX_MEMORY`.

Локальный smoke (один процесс, D: env, ≤10 строк, без LLM):

```powershell
cd "D:\E-CUP 2026"
. .\scripts\set_d_drive_env.ps1
.\.venv\Scripts\Activate.ps1
python scripts/prepare_quality_subset.py --data_csv D:\data.csv --size 10 --seed 42
python baseline/quality-baseline-submit/run.py `
  --test_data_path local_data/quality_subset.csv `
  --images_path D:\images `
  --output_path local_data/phase6_smoke_submit.csv `
  --comments_mode rules --embed_batch 1 --llm_batch 1 --pixel_preset S
python scripts/validate_quality_submission.py `
  --input_csv local_data/quality_subset.csv `
  --submission_csv local_data/phase6_smoke_submit.csv
```

Опционально: `python scripts/score_quality_offline.py ...` если в subset есть `label`.

Docker inspect образа — если Docker доступен локально; иначе pending до ODS.

## Glue `run.py`

1. Embeddings (Qwen3-VL-Embedding-2B)  
2. Baseline logreg  
3. Голова БАД (Арсений)  
4. Голова огня (Марк)  
5. Комментарии LLM или `--comments_mode rules`  
6. `format_results` → CSV `id,result`

## Тесты без GPU

```powershell
python -m unittest tests.test_run_pipeline tests.test_postprocess tests.test_rules -v
```

## ODS Check / Public

Загрузка ZIP и фиксация public score — **вручную**. Не записывать выдуманный public F1 в доки. Кандидат в финальные 2 выбирается после серии сабмитов (фаза 6 Done в TEAM_PLAN).
