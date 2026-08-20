# Phase 6 — Сборка inference-сабмита (Марк)

**Owner:** Марк  
**Статус:** пакет для ODS готов локально; **Check / Public не прогнан**. Локальный GPU-smoke **20.08.2026 пропущен** (на RTX 3060 Ti уже был `python.exe -u train_archive.py`, PID 4828, ~2682 MiB). Второй inference не запускали. Фаза 6 в `TEAM_PLAN.md` не закрыта.

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

## Evidence: локальный GPU-smoke (20.08.2026)

| Поле | Значение |
|------|----------|
| Дата | 2026-08-20 12:05 +08 |
| GPU | RTX 3060 Ti, driver 610.88, занято ~2682 / 8192 MiB |
| Решение | **skip** — competing `train_archive.py` (PID 4828) |
| Строк / comments | не запускалось |
| Отчёт | `reports/mark/phase6-smoke.json` |

Когда GPU свободна, выполнить команды выше (один процесс, D: env). Residual: повторить smoke, затем валидатор и при наличии `label` — `score_quality_offline.py --mode model` в `reports/mark/` (это **не** ODS score).

## ODS Check / Public (ручной чеклист Марка)

ZIP: `local_data/quality-baseline-submit.zip` (пересобрать `python scripts/pack_quality_submit.py` если менялся `run.py`). Не коммитить ZIP.

Заполнять **только факты с площадки**, без плейсхолдеров в git, пока нет реального прогона.

1. **Check** — загрузить ZIP, дождаться вердикта (лимит ~3 мин). Записать: дата, статус (pass/fail/timeout), wall time.  
   Check status: _pending_ · time: _—_ · notes: _—_
2. Если Check зелёный — **Public**. Записать: дата, macro F1, F1_БАД, F1_огонь, wall time (~20 мин лимит). При регрессии — не выбирать этот ZIP кандидатом.  
   Public macro F1: _pending_ · per-category: _—_
3. Передать фактические Check/Public числа **Владу** для фазы 7 (его трекер здесь не правим).
4. Фаза 6 Done в `TEAM_PLAN.md` — только после стабильного Public и выбора кандидата в финальные 2. **Сейчас: не Done.**
