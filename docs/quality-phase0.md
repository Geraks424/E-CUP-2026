# Phase 0 — Task 2 Quality Control: инфраструктура и baseline

**Owner:** Марк  
**Done:** команда может запускать пайплайн локально; зафиксирован offline-score (dry-run) и лимиты времени.

## Baseline contract

Каталог сабмита: `baseline/quality-baseline-submit/`

| Компонент | Значение |
|-----------|----------|
| Docker-образ | `odsai/ecup26-quality-baseline:1.0` (`metadata.json`) |
| Entry point | `python -u run.py` |
| Аргументы CLI | `--test_data_path <csv>`, `--output_path <submit.csv>` |
| Классификатор | `baseline_qwen3vl_bf16.joblib` (в корне submit, путь в `run.py`) |
| Модели (env) | `SHARED_MODELS_PATH` → по умолчанию `/shared_models` |
| Embedding | `Qwen/Qwen3-VL-Embedding-2B` |
| LLM | `Qwen/Qwen3.5-4B` |

### Входные данные

CSV с колонками: `id`, `name`, `category`, `description` (+ опционально `label` для offline-score).

Изображения: `images/<id>/` рядом с CSV (расширения `.jpg`, `.jpeg`, `.png`). Подготовка — `src/utils_data_prep.prepare_dataframe`.

### Формат выхода

CSV: `id`, `result`, где `result` =

```text
<комментарий>{50–300 символов}<вердикт>бан|не бан
```

Маппинг вердикта из baseline (`src/utils_postprocess.py`):

- label `1` / pred `1` → **не бан**
- label `0` / pred `0` → **бан**

### Метрика соревнования

Macro-averaged F1 по двум категориям (**БАД**, **Легковоспламеняющиеся**): для каждой категории `sklearn.metrics.f1_score(y_true, y_pred)`, затем среднее.

Offline-scoring: вердикт из `result` после `<вердикт>`, `не бан`→1, `бан`→0.

## Локальная настройка (Windows)

```powershell
cd "D:\E-CUP 2026"
pip install -r requirements-dev.txt
```

Данные **не в репозитории** (см. `.gitignore`):

- `D:\data.csv`
- `D:\images\` или распакованный `images.zip`

## Команды Phase 0

### Один скрипт (рекомендуется)

Dry-run без GPU/CUDA — проверка упаковки, формата и offline-score на подмножестве:

```powershell
.\scripts\check_quality_submission.ps1 -DataCsv "D:\data.csv" -SubsetSize 200
```

Параметры:

- `-ImagesDir` — если картинки не в `D:\images\`
- `-Mode full` — только напоминание про Docker+GPU (полный инференс здесь не запускается)

### По шагам

```powershell
# 1. Подмножество (stratified, seed=42) → local_data/quality_subset.csv
python scripts/prepare_quality_subset.py --data_csv D:\data.csv --size 200 --seed 42

# 2. Dry-run сабмит (плейсхолдер-комментарии, вердикт = label)
python scripts/smoke_quality_baseline.py `
  --input_csv local_data/quality_subset.csv `
  --output_csv local_data/smoke_submit.csv

# 3. Валидация формата
python scripts/validate_quality_submission.py `
  --input_csv local_data/quality_subset.csv `
  --submission_csv local_data/smoke_submit.csv

# 4. Offline macro F1 (на labeled subset)
python scripts/score_quality_offline.py `
  --input_csv local_data/quality_subset.csv `
  --submission_csv local_data/smoke_submit.csv `
  --report reports/baseline/phase0-baseline.json `
  --mode dry_run
```

## Dry-run vs полный прогон

| Режим | GPU | Модели | F1 |
|-------|-----|--------|-----|
| **dry-run** (`check_quality_submission.ps1`) | не нужен | не нужны | oracle по `label` (формат/infra), **не baseline-модель** |
| **full baseline** | CUDA H100-class | `SHARED_MODELS_PATH` | реальный baseline-score — после GPU-прогона |

Полный baseline:

```bash
docker run --gpus all \
  -e SHARED_MODELS_PATH=/shared_models \
  -v /path/to/models:/shared_models \
  -v /path/to/test:/data \
  odsai/ecup26-quality-baseline:1.0 \
  python -u run.py --test_data_path /data/test.csv --output_path /data/submit.csv
```

## Отчёты

| Файл | Содержание |
|------|------------|
| `reports/baseline/phase0-baseline.json` | offline macro F1, per-category, status |
| `reports/baseline/phase0-runtime.json` | лимиты Check 3 / Public 20 / Private 40 мин |

Текущий статус Phase 0: **awaiting_gpu** для реального baseline F1 и **awaiting_gpu_measurement** для тайминга.

## Ограничения

- `local_data/` и большие артефакты в `.gitignore` — не коммитить.
- Dry-run score ≈ 1.0 (вердикты = labels) — только проверка scoring-пайплайна.
- `run.py` использует `CLASSIFIER_PATH = "baseline_qwen3vl_bf16.joblib"` в корне submit — не менять без согласования с Docker.

## CI (далее)

Локальный аналог CI: `scripts/check_quality_submission.ps1` (можно обернуть в GHA на Windows runner или вызвать Python-скрипты напрямую).
