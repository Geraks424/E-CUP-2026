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

## Локальная настройка (Windows, только диск D:)

**Все артефакты Phase 0 хранятся на `D:`** — не использовать `C:\Users\...` для venv, кэшей HF/torch/pip, моделей или больших выходов.

| Назначение | Путь |
|------------|------|
| Репозиторий / venv | `D:\E-CUP 2026\`, `D:\E-CUP 2026\.venv` |
| Данные | `D:\data.csv`, `D:\images\` (из `D:\images.zip`) |
| Модели | `D:\E-CUP 2026\shared_models\` (`SHARED_MODELS_PATH`) |
| HF / torch / pip cache | `D:\E-CUP 2026\.cache\...` |
| Temp | `D:\E-CUP 2026\.tmp` |
| Subset / submit outputs | `D:\E-CUP 2026\local_data\` |

Перед любой GPU-командой:

```powershell
cd "D:\E-CUP 2026"
. .\scripts\set_d_drive_env.ps1   # HF_HOME, TMP, PIP_CACHE_DIR, SHARED_MODELS_PATH → D:
.\.venv\Scripts\Activate.ps1
```

Создание venv и зависимости (один раз):

```powershell
. .\scripts\set_d_drive_env.ps1
python -m venv "D:\E-CUP 2026\.venv"
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install transformers accelerate huggingface_hub pillow pandas numpy scikit-learn joblib
python scripts/download_shared_models.py
```

Проверка CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Данные **не в репозитории** (см. `.gitignore`):

- `D:\data.csv`
- `D:\images\` (распаковать `D:\images.zip` → `D:\images\<id>\*.jpg|png`, без вложенного `images/images/`)

Распаковка изображений:

```powershell
python -c "
import zipfile
from pathlib import Path
dest = Path(r'D:\images'); dest.mkdir(exist_ok=True)
with zipfile.ZipFile(r'D:\images.zip') as z:
    for m in z.namelist():
        if m.endswith('/'): continue
        rel = m[len('images/'):] if m.startswith('images/') else m
        if not rel: continue
        t = dest / rel; t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(z.read(m))
"
```

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
| **model** (локальный GPU) | CUDA (8 GB: RTX 3060 Ti) | `SHARED_MODELS_PATH` на D: | реальный baseline F1 (logreg + LLM) |
| **full baseline** (Docker H100) | CUDA H100-class | `/shared_models` в контейнере | эталон для лимитов времени |

### Профиль 8 GB (RTX 3060 Ti)

H100-дефолты в `run.py` не меняются (`embed_batch=128`, `llm_batch=64`, `pixel_preset=M`). Для 8 GB переопределяйте CLI или env:

| Параметр | H100 default | 8 GB local |
|----------|--------------|------------|
| `--embed_batch` / `EMBED_BATCH_SIZE` | 128 | 1–2 |
| `--llm_batch` / `LLM_BATCH_SIZE` | 64 | 1 |
| `--pixel_preset` / `PIXEL_PRESET` | M | S |

LLM использует `float16`, `device_map=auto`, `max_memory={0: 7GiB, cpu: 16GiB}` для offload при OOM.

### Локальный GPU baseline (subset)

CSV и `images/` должны быть **siblings**: `run.py` ищет `{test_data_path.parent}/images/`. Для `D:\data.csv` + `D:\images\`:

```powershell
. .\scripts\set_d_drive_env.ps1
cd baseline\quality-baseline-submit

# subset → D:\E-CUP 2026\local_data\quality_subset.csv; для inference положить CSV на D:\:
Copy-Item "D:\E-CUP 2026\local_data\quality_subset.csv" "D:\quality_subset.csv"

python ..\..\scripts\prepare_quality_subset.py --data_csv D:\data.csv --images_dir D:\images --size 50 --seed 42 `
  --output D:\E-CUP 2026\local_data\quality_subset.csv

# smoke 3 rows, then full subset
python -u run.py --test_data_path D:\quality_subset.csv --output_path D:\E-CUP 2026\local_data\model_submit.csv `
  --embed_batch 1 --llm_batch 1 --pixel_preset S

python ..\..\scripts\validate_quality_submission.py `
  --input_csv D:\quality_subset.csv `
  --submission_csv D:\E-CUP 2026\local_data\model_submit.csv

python ..\..\scripts\score_quality_offline.py `
  --input_csv D:\quality_subset.csv `
  --submission_csv D:\E-CUP 2026\local_data\model_submit.csv `
  --report reports\baseline\phase0-baseline.json `
  --mode model
```

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

Текущий статус Phase 0: после GPU-прогона на subset — `mode=model`, `status=completed` в `phase0-baseline.json`; dry-run F1≈1.0 остаётся только как проверка infra.

## Ограничения

- **Только диск D:** venv, `.cache/`, `.tmp/`, `shared_models/`, `local_data/` — всё под `D:\E-CUP 2026\`; данные `D:\data.csv`, `D:\images\`.
- `local_data/`, `.cache/`, `shared_models/` в `.gitignore` — не коммитить.
- Dry-run score ≈ 1.0 (вердикты = labels) — только проверка scoring-пайплайна, **не** primary baseline result.
- `run.py` использует `CLASSIFIER_PATH = "baseline_qwen3vl_bf16.joblib"` в корне submit — не менять без согласования с Docker.

## CI (далее)

Локальный аналог CI: `scripts/check_quality_submission.ps1` (можно обернуть в GHA на Windows runner или вызвать Python-скрипты напрямую).
