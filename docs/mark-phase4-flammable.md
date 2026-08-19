# Phase 4 — Flammable classifier (Mark)

Owner: **Марк Досков** · Category: **Легковоспламеняющиеся**

## Goal

Dedicated multimodal head for the flammable category: Qwen3-VL embeddings plus official rule features from `src/rules.py`, with leakage-safe grouped OOF validation and integration into `run.py`.

## Data contract

| Item | Value |
|------|-------|
| Category | `Легковоспламеняющиеся` |
| Rows (full train) | 5502 |
| Positives (`label=1`, не бан) | 198 (~3.6%) |
| Validation | `StratifiedGroupKFold`, 5 folds, seed 42 |
| Groups | normalized exact product `name` |

Label contract: `0` = бан / не качественный, `1` = не бан / качественный.

## Artifacts

| Path | Role |
|------|------|
| `baseline/quality-baseline-submit/src/flammable_classifier.py` | Model + predict helpers |
| `baseline/quality-baseline-submit/src/embed_cache.py` | Resumable per-id embedding cache |
| `baseline/quality-baseline-submit/mark_flammable_model.joblib` | Trained artifact (committed) |
| `scripts/cache_flammable_embeddings.py` | GPU embed cache builder |
| `scripts/train_flammable_classifier.py` | OOF training + reports |
| `reports/mark/phase4-flammable.json` | Metrics report |
| `reports/mark/phase4-flammable-errors.csv` | Error analysis |

Embedding cache (gitignored): `local_data/flammable_embed_cache/{id}.npy`

## Local workflow (D: only)

```powershell
. D:\E-CUP 2026\scripts\set_d_drive_env.ps1
cd D:\E-CUP 2026
.\.venv\Scripts\Activate.ps1

# Один процесс. Не запускать второй кэш параллельно.
# GPU fraction 0.65, batch 1, chunk 4, максимум 3 фото, за шаг не больше 150 новых строк.
python -u scripts/cache_flammable_embeddings.py `
  --data_csv D:\data.csv `
  --images_path D:\images `
  --cache_dir local_data/flammable_embed_cache `
  --pixel_preset S `
  --embed_batch 1 `
  --chunk_size 4 `
  --max_new_rows 150 `
  --max_images 3 `
  --gpu_mem_fraction 0.65

# 2. Train + OOF report
python -u scripts/train_flammable_classifier.py `
  --data_csv D:\data.csv `
  --cache_dir local_data/flammable_embed_cache `
  --oversample_positives
```

Lock-файл: `local_data/flammable_embed_cache/cache.lock`. Если процесс умер, следующий запуск снимет stale lock сам. Если lock живой — второй процесс **сразу выходит**, GPU не трогает.

**Не запускайте несколько `cache_flammable_embeddings.py` сразу** — раньше это забивало 8 GB VRAM и RAM.

## Inference integration

`run.py` loads `mark_flammable_model.joblib` when present and overrides predictions only for flammable rows. `classifier_source` becomes `mark_flammable_embed_rules`. Embeddings are computed once for the full batch (same as baseline); no second GPU embedding pass.

Fallback: multimodal baseline logreg per category if the artifact is missing.

## Baseline reference

50-row offline subset (`reports/baseline/phase0-baseline.json`): F1_огонь = **0.1176**.

Phase 4 OOF (19.08.2026): F1_огонь = **0.893** on **931** cached rows (all 198 positives + 733 negatives). Previous 0.908 was on 694 rows. Target ≥ 0.30 met. Resume bounded cache (`--max_new_rows 150`) toward 5502; do not run parallel jobs.

## Comparison modes (same outer folds)

1. **rules_only** — `apply_flammable_rules` labels
2. **embedding_logreg_only** — cached embeddings + balanced logreg
3. **embeddings + rules** — full `FlammableQualityClassifier`

Thresholds tuned on train-fold probabilities; global threshold on stacked OOF for reporting.
