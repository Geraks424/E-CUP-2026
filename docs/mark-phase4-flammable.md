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

# 1. Cache embeddings (resume by id; RTX 3060 Ti profile)
python -u scripts/cache_flammable_embeddings.py `
  --data_csv D:\data.csv `
  --images_path D:\images `
  --cache_dir local_data/flammable_embed_cache `
  --pixel_preset S `
  --embed_batch 1

# 2. Train + OOF report
python -u scripts/train_flammable_classifier.py `
  --data_csv D:\data.csv `
  --cache_dir local_data/flammable_embed_cache `
  --oversample_positives

# 3. Unit tests (no extra GPU pass)
python -m unittest discover -s tests -p "test_*.py"
```

## Inference integration

`run.py` loads `mark_flammable_model.joblib` when present and overrides predictions only for flammable rows. `classifier_source` becomes `mark_flammable_embed_rules`. Embeddings are computed once for the full batch (same as baseline); no second GPU embedding pass.

Fallback: multimodal baseline logreg per category if the artifact is missing.

## Baseline reference

50-row offline subset (`reports/baseline/phase0-baseline.json`): F1_огонь = **0.1176**.

Phase 4 OOF (19.08.2026): F1_огонь = **0.908** on 694 cached rows (all 198 positives + 496 negatives). Target ≥ 0.30 met. Resume `cache_flammable_embeddings.py` to extend negative coverage toward 5502 rows.

## Comparison modes (same outer folds)

1. **rules_only** — `apply_flammable_rules` labels
2. **embedding_logreg_only** — cached embeddings + balanced logreg
3. **embeddings + rules** — full `FlammableQualityClassifier`

Thresholds tuned on train-fold probabilities; global threshold on stacked OOF for reporting.
