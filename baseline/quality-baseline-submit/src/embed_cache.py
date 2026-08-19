"""Resumable per-id embedding cache for offline training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np


def embedding_path(cache_dir: Path, row_id: object) -> Path:
    return cache_dir / f"{row_id}.npy"


def cached_ids(cache_dir: Path) -> set[str]:
    if not cache_dir.is_dir():
        return set()
    return {path.stem for path in cache_dir.glob("*.npy")}


def save_embedding(cache_dir: Path, row_id: object, vector: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = embedding_path(cache_dir, row_id)
    np.save(path, np.asarray(vector, dtype=np.float32))
    return path


def load_cached_embeddings(
    ids: Sequence[object],
    cache_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings aligned to ids; returns matrix and list of ids actually loaded."""

    vectors: list[np.ndarray] = []
    loaded_ids: list[str] = []
    for row_id in ids:
        path = embedding_path(cache_dir, row_id)
        if not path.is_file():
            continue
        vectors.append(np.load(path))
        loaded_ids.append(str(row_id))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), loaded_ids
    return np.vstack(vectors).astype(np.float32), loaded_ids


def write_manifest(
    cache_dir: Path,
    *,
    data_csv: Path,
    images_path: Path,
    pixel_preset: str,
    embed_batch: int,
    rows_requested: int,
    rows_cached: int,
    newly_cached: int,
) -> dict:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_csv": str(data_csv),
        "images_path": str(images_path),
        "cache_dir": str(cache_dir),
        "pixel_preset": pixel_preset,
        "embed_batch": embed_batch,
        "rows_requested": int(rows_requested),
        "rows_cached": int(rows_cached),
        "rows_missing": int(rows_requested - rows_cached),
        "newly_cached_this_run": int(newly_cached),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
