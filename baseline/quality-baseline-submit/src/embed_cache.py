"""Resumable per-id embedding cache for offline training."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

LOCK_FILENAME = "cache.lock"


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ExclusivePidLock:
    """Single-process lock using exclusive file create + live PID check."""

    def __init__(self, cache_dir: Path) -> None:
        self.path = Path(cache_dir) / LOCK_FILENAME
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._reclaim_if_stale()
        payload = json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()})
        try:
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except FileExistsError as exc:
            existing = self.path.read_text(encoding="utf-8")
            raise RuntimeError(
                f"Another embedding cache is running (lock {self.path}): {existing}. "
                "Do not start a second GPU process."
            ) from exc
        self._owned = True

    def _reclaim_if_stale(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            self.path.unlink(missing_ok=True)
            return
        if pid_is_alive(pid):
            return
        self.path.unlink(missing_ok=True)

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> "ExclusivePidLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def embedding_path(cache_dir: Path, row_id: object) -> Path:
    return cache_dir / f"{row_id}.npy"


def cached_ids(cache_dir: Path) -> set[str]:
    if not cache_dir.is_dir():
        return set()
    return {path.stem for path in cache_dir.glob("*.npy")}


def select_uncached_ids(
    ordered_ids: Sequence[object],
    cached: set[str],
    max_new_rows: int | None,
) -> list[str]:
    """Skip already cached ids, then apply a bounded increment."""

    missing = [str(row_id) for row_id in ordered_ids if str(row_id) not in cached]
    if max_new_rows is None:
        return missing
    if max_new_rows < 0:
        raise ValueError("max_new_rows must be >= 0")
    return missing[:max_new_rows]


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
    extra: dict | None = None,
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
    if extra:
        manifest.update(extra)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
