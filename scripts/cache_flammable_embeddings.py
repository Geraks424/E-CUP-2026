#!/usr/bin/env python3
"""Resumable multimodal embedding cache for flammable-category training rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(_BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BASELINE_ROOT))

from src.constants import PIXEL_PRESETS  # noqa: E402
from src.embed_cache import (  # noqa: E402
    cached_ids,
    embedding_path,
    save_embedding,
    write_manifest,
)
from src.rules import FLAMMABLE_CATEGORY  # noqa: E402
from src.utils_data_prep import prepare_dataframe  # noqa: E402
from src.utils_embed_cuda import embed_dataframe_chunks_cuda  # noqa: E402


def cache_flammable_embeddings(
    data_csv: Path,
    images_path: Path,
    cache_dir: Path,
    *,
    embed_model_path: str,
    pixel_preset: str = "S",
    embed_batch: int = 1,
    limit: int | None = None,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_csv)
    flame = frame.loc[frame["category"].astype(str).eq(FLAMMABLE_CATEGORY)].copy()
    if limit is not None:
        flame = flame.head(limit).copy()

    cached = cached_ids(cache_dir)
    missing = flame.loc[~flame["id"].astype(str).isin(cached)].copy()
    missing = missing.sort_values(["label", "id"], ascending=[False, True]).reset_index(drop=True)
    if missing.empty:
        rows_cached = len([row_id for row_id in flame["id"].astype(str).tolist() if row_id in cached])
        return write_manifest(
            cache_dir,
            data_csv=data_csv,
            images_path=images_path,
            pixel_preset=pixel_preset,
            embed_batch=embed_batch,
            rows_requested=len(flame),
            rows_cached=rows_cached,
            newly_cached=0,
        )

    temp_csv = cache_dir / "_missing_flame_rows.csv"
    missing.to_csv(temp_csv, index=False)
    prepared = prepare_dataframe(temp_csv, images_path).reset_index(drop=True)

    newly_cached = 0
    if len(prepared) > 0:
        max_pixels = PIXEL_PRESETS[pixel_preset]
        print(
            f"Embedding {len(prepared)} missing rows in chunks (batch_size={embed_batch})",
            flush=True,
        )
        for chunk_start, chunk_end, embeddings in embed_dataframe_chunks_cuda(
            embed_model_path,
            prepared,
            max_pixels=max_pixels,
            batch_size=embed_batch,
            chunk_size=32,
        ):
            chunk = prepared.iloc[chunk_start:chunk_end]
            for row_idx, row_id in enumerate(chunk["id"].tolist()):
                save_embedding(cache_dir, row_id, embeddings[row_idx])
                newly_cached += 1
            print(
                f"Cached {newly_cached}/{len(missing)} missing embeddings "
                f"(total flame rows={len(flame)})",
                flush=True,
            )
    temp_csv.unlink(missing_ok=True)

    available = cached_ids(cache_dir)
    all_ids = flame["id"].astype(str).tolist()
    rows_cached = sum(1 for row_id in all_ids if row_id in available)
    return write_manifest(
        cache_dir,
        data_csv=data_csv,
        images_path=images_path,
        pixel_preset=pixel_preset,
        embed_batch=embed_batch,
        rows_requested=len(flame),
        rows_cached=rows_cached,
        newly_cached=newly_cached,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_csv", type=Path, default=Path("D:/data.csv"))
    parser.add_argument("--images_path", type=Path, default=Path("D:/images"))
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("local_data/flammable_embed_cache"),
    )
    parser.add_argument(
        "--embed_model_path",
        type=str,
        default=None,
        help="Path to Qwen3-VL embedding model (defaults to SHARED_MODELS_PATH)",
    )
    parser.add_argument("--pixel_preset", choices=tuple(PIXEL_PRESETS.keys()), default="S")
    parser.add_argument("--embed_batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import os

    shared = os.environ.get("SHARED_MODELS_PATH", "D:/E-CUP 2026/shared_models")
    embed_model_path = args.embed_model_path or os.path.join(
        shared,
        "Qwen/Qwen3-VL-Embedding-2B",
    )

    if not args.data_csv.is_file():
        print(f"ERROR: data CSV not found: {args.data_csv}", file=sys.stderr)
        return 1

    manifest = cache_flammable_embeddings(
        args.data_csv,
        args.images_path,
        args.cache_dir,
        embed_model_path=embed_model_path,
        pixel_preset=args.pixel_preset,
        embed_batch=args.embed_batch,
        limit=args.limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["rows_missing"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
