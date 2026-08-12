#!/usr/bin/env python3
"""Deterministic stratified sample from competition CSV for local Phase 0 runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("id", "name", "category", "description")
OPTIONAL_COLUMNS = ("label",)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _find_images_for_id(product_id, images_dir: Path) -> list[str]:
    img_dir = images_dir / str(product_id)
    if not img_dir.is_dir():
        return []
    return [
        str(img_dir / name)
        for name in sorted(img_dir.iterdir())
        if name.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def _stratified_sample(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if "label" not in df.columns:
        return df.sample(n=size, random_state=seed).reset_index(drop=True)

    groups = []
    strat_col = df["category"].astype(str) + "__" + df["label"].astype(str)
    for _, group in df.groupby(strat_col, sort=False):
        n = max(1, round(size * len(group) / len(df)))
        groups.append(group.sample(n=min(n, len(group)), random_state=seed))

    sampled = pd.concat(groups, ignore_index=True)
    if len(sampled) > size:
        sampled = sampled.sample(n=size, random_state=seed).reset_index(drop=True)
    elif len(sampled) < size:
        remaining = df[~df["id"].isin(sampled["id"])]
        extra = remaining.sample(
            n=min(size - len(sampled), len(remaining)),
            random_state=seed + 1,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)
    return sampled.sort_values("id").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare stratified local subset CSV")
    parser.add_argument("--data_csv", type=Path, required=True, help="Source data.csv path")
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=None,
        help="Optional images root (default: <data_csv.parent>/images)",
    )
    parser.add_argument("--size", type=int, default=200, help="Subset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_data/quality_subset.csv"),
        help="Output CSV path (gitignored local path)",
    )
    args = parser.parse_args()

    if not args.data_csv.is_file():
        print(f"ERROR: data CSV not found: {args.data_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.data_csv)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: missing required columns: {missing_cols}", file=sys.stderr)
        return 1

    images_dir = args.images_dir or (args.data_csv.parent / "images")
    if images_dir.is_dir():
        image_counts = df["id"].apply(lambda pid: len(_find_images_for_id(pid, images_dir)))
        missing_images = int((image_counts == 0).sum())
        print(
            f"Images dir: {images_dir} | rows without images: {missing_images}/{len(df)}"
        )
    else:
        print(f"WARNING: images dir not found: {images_dir}", file=sys.stderr)

    subset = _stratified_sample(df, args.size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, index=False)

    print(f"Wrote subset: {args.output} ({len(subset)} rows)")
    if "label" in subset.columns:
        print(subset.groupby(["category", "label"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
