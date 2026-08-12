#!/usr/bin/env python3
"""Download Qwen models into SHARED_MODELS_PATH (D: layout for local runs)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = (
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3.5-4B",
)


def main() -> int:
    root = os.environ.get("SHARED_MODELS_PATH")
    if not root:
        print("ERROR: set SHARED_MODELS_PATH (e.g. D:\\E-CUP 2026\\shared_models)", file=sys.stderr)
        return 1

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)

    for repo_id in MODELS:
        dest = base / repo_id
        print(f"Downloading {repo_id} -> {dest}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        print(f"  OK: {dest}")

    print("All models ready for local_files_only inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
