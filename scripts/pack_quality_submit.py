#!/usr/bin/env python3
"""Pack inference-only Task 2 ZIP for ODS (gitignored under local_data/)."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_ROOT = REPO_ROOT / "baseline" / "quality-baseline-submit"
DEFAULT_OUT = REPO_ROOT / "local_data" / "quality-baseline-submit.zip"
MAX_ARCHIVE_BYTES = 5_000_000_000

# Inference-only allowlist. Training/cache/tests stay out of the ZIP.
ALLOWLIST = (
    "metadata.json",
    "run.py",
    "baseline_qwen3vl_bf16.joblib",
    "arseniy_bad_text_model.joblib",
    "mark_flammable_model.joblib",
    "src/__init__.py",
    "src/constants.py",
    "src/utils_data_prep.py",
    "src/utils_logreg.py",
    "src/utils_postprocess.py",
    "src/utils_embed_cuda.py",
    "src/utils_generate_cuda.py",
    "src/explanations.py",
    "src/rules.py",
    "src/bad_classifier.py",
    "src/flammable_classifier.py",
)

FORBIDDEN_NAMES = frozenset({"embed_cache.py"})


def pack_submit(src_root: Path, out_zip: Path) -> tuple[int, str]:
    missing = [name for name in ALLOWLIST if not (src_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing submit files: {missing}")

    extra_cache = src_root / "src" / "embed_cache.py"
    if extra_cache.is_file() and "src/embed_cache.py" in ALLOWLIST:
        raise RuntimeError("embed_cache.py must not be in the inference ZIP allowlist")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ALLOWLIST:
            if Path(name).name in FORBIDDEN_NAMES:
                raise RuntimeError(f"refusing to pack forbidden file: {name}")
            archive.write(src_root / name, arcname=name)

    size = out_zip.stat().st_size
    if size >= MAX_ARCHIVE_BYTES:
        raise RuntimeError(f"archive too large: {size} bytes (limit {MAX_ARCHIVE_BYTES})")

    digest = hashlib.sha256(out_zip.read_bytes()).hexdigest()
    names = zipfile.ZipFile(out_zip).namelist()
    if "metadata.json" not in names:
        raise RuntimeError("metadata.json missing at ZIP root")
    return size, digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack quality inference ZIP")
    parser.add_argument("--src", type=Path, default=SUBMIT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        size, digest = pack_submit(args.src, args.out)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    print(f"size_bytes={size}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
