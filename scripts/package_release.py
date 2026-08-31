#!/usr/bin/env python3
"""Create deterministic GitHub Release archives for the large artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ZIP_TIME = (2026, 8, 31, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        while chunk := source.read(8 << 20):
            target.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "release/final_models.zip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}")

    files = []
    for path in sorted((ROOT / "models").glob("*/*/*")):
        if path.is_file():
            files.append((path, path.relative_to(ROOT).as_posix()))
    if not files:
        raise SystemExit("no model files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with zipfile.ZipFile(partial, "w", allowZip64=True) as archive:
        for path, name in files:
            add_file(archive, path, name)
    partial.replace(args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"SHA-256: {sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
