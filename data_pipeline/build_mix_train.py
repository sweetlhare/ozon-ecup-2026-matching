#!/usr/bin/env python3
"""Rebuild the exact shuffled soft + anti training mixture."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd


EXPECTED_SHA256 = "7a7e8c7efa54a2cfcfc16fd30babde5d5d50dc95b8a2cac6862703d39a24cd67"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", type=Path, default=Path("data/soft_train.parquet"))
    parser.add_argument("--anti", type=Path, default=Path("data/anti_train.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/mix_train.parquet"))
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    columns = ["text1", "text2", "target"]
    soft = pd.read_parquet(args.soft)[columns]
    anti = pd.read_parquet(args.anti)[columns]
    mixed = pd.concat([soft, anti], ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=11).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {len(mixed)} rows")
    actual = sha256(args.output)
    print(f"SHA-256: {actual}")
    if actual != EXPECTED_SHA256:
        raise SystemExit(f"exact-build hash mismatch; expected {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

