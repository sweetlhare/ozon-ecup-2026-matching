#!/usr/bin/env python3
"""Build the exact category-stratified holdout used by the final recipes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/llmval_pairs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/llmvalS_pairs.parquet"))
    parser.add_argument("--per-category", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    pairs = pd.read_parquet(args.input)
    rng = np.random.default_rng(args.seed)
    parts = []
    for _, group in pairs.groupby("category", sort=True):
        count = min(args.per_category, len(group))
        parts.append(group.iloc[rng.choice(len(group), count, replace=False)])
    sample = pd.concat(parts).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    print(f"wrote {args.output}: {len(sample)} rows, {sample.category.nunique()} categories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
