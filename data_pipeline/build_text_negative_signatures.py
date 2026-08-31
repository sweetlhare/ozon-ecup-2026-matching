#!/usr/bin/env python3
"""Build the exact-input negative consensus used by the primary runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SIGNATURE_DTYPE = np.dtype([
    ("lo1", "<u8"), ("lo2", "<u8"),
    ("hi1", "<u8"), ("hi2", "<u8"),
])
HASH_KEYS = ("0123456789abcdef", "fedcba9876543210")


def item_hashes(items: pd.DataFrame):
    text = items.text.fillna("")
    first = pd.util.hash_pandas_object(
        text, index=False, hash_key=HASH_KEYS[0], categorize=True,
    ).to_numpy(np.uint64, copy=False)
    second = pd.util.hash_pandas_object(
        text, index=False, hash_key=HASH_KEYS[1], categorize=True,
    ).to_numpy(np.uint64, copy=False)
    return (pd.Series(first, index=items.id), pd.Series(second, index=items.id))


def pair_signatures(pairs: pd.DataFrame, hash1: pd.Series, hash2: pd.Series):
    left1 = hash1.reindex(pairs.id1).to_numpy(np.uint64)
    left2 = hash2.reindex(pairs.id1).to_numpy(np.uint64)
    right1 = hash1.reindex(pairs.id2).to_numpy(np.uint64)
    right2 = hash2.reindex(pairs.id2).to_numpy(np.uint64)
    left_first = (left1 < right1) | ((left1 == right1) & (left2 <= right2))
    result = np.empty(len(pairs), dtype=SIGNATURE_DTYPE)
    result["lo1"] = np.where(left_first, left1, right1)
    result["lo2"] = np.where(left_first, left2, right2)
    result["hi1"] = np.where(left_first, right1, left1)
    result["hi2"] = np.where(left_first, right2, left2)
    return result


def unanimous_repeated_negatives(signatures, target):
    order = np.argsort(signatures, kind="stable")
    sorted_signatures = signatures[order]
    boundaries = np.r_[True, sorted_signatures[1:] != sorted_signatures[:-1], True]
    starts = np.flatnonzero(boundaries[:-1])
    stops = np.flatnonzero(boundaries[1:]) + 1
    negative = np.asarray(target)[order] < (0.5 / 9.0)
    negative_count = np.add.reduceat(negative.astype(np.int64), starts)
    size = stops - starts
    return sorted_signatures[starts[(size >= 2) & (negative_count == size)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/textsig_negative_u64.npy"))
    args = parser.parse_args()

    pairs = pd.read_parquet(args.data / "llmfull_ord_pairs.parquet")
    items = pd.read_parquet(args.data / "llmfull_ord_items.parquet")
    hash1, hash2 = item_hashes(items)
    signatures = pair_signatures(pairs, hash1, hash2)
    selected = unanimous_repeated_negatives(signatures, pairs.target.to_numpy())
    output = selected.view("<u8").reshape(-1, 4)
    np.save(args.output, output)
    print(f"{args.output}: {len(output):,} unanimous repeated negatives")


if __name__ == "__main__":
    main()
