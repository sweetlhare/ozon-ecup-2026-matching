#!/usr/bin/env python3
"""Build the two retrieval-shaped LLM pools used by foundation training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    from .text import item_texts
except ImportError:
    from text import item_texts


def candidate_pairs(matches: pd.DataFrame) -> pd.DataFrame:
    """Keep edges touching an item that has at least one positive candidate."""
    positive = matches.target.to_numpy() > 0.5
    ids = pd.concat([matches.id1, matches.id2], ignore_index=True)
    codes, _ = pd.factorize(ids, sort=False)
    n = len(matches)
    left, right = codes[:n], codes[n:]
    has_positive = np.zeros(codes.max() + 1, dtype=bool)
    has_positive[left[positive]] = True
    has_positive[right[positive]] = True
    return matches.loc[has_positive[left] | has_positive[right]].reset_index(drop=True)


def gray_sample(matches: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """Historical 6M pool: all ambiguous rows plus a fixed sample of endpoints."""
    rng = np.random.default_rng(seed)
    target = matches.target.to_numpy()
    gray = np.where((target > 0) & (target < 1))[0]
    edge = np.where((target == 0) | (target == 1))[0]
    n_edge = min(len(edge), max(n_pairs - len(gray), n_pairs // 4))
    gray_n = min(len(gray), n_pairs - n_edge)
    index = np.concatenate([
        gray[rng.permutation(len(gray))[:gray_n]],
        rng.choice(edge, n_edge, replace=False),
    ])
    return matches.iloc[np.sort(index)].reset_index(drop=True)


def render_items(items_path: Path, needed_ids, output_path: Path) -> None:
    needed = set(np.asarray(needed_ids, dtype=np.int64).tolist())
    writer = None
    written = 0
    try:
        dataset = ds.dataset(str(items_path), format="parquet")
        for batch in dataset.to_batches(
                columns=["id", "name", "attributes", "category"],
                batch_size=200_000):
            frame = batch.to_pandas()
            frame = frame[frame.id.isin(needed)]
            if frame.empty:
                continue
            table = pa.table({
                "id": pa.array(frame.id.to_numpy(np.int64)),
                "text": pa.array(item_texts(frame).to_numpy()),
            })
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            written += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if written != len(needed):
        raise RuntimeError(f"rendered {written:,}/{len(needed):,} items")


def write_pool(pairs: pd.DataFrame, items_path: Path, prefix: Path) -> None:
    pair_path = Path(f"{prefix}_pairs.parquet")
    item_path = Path(f"{prefix}_items.parquet")
    pair_tmp = pair_path.with_name(f"{pair_path.name}.tmp-{os.getpid()}")
    item_tmp = item_path.with_name(f"{item_path.name}.tmp-{os.getpid()}")
    needed = pd.unique(pd.concat([pairs.id1, pairs.id2], ignore_index=True))
    try:
        pairs.to_parquet(pair_tmp, index=False)
        render_items(items_path, needed, item_tmp)
        pair_tmp.replace(pair_path)
        item_tmp.replace(item_path)
    finally:
        pair_tmp.unlink(missing_ok=True)
        item_tmp.unlink(missing_ok=True)
    print(f"{prefix.name}: {len(pairs):,} pairs, {len(needed):,} items")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    matches = pd.read_parquet(args.data / "matches_llm.parquet")
    original_candidates = candidate_pairs(matches)
    validation_ids = set(np.load(
        args.data / "llmval_items_ids.npy", allow_pickle=True).tolist())
    matches = matches.loc[
        ~(matches.id1.isin(validation_ids) | matches.id2.isin(validation_ids))
    ].reset_index(drop=True)
    candidates = candidate_pairs(matches)

    # llmcand was the original 60k foundation pool; llmcand2 was the larger
    # candidate-only pool used by the 220k scale runs.
    write_pool(gray_sample(original_candidates, 6_000_000, args.seed),
               args.data / "items.parquet", args.data / "llmcand")
    write_pool(candidates, args.data / "items.parquet", args.data / "llmcand2")


if __name__ == "__main__":
    main()
