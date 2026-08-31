#!/usr/bin/env python3
"""Rebuild the exact ORDERED LLM pool used by the closure pipeline."""

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
except ImportError:  # direct script execution
    from text import item_texts


PAIR_ROWS = 10_289_587
PER_CATEGORY = 525_000
SEED = 42


def select_pairs(matches, validation_ids, category_of):
    validation_ids = set(np.asarray(validation_ids, dtype=np.int64).tolist())
    keep = ~(matches.id1.isin(validation_ids) | matches.id2.isin(validation_ids))
    pairs = matches.loc[keep].reset_index(drop=True)
    categories = category_of.reindex(pairs.id1).astype(str).to_numpy()
    strong = pairs.target.to_numpy() >= 5 / 9 - 1e-6
    rng = np.random.default_rng(SEED)
    selected = []
    for category in np.unique(categories):
        rows = np.where(categories == category)[0]
        keep_strong = rows[strong[rows]][:PER_CATEGORY]
        rest = rows[~strong[rows]]
        need = max(0, PER_CATEGORY - len(keep_strong))
        rest = rng.permutation(rest)[:need] if need and len(rest) else rest[:0]
        selected.append(np.concatenate([keep_strong, rest]))
    output = pairs.iloc[np.sort(np.concatenate(selected))].reset_index(drop=True)
    if len(output) != PAIR_ROWS:
        raise RuntimeError(f"llmfull selection produced {len(output):,} pairs")
    return output


def render_items(items_path, needed_ids, output_path, batch_size=200_000):
    needed = set(np.asarray(needed_ids, dtype=np.int64).tolist())
    writer = None
    written = 0
    try:
        dataset = ds.dataset(str(items_path), format="parquet")
        for batch in dataset.to_batches(
                columns=["id", "name", "attributes", "category"],
                batch_size=batch_size):
            frame = batch.to_pandas()
            frame = frame[frame.id.isin(needed)]
            if not len(frame):
                continue
            table = pa.table({
                "id": pa.array(frame.id.to_numpy(np.int64)),
                "text": pa.array(item_texts(frame, order=True).to_numpy()),
            })
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            writer.write_table(table)
            written += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if written != len(needed):
        raise RuntimeError(f"rendered {written:,}/{len(needed):,} llmfull items")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out-prefix", type=Path,
                        default=Path("data/llmfull_ord"))
    args = parser.parse_args()
    data = args.data
    pairs = pd.read_parquet(data / "matches_llm.parquet",
                            columns=["id1", "id2", "target"])
    categories = pd.read_parquet(data / "items.parquet", columns=["id", "category"])
    category_of = categories.set_index("id").category
    validation_ids = np.load(data / "llmval_items_ids.npy", allow_pickle=True)
    selected = select_pairs(pairs, validation_ids, category_of)
    pair_path = Path(f"{args.out_prefix}_pairs.parquet")
    item_path = Path(f"{args.out_prefix}_items.parquet")
    pair_tmp = pair_path.with_name(f"{pair_path.name}.tmp-{os.getpid()}")
    item_tmp = item_path.with_name(f"{item_path.name}.tmp-{os.getpid()}")
    needed = pd.unique(pd.concat([selected.id1, selected.id2], ignore_index=True))
    try:
        selected.to_parquet(pair_tmp, index=False)
        render_items(data / "items.parquet", needed, item_tmp)
        pair_tmp.replace(pair_path)
        item_tmp.replace(item_path)
    finally:
        pair_tmp.unlink(missing_ok=True)
        item_tmp.unlink(missing_ok=True)
    print(f"llmfull_ord={len(selected):,} pairs; {len(needed):,} items")


if __name__ == "__main__":
    main()
