#!/usr/bin/env python3
"""Build the item-novelty validation mask used for checkpoint selection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from training.split import make_split
except ImportError:
    from split import make_split


TOKEN = re.compile(r"[a-zа-я0-9]+")


def hashed_ngrams(texts, n_features=2 ** 18, ngram=3):
    indptr, indices, values = [0], [], []
    for text in texts:
        text = " ".join(TOKEN.findall(str(text).lower()))
        counts = {}
        for offset in range(max(len(text) - ngram + 1, 1)):
            bucket = hash(text[offset:offset + ngram]) % n_features
            counts[bucket] = counts.get(bucket, 0) + 1
        if not counts:
            counts = {0: 1}
        norm = np.sqrt(sum(value * value for value in counts.values()))
        indices.extend(counts)
        values.extend(value / norm for value in counts.values())
        indptr.append(len(indices))
    return csr_matrix(
        (np.asarray(values, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
        shape=(len(indptr) - 1, n_features),
    )


def max_similarity(val_ids, train_ids, texts, categories, block=256):
    result = {}
    for category in np.unique([categories[index] for index in val_ids]):
        val = [index for index in val_ids if categories[index] == category]
        train = [index for index in train_ids if categories[index] == category]
        if not train:
            result.update({index: 0.0 for index in val})
            continue
        val_matrix = hashed_ngrams([texts[index] for index in val])
        train_matrix = hashed_ngrams([texts[index] for index in train])
        for start in range(0, len(val), block):
            stop = min(start + block, len(val))
            similarities = (val_matrix[start:stop] @ train_matrix.T).toarray()
            for row, index in enumerate(val[start:stop]):
                result[index] = float(similarities[row].max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    items = pd.read_parquet(
        args.data / "items_human.parquet",
        columns=["id", "name", "attributes", "category"],
    )
    pairs = pd.read_parquet(args.data / "matches.parquet")
    category_of = items.set_index("id").category
    is_validation = make_split(pairs, pairs.id1.map(category_of))

    position = {item_id: index for index, item_id in enumerate(items.id.tolist())}
    texts = (items.name.astype(str) + " " + items.attributes.astype(str)).tolist()
    categories = items.category.astype(str).to_numpy()
    val_rows = np.where(is_validation)[0]
    train_rows = np.where(~is_validation)[0]
    val_ids = sorted({position[item_id] for item_id in pairs.id1.iloc[val_rows]} |
                     {position[item_id] for item_id in pairs.id2.iloc[val_rows]})
    train_ids = sorted({position[item_id] for item_id in pairs.id1.iloc[train_rows]} |
                       {position[item_id] for item_id in pairs.id2.iloc[train_rows]})
    similarity = max_similarity(val_ids, train_ids, texts, categories)
    output = np.asarray(
        [[index, similarity.get(index, 0.0)] for index in val_ids],
        dtype=np.float32,
    )
    np.save(args.data / "val_maxsim.npy", output)
    print(f"val_maxsim.npy: {len(output):,} validation items")


if __name__ == "__main__":
    main()
