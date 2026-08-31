#!/usr/bin/env python3
"""Build exact transitive hard negatives and matched positives for mix408."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_llm_targets import target_to_k
from text import item_texts


EXPECTED_SHA256 = "a96c8a68ddbcef473d022b1c89fa0591dd1e14dc967a052aa52b366d9f91595f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/anti_train.parquet"))
    parser.add_argument("--per-category", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    rng = np.random.default_rng(args.seed)
    pairs = pd.read_parquet(args.data / "matches_llm.parquet", columns=["id1", "id2", "target"])
    levels = target_to_k(pairs.target.to_numpy())
    codes, unique_ids = pd.factorize(np.concatenate([pairs.id1.to_numpy(), pairs.id2.to_numpy()]))
    node_count = np.int64(len(unique_ids))
    left = codes[:len(pairs)].astype(np.int64)
    right = codes[len(pairs):].astype(np.int64)

    def adjacency(mask):
        source = np.concatenate([left[mask], right[mask]])
        target = np.concatenate([right[mask], left[mask]])
        order = np.argsort(source, kind="stable")
        source, target = source[order], target[order]
        starts = np.searchsorted(source, np.arange(node_count), "left")
        ends = np.searchsorted(source, np.arange(node_count), "right")
        return target, starts, ends, ends - starts

    positive_neighbours, positive_starts, positive_ends, positive_degree = adjacency(levels == 9)
    negative_neighbours, negative_starts, negative_ends, negative_degree = adjacency(levels == 0)
    centers = np.where((positive_degree > 0) & (negative_degree > 0))[0]

    def pack(x, y):
        return np.minimum(x, y) * node_count + np.maximum(x, y)

    seen = np.sort(pack(left, right))
    items = pd.read_parquet(args.data / "items.parquet", columns=["id", "category"])
    categories = pd.Series(items.category.astype(str).values, index=items.id.astype("int64").values)
    ids = unique_ids.astype("int64")
    rows = []
    rng.shuffle(centers)
    for center in centers:
        if positive_degree[center] > 8 or negative_degree[center] > 40:
            continue
        positives = positive_neighbours[positive_starts[center]:positive_ends[center]]
        negatives = negative_neighbours[negative_starts[center]:negative_ends[center]]
        for x in positives[:3]:
            for y in negatives[:6]:
                if x == y:
                    continue
                key = pack(np.int64(x), np.int64(y))
                position = np.searchsorted(seen, key)
                if position < len(seen) and seen[position] == key:
                    continue
                rows.append((ids[x], ids[y]))
        if len(rows) > 400000:
            break

    negatives = pd.DataFrame(rows, columns=["id1", "id2"]).drop_duplicates()
    negatives["category"] = categories.reindex(negatives.id1.astype("int64")).values
    negatives = negatives[negatives.category.notna()]
    negatives["target"] = 0.0

    positives = pairs[levels == 9].copy()
    positives["category"] = categories.reindex(positives.id1.astype("int64")).values
    positives = positives[positives.category.notna()]
    positives["target"] = 1.0

    parts = []
    for category, group in negatives.groupby("category"):
        take = min(args.per_category, len(group))
        parts.append(group.iloc[rng.permutation(len(group))[:take]])
        matching = positives[positives.category == category]
        if len(matching):
            parts.append(matching.iloc[rng.permutation(len(matching))[:take]][["id1", "id2", "category", "target"]])
    sample = pd.concat(parts, ignore_index=True)

    needed = set(sample.id1.astype("int64")) | set(sample.id2.astype("int64"))
    raw = pd.read_parquet(args.data / "items.parquet")
    selected = raw[raw.id.astype("int64").isin(needed)]
    texts = pd.Series(np.asarray(item_texts(selected, order=True)), index=selected.id.astype("int64").values)
    sample["text1"] = texts.reindex(sample.id1.astype("int64")).values
    sample["text2"] = texts.reindex(sample.id2.astype("int64")).values
    sample = sample[sample.text1.notna() & sample.text2.notna()].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    actual = sha256(args.output)
    print(f"wrote {args.output}: {len(sample)} rows")
    print(f"SHA-256: {actual}")
    if args.per_category == 2000 and args.seed == 20260829 and actual != EXPECTED_SHA256:
        raise SystemExit(f"exact-build hash mismatch; expected {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
