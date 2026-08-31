#!/usr/bin/env python3
"""Build the exact 80k soft-label corpus used by both final primary models."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_llm_targets import target_to_k
from text import item_texts


EXPECTED_SHA256 = "d969792a4eb7cc21b867f48d79bdffa96622a7f2d59acf7e9521774f9689bfa2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/soft_train.parquet"))
    parser.add_argument("--per-category", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=31337)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    pairs = pd.read_parquet(args.data / "matches_llm.parquet", columns=["id1", "id2", "target"])
    holdout = pd.read_parquet(args.data / "llmvalS_pairs.parquet", columns=["id1", "id2"])
    banned = set(holdout.id1.astype("int64")) | set(holdout.id2.astype("int64"))
    keep = ~(pairs.id1.astype("int64").isin(banned) | pairs.id2.astype("int64").isin(banned)).to_numpy()
    pairs = pairs[keep].reset_index(drop=True)

    items = pd.read_parquet(args.data / "items.parquet", columns=["id", "category"])
    categories = pd.Series(items.category.astype(str).values, index=items.id.astype("int64").values)
    pairs["category"] = categories.reindex(pairs.id1.astype("int64")).values
    pairs = pairs[pairs.category.notna()].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    parts = []
    for _, group in pairs.groupby("category"):
        take = min(args.per_category, len(group))
        parts.append(group.iloc[rng.permutation(len(group))[:take]])
    sample = pd.concat(parts, ignore_index=True)

    needed = set(sample.id1.astype("int64")) | set(sample.id2.astype("int64"))
    raw = pd.read_parquet(args.data / "items.parquet")
    selected = raw[raw.id.astype("int64").isin(needed)]
    texts = pd.Series(np.asarray(item_texts(selected, order=True)), index=selected.id.astype("int64").values)
    sample["text1"] = texts.reindex(sample.id1.astype("int64")).values
    sample["text2"] = texts.reindex(sample.id2.astype("int64")).values
    if not sample.text1.notna().all() or not sample.text2.notna().all():
        raise RuntimeError("some item texts are missing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output, index=False)
    actual = sha256(args.output)
    print(f"wrote {args.output}: {len(sample)} rows")
    print(f"SHA-256: {actual}")
    if args.per_category == 4000 and args.seed == 31337 and actual != EXPECTED_SHA256:
        raise SystemExit(f"exact-build hash mismatch; expected {EXPECTED_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
