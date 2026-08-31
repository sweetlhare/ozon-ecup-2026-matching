#!/usr/bin/env python3
"""Score the human training split with an ensemble for margin distillation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from split import make_split
from text import item_texts


class Batches(Dataset):
    def __init__(self, left, right, order, batch_size, tokenizer, max_length):
        self.left = left
        self.right = right
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batches = [order[start:start + batch_size]
                        for start in range(0, len(order), batch_size)]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        rows = self.batches[index]
        encoded = self.tokenizer(
            [self.left[row] for row in rows],
            [self.right[row] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded, rows


def first(value):
    return value[0]


@torch.inference_mode()
def score(left, right, model_path, max_length, batch_size, workers):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda().eval()
    order = np.argsort([len(a) + len(b) for a, b in zip(left, right)])
    result = np.empty(len(left), dtype=np.float32)
    loader = DataLoader(
        Batches(left, right, order, batch_size, tokenizer, max_length),
        batch_size=1,
        num_workers=workers,
        collate_fn=first,
        pin_memory=True,
        prefetch_factor=4 if workers else None,
    )
    for encoded, rows in loader:
        encoded = {name: value.cuda(non_blocking=True)
                   for name, value in encoded.items()}
        result[rows] = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.data / "matches.parquet")
    items = pd.read_parquet(args.data / "items_human.parquet")
    category_of = items.set_index("id").category
    is_validation = make_split(pairs, pairs.id1.map(category_of))
    pairs = pairs.loc[~is_validation].reset_index(drop=True)

    ordered_modes = {Path(model).joinpath("ORDERED").is_file()
                     for model in args.models}
    representations = {}
    for ordered in ordered_modes:
        lookup = dict(zip(items.id, item_texts(items, order=ordered)))
        representations[ordered] = (
            [lookup.get(item_id, "") for item_id in pairs.id1],
            [lookup.get(item_id, "") for item_id in pairs.id2],
        )

    scores = {}
    for model_path in args.models:
        ordered = Path(model_path).joinpath("ORDERED").is_file()
        left, right = representations[ordered]
        values = score(left, right, model_path, args.max_len,
                       args.batch, args.workers)
        name = Path(model_path).name.removeprefix("ce_")
        scores[name] = (values - values.mean()) / (values.std() + 1e-6)

    output = pd.DataFrame({"id1": pairs.id1, "id2": pairs.id2})
    for name, values in scores.items():
        output[f"s_{name}"] = values
    output["teacher"] = np.mean(list(scores.values()), axis=0)
    temporary = args.out.with_name(f"{args.out.name}.tmp-{os.getpid()}")
    try:
        output.to_parquet(temporary, index=False)
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.out}: {len(output):,} rows, {len(scores)} teachers")


if __name__ == "__main__":
    main()
