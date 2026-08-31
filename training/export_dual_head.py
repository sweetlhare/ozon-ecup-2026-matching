#!/usr/bin/env python3
"""Export a dual-head checkpoint to the production FP16 model plus both heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


HEAD_KEYS = (
    "classifier.weight", "classifier.bias",
    "human_classifier.weight", "human_classifier.bias",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--heads-out", type=Path, required=True)
    args = parser.parse_args()
    source_weights = args.source / "model.safetensors"
    state = load_file(source_weights, device="cpu")
    missing = [key for key in HEAD_KEYS if key not in state]
    if missing:
        raise SystemExit(f"dual-head checkpoint misses: {missing}")

    heads = {key: state[key].float().contiguous() for key in HEAD_KEYS}
    production = {
        key: value.half().contiguous()
        for key, value in state.items() if not key.startswith("human_classifier.")
    }
    args.model_out.mkdir(parents=True, exist_ok=False)
    for path in args.source.iterdir():
        if path.is_file() and path.name != "model.safetensors":
            shutil.copy2(path, args.model_out / path.name)
    metadata = {}
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    save_file(production, args.model_out / "model.safetensors", metadata=metadata)
    save_file(heads, args.heads_out, metadata={"format": "pt"})
    marker = {
        "source": str(args.source),
        "production_dtype": "float16",
        "heads_dtype": "float32",
        "head_keys": list(HEAD_KEYS),
    }
    (args.model_out / "DUAL_HEAD_EXPORT.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
