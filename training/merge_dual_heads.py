#!/usr/bin/env python3
"""Interpolate the production and auxiliary heads of a dual-head checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True,
                        help="0=consensus production head, 1=human head")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be in [0, 1]")
    state = load_file(args.model, device="cpu")
    heads = load_file(args.heads, device="cpu")
    for suffix in ("weight", "bias"):
        key = f"classifier.{suffix}"
        human_key = f"human_classifier.{suffix}"
        if key not in state or key not in heads or human_key not in heads:
            raise SystemExit(f"missing head tensor for {suffix}")
        merged = torch.lerp(heads[key].float(), heads[human_key].float(), args.alpha)
        state[key] = merged.to(state[key].dtype).contiguous()
    metadata = {}
    with safe_open(args.model, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    save_file(state, args.out, metadata=metadata)


if __name__ == "__main__":
    main()
