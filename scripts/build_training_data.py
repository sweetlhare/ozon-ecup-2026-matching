#!/usr/bin/env python3
"""Build the final training datasets from the organizer parquet files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "soft_train.parquet": "d969792a4eb7cc21b867f48d79bdffa96622a7f2d59acf7e9521774f9689bfa2",
    "anti_train.parquet": "a96c8a68ddbcef473d022b1c89fa0591dd1e14dc967a052aa52b366d9f91595f",
    "mix_train.parquet": "7a7e8c7efa54a2cfcfc16fd30babde5d5d50dc95b8a2cac6862703d39a24cd67",
}
CLOSURE_EXPECTED = {
    "closure_consensus_pairs.parquet": {
        "rows": 20_400,
        "sha256": "5d6f20fa2c42d588e252bf546de92613b586e0ebce8e7dae9670af9e934f6e77",
    },
    "closure_consensus_items.parquet": {
        "rows": 38_195,
        "sha256": "dc6d66dc411560132b06fb3930dc6edfc11cd180b6fedf7ff9aa6c681405d362",
    },
    "closure_big_pairs.parquet": {"rows": 123_733},
    "closure_big_items.parquet": {"rows": 205_202},
    "closure_gapcats_pairs.parquet": {"rows": 3_600},
    "closure_gapcats_items.parquet": {"rows": 7_018},
    "closure_all20_pairs.parquet": {
        "rows": 24_000,
        "sha256": "8fb3d44fbe98d353fbc359d357e311a52eba2218ea638af5bb7880c7cfeca62d",
    },
    "closure_all20_items.parquet": {
        "rows": 45_213,
        "sha256": "0fc16a96ca9ea485e590a22f831e506e3f2a1d8f5f94369aadf1ac34d190c35f",
    },
}
LLMFULL_EXPECTED = {
    "llmfull_ord_pairs.parquet": {
        "rows": 10_289_587,
        "sha256": "db88b65eee0ec260bad35ef9ea87487d4330fe51b1d97d0989a40ece53b4b76c",
    },
    "llmfull_ord_items.parquet": {
        "rows": 11_451_316,
        "sha256": "a2eaefd54a55856451e098e315bd7980deddf0676c5a00eaa90af6a54cc42c18",
    },
}
FOUNDATION_DATA_EXPECTED = {
    "llmcand_pairs.parquet": 6_000_000,
    "llmcand2_pairs.parquet": 5_750_952,
}
RAW_MANIFEST = ROOT / "data_pipeline" / "raw_data.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "data_pipeline")
    env.setdefault("PYTHONHASHSEED", "0")
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true",
                        help="download the official parquet files before building")
    args = parser.parse_args()
    data = args.data.resolve()

    if args.download:
        run("scripts/download_data.py", "--output", str(data))

    required = ("items.parquet", "items_human.parquet",
                "matches.parquet", "matches_llm.parquet")
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        raise SystemExit(f"missing organizer files in {data}: {', '.join(missing)}")
    raw = json.loads(RAW_MANIFEST.read_text())["files"]
    for name in required:
        actual = sha256(data / name)
        expected = raw[name]["sha256"]
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected}, got {actual}")

    llmval_outputs = [
        data / "llmval_pairs.parquet", data / "llmval_items.parquet",
        data / "llmval_items_ids.npy", data / "llmval_comp_mask.npy",
        data / "llmvalS_pairs.parquet",
    ]
    if not all(path.is_file() for path in llmval_outputs):
        if any(path.exists() for path in llmval_outputs):
            raise SystemExit(
                "llmval data is incomplete; remove llmval* derived files and rerun")
        run("data_pipeline/make_llm_val.py", "--data", str(data),
            "--out_prefix", str(data / "llmval"))
        run("data_pipeline/build_llmval_sample.py",
            "--input", str(data / "llmval_pairs.parquet"),
            "--output", str(data / "llmvalS_pairs.parquet"))

    novelty_path = data / "val_maxsim.npy"
    if not novelty_path.is_file():
        run("data_pipeline/build_novelty_validation.py", "--data", str(data))

    foundation_paths = [
        data / "llmcand_pairs.parquet", data / "llmcand_items.parquet",
        data / "llmcand2_pairs.parquet", data / "llmcand2_items.parquet",
    ]
    if not all(path.is_file() for path in foundation_paths):
        if any(path.exists() for path in foundation_paths):
            raise SystemExit(
                "foundation data is incomplete; remove llmcand*_*.parquet and rerun")
        run("data_pipeline/build_candidate_pool.py", "--data", str(data))

    import pandas as pd
    for name, rows_expected in FOUNDATION_DATA_EXPECTED.items():
        rows = len(pd.read_parquet(data / name, columns=["id1"]))
        if rows != rows_expected:
            raise SystemExit(f"{name}: expected {rows_expected} rows, got {rows}")

    outputs = [data / name for name in EXPECTED]
    existing = [output for output in outputs if output.exists()]
    if existing:
        if len(existing) != len(outputs):
            raise SystemExit(
                "derived data is incomplete; remove the derived files and rerun")
        for name, expected in EXPECTED.items():
            actual = sha256(data / name)
            if actual != expected:
                raise SystemExit(f"{name}: expected {expected}, got {actual}")
        print("last-stage training data already exists and all SHA-256 values match")
    else:
        run("data_pipeline/build_soft_train.py", "--data", str(data),
            "--output", str(data / "soft_train.parquet"))
        run("data_pipeline/build_anti_train.py", "--data", str(data),
            "--output", str(data / "anti_train.parquet"))
        run("data_pipeline/build_mix_train.py",
            "--soft", str(data / "soft_train.parquet"),
            "--anti", str(data / "anti_train.parquet"),
            "--output", str(data / "mix_train.parquet"))

    llmfull_paths = [data / name for name in LLMFULL_EXPECTED]
    if not all(path.is_file() for path in llmfull_paths):
        if any(path.exists() for path in llmfull_paths):
            raise SystemExit(
                "llmfull data is incomplete; remove llmfull_ord_* and rerun")
        run("data_pipeline/build_llmfull.py", "--data", str(data),
            "--out-prefix", str(data / "llmfull_ord"))

    for name, expected in LLMFULL_EXPECTED.items():
        path = data / name
        if "rows" in expected:
            id_column = "id1" if name.endswith("_pairs.parquet") else "id"
            rows = len(pd.read_parquet(path, columns=[id_column]))
            if rows != expected["rows"]:
                raise SystemExit(f"{name}: expected {expected['rows']} rows, got {rows}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise SystemExit(f"{name}: expected {expected['sha256']}, got {actual}")

    text_negative = data / "textsig_negative_u64.npy"
    if not text_negative.is_file():
        run("data_pipeline/build_text_negative_signatures.py",
            "--data", str(data), "--output", str(text_negative))
    expected_text_negative = (
        "613bdf2703daa1c3d7c80429b2961c35b2d8f10f3dbd9db349ec6957f8f0df73"
    )
    actual = sha256(text_negative)
    if actual != expected_text_negative:
        raise SystemExit(
            f"textsig_negative_u64.npy: expected {expected_text_negative}, got {actual}")

    closure_paths = [data / name for name in CLOSURE_EXPECTED]
    if not all(path.is_file() for path in closure_paths):
        if any(path.exists() for path in closure_paths):
            raise SystemExit(
                "closure data is incomplete; remove closure_* derived files and rerun")
        run("data_pipeline/build_closure_data.py", "--data", str(data))

    for name, expected in CLOSURE_EXPECTED.items():
        path = data / name
        if not path.is_file():
            raise SystemExit(f"missing derived file: {path}")
        if "rows" in expected:
            id_column = "id1" if name.endswith("_pairs.parquet") else "id"
            rows = len(pd.read_parquet(path, columns=[id_column]))
            if rows != expected["rows"]:
                raise SystemExit(f"{name}: expected {expected['rows']} rows, got {rows}")
        if "sha256" in expected:
            actual = sha256(path)
            if actual != expected["sha256"]:
                raise SystemExit(f"{name}: expected {expected['sha256']}, got {actual}")

    print("\nSelected training data:")
    for name, expected in EXPECTED.items():
        path = data / name
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected}, got {actual}")
        print(f"  {name}: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
