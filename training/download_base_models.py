#!/usr/bin/env python3
"""Download immutable snapshots of the three public foundation models."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("base_models.json")


def main() -> None:
    models = json.loads(MANIFEST.read_text())
    for name, model in models.items():
        destination = ROOT / model["local_dir"]
        print(f"{name}: {model['repo_id']}@{model['revision']}")
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            local_dir=destination,
        )


if __name__ == "__main__":
    main()
