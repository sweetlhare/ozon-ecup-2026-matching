#!/usr/bin/env python3
"""Download and verify the official competition parquet files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_pipeline" / "raw_data.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    base = manifest["base_url"]
    for name, expected in manifest["files"].items():
        destination = output / name
        if destination.exists():
            actual = sha256(destination)
            if actual != expected["sha256"]:
                raise SystemExit(
                    f"{destination}: expected {expected['sha256']}, got {actual}")
            print(f"OK {destination}")
            continue

        partial = destination.with_suffix(destination.suffix + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "ozon-ecup-reproducer"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(f"{base}/{name}", headers=headers)
        print(f"downloading {name} from byte {offset} ...", flush=True)
        with urllib.request.urlopen(request) as response:
            append = offset > 0 and response.status == 206
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=8 << 20)

        if partial.stat().st_size != expected["size"]:
            raise SystemExit(
                f"{partial}: expected {expected['size']} bytes, "
                f"got {partial.stat().st_size}")
        actual = sha256(partial)
        if actual != expected["sha256"]:
            raise SystemExit(
                f"{partial}: expected {expected['sha256']}, got {actual}")
        partial.replace(destination)
        print(f"OK {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
