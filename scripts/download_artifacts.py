#!/usr/bin/env python3
"""Download release-hosted checkpoints and the two frozen submission archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release" / "artifacts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ozon-ecup-reproducer"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 << 20)
    partial.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["models", "submissions", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    selected = manifest["assets"] if args.only == "all" else [
        asset for asset in manifest["assets"] if asset["kind"] == args.only
    ]

    for asset in selected:
        destination = ROOT / asset["destination"]
        if not destination.exists() or args.force:
            print(f"downloading {asset['name']} ...", flush=True)
            download(asset["url"], destination)
        actual = sha256(destination)
        if actual != asset["sha256"]:
            raise SystemExit(f"{destination}: expected {asset['sha256']}, got {actual}")
        print(f"OK {destination}")
        if asset.get("extract"):
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(ROOT)

    import subprocess
    if args.only in ("models", "all"):
        print("verifying final source trees ...")
        for solution in ("mix408_textneg", "final_safe_fwdgate"):
            subprocess.run(["python3", "scripts/submission.py", "sources", solution], cwd=ROOT, check=True)
    if args.only in ("submissions", "all"):
        subprocess.run(
            ["python3", "scripts/submission.py", "verify", "mix408_textneg",
             "submissions/submission_mix408_textneg.zip"], cwd=ROOT, check=True)
        subprocess.run(
            ["python3", "validate_submission.py",
             "submissions/rebuilt_submission_final_safe_fwdgate.zip",
             "--expect-models", "2", "--expect-ordered", "model1_nosym", "model2_nosym",
             "--require-fp16"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
