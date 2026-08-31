#!/usr/bin/env python3
"""Build and verify the two frozen finalist submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = ROOT / "solutions"
FIXED_ZIP_TIME = (2026, 8, 31, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(solution: str) -> tuple[Path, dict]:
    solution_dir = SOLUTIONS / solution
    manifest_path = solution_dir / "manifest.json"
    if not manifest_path.is_file():
        available = ", ".join(
            sorted(path.name for path in SOLUTIONS.iterdir() if path.is_dir())
        )
        raise SystemExit(f"unknown solution {solution!r}; available: {available}")
    return solution_dir, json.loads(manifest_path.read_text())


def source_map(solution_dir: Path, manifest: dict) -> dict[str, Path]:
    sources: dict[str, Path] = {}

    for name in manifest["runtime_files"]:
        sources[name] = solution_dir / "runtime" / name

    for model in manifest["models"]:
        archive_dir = model["archive_dir"]
        source_dir = ROOT / model["source_dir"]
        for name in model["files"]:
            sources[f"{archive_dir}/{name}"] = source_dir / name
        marker_dir = solution_dir / "markers" / archive_dir
        if marker_dir.is_dir():
            for marker in sorted(marker_dir.iterdir()):
                if marker.is_file():
                    sources[f"{archive_dir}/{marker.name}"] = marker

    for asset in manifest.get("assets", []):
        sources[asset["archive_name"]] = ROOT / asset["source"]

    return sources


def verify_sources(solution_dir: Path, manifest: dict) -> dict[str, Path]:
    sources = source_map(solution_dir, manifest)
    expected = manifest["content_sha256"]
    if set(sources) != set(expected):
        missing = sorted(set(expected) - set(sources))
        extra = sorted(set(sources) - set(expected))
        raise SystemExit(
            f"manifest/source mismatch; missing={missing}, extra={extra}"
        )

    errors = []
    for archive_name, source in sorted(sources.items()):
        if not source.is_file():
            errors.append(f"missing {source}")
            continue
        actual = sha256_file(source)
        if actual != expected[archive_name]:
            errors.append(
                f"{archive_name}: expected {expected[archive_name]}, "
                f"got {actual} ({source})"
            )
    if errors:
        raise SystemExit("source verification failed:\n  " + "\n  ".join(errors))
    return sources


def verify_archive(path: Path, manifest: dict, exact_outer_sha: bool) -> None:
    if not path.is_file():
        raise SystemExit(f"archive not found: {path}")
    if exact_outer_sha:
        actual_outer = sha256_file(path)
        expected_outer = manifest["submitted_archive_sha256"]
        if actual_outer != expected_outer:
            raise SystemExit(
                f"archive SHA-256 mismatch: expected {expected_outer}, "
                f"got {actual_outer}"
            )

    expected = manifest["content_sha256"]
    errors = []
    with zipfile.ZipFile(path) as archive:
        regular = {info.filename for info in archive.infolist() if not info.is_dir()}
        if regular != set(expected):
            errors.append(
                f"member set mismatch; missing={sorted(set(expected) - regular)}, "
                f"extra={sorted(regular - set(expected))}"
            )
        for name in sorted(regular & set(expected)):
            actual = sha256_zip_member(archive, name)
            if actual != expected[name]:
                errors.append(f"{name}: expected {expected[name]}, got {actual}")
    if errors:
        raise SystemExit("archive verification failed:\n  " + "\n  ".join(errors))


def add_directory(archive: zipfile.ZipFile, name: str) -> None:
    info = zipfile.ZipInfo(name.rstrip("/") + "/", FIXED_ZIP_TIME)
    info.external_attr = (stat.S_IFDIR | 0o755) << 16
    archive.writestr(info, b"")


def add_file(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    info = zipfile.ZipInfo(archive_name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with source.open("rb") as input_handle, archive.open(
        info, "w", force_zip64=True
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 << 20)


def build(solution_dir: Path, manifest: dict, output: Path, force: bool,
          trained: bool = False) -> None:
    sources = source_map(solution_dir, manifest) if trained \
        else verify_sources(solution_dir, manifest)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing submission sources:\n  " + "\n  ".join(missing))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force explicitly")

    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for model in manifest["models"]:
                add_directory(archive, model["archive_dir"])
            for archive_name, source in sorted(sources.items()):
                add_file(archive, source, archive_name)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    if not trained:
        verify_archive(output, manifest, exact_outer_sha=False)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "validate_submission.py"),
            str(output),
            *manifest["validator_args"],
        ],
        check=True,
    )
    print(f"built: {output}")
    print(f"SHA-256: {sha256_file(output)}")
    print("content: freshly trained checkpoints" if trained
          else "content: matches frozen manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")
    for command in ("sources", "verify", "build"):
        child = subparsers.add_parser(command)
        child.add_argument("solution")
        if command == "verify":
            child.add_argument("archive", type=Path)
        elif command == "build":
            child.add_argument("--output", type=Path)
            child.add_argument("--force", action="store_true")
            child.add_argument("--trained", action="store_true",
                               help="package freshly trained checkpoints")

    args = parser.parse_args()
    if args.command == "list":
        for path in sorted(SOLUTIONS.glob("*/manifest.json")):
            manifest = json.loads(path.read_text())
            print(
                f"{manifest['id']:<24} {manifest['role']:<24} "
                f"public={manifest['public_macro_ap']:.10f}"
            )
        return 0

    solution_dir, manifest = load_manifest(args.solution)
    if args.command == "sources":
        verify_sources(solution_dir, manifest)
        print(
            f"OK: {args.solution} sources match "
            f"{len(manifest['content_sha256'])} hashes"
        )
    elif args.command == "verify":
        verify_archive(args.archive.resolve(), manifest, exact_outer_sha=True)
        print(f"OK: exact submitted archive {args.archive.resolve()}")
    elif args.command == "build":
        output = args.output or ROOT / manifest["rebuild_output"]
        build(solution_dir, manifest, output, args.force, args.trained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
