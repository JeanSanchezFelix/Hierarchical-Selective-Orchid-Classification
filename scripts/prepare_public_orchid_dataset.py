#!/usr/bin/env python3
"""Build and download a bounded public iNaturalist Orchidaceae benchmark.

This tool never downloads an image unless the explicit ``download-images``
subcommand is used. ``build-manifest`` creates an auditable 50k-image plan from
an official AWS metadata snapshot; it is safe to run before reserving storage.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_compression.src.orchid.public_dataset import (
    AWS_BUCKET_URL,
    REQUIRED_METADATA,
    read_config,
    select_from_aws_metadata,
    sha256_file,
    validate_manifests,
    write_manifests,
)


def _metadata_url(name: str, snapshot: str | None) -> str:
    if snapshot:
        return f"{AWS_BUCKET_URL}/metadata/inaturalist-open-data-{snapshot}.tar.gz"
    return f"{AWS_BUCKET_URL}/{name}"


def fetch_metadata(args: argparse.Namespace) -> None:
    destination = Path(args.metadata_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        archive = destination / f"inaturalist-open-data-{args.snapshot}.tar.gz"
        if archive.exists() and not args.overwrite:
            raise FileExistsError(f"{archive} already exists; pass --overwrite to replace it.")
        urllib.request.urlretrieve(_metadata_url("", args.snapshot), archive)
        shutil.unpack_archive(str(archive), str(destination))
        print(f"Fetched and extracted {archive}")
        return
    for name in REQUIRED_METADATA:
        target = destination / name
        if target.exists() and not args.overwrite:
            print(f"Keeping existing {target}")
            continue
        urllib.request.urlretrieve(_metadata_url(name, None), target)
        print(f"Fetched {target}")


def build_manifest(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    if args.output_root:
        config["output_root"] = args.output_root
    config["source"]["source_snapshot"] = args.source_snapshot
    records, summary = select_from_aws_metadata(args.metadata_dir, config)
    manifest_dir = write_manifests(config["output_root"], records, config, summary)
    verified = validate_manifests(manifest_dir, config)
    print(json.dumps({"manifest_dir": str(manifest_dir), **verified}, sort_keys=True))


def _download_one(row: dict[str, str], root: Path, overwrite: bool) -> tuple[str, str, str]:
    target = root / row["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return row["relative_path"], "existing", sha256_file(target)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(row["source_url"], timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(target)
        return row["relative_path"], "downloaded", sha256_file(target)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        partial.unlink(missing_ok=True)
        return row["relative_path"], f"failed:{error}", ""


def download_images(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    root = Path(args.output_root or config["output_root"])
    manifest = root / config["outputs"]["manifest_directory"] / config["outputs"]["images"]
    if not manifest.is_file():
        raise FileNotFoundError(f"Build the manifest first: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download_one, row, root, args.overwrite) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index == 1 or index % args.progress_every == 0:
                print(f"Processed {index}/{len(rows)} images", flush=True)
    checksums = root / config["outputs"]["manifest_directory"] / config["outputs"]["checksums"]
    with checksums.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("relative_path", "status", "sha256"))
        writer.writerows(sorted(results))
    failures = [result for result in results if result[1].startswith("failed:")]
    if failures:
        raise RuntimeError(f"{len(failures)} image downloads failed; rerun to resume. See {checksums}.")
    print(f"Downloaded or verified {len(results)} images; checksums: {checksums}")


def validate(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    root = Path(args.output_root or config["output_root"])
    result = validate_manifests(root / config["outputs"]["manifest_directory"], config)
    print(json.dumps(result, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, handler in (("fetch-metadata", fetch_metadata), ("build-manifest", build_manifest), ("download-images", download_images), ("validate", validate)):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/orchid/public_orchid_dataset.yaml")
        command.set_defaults(handler=handler)
    fetch = commands.choices["fetch-metadata"]
    fetch.add_argument("--metadata-dir", required=True)
    fetch.add_argument("--snapshot", help="Optional official snapshot date, such as 20241127.")
    fetch.add_argument("--overwrite", action="store_true")
    build = commands.choices["build-manifest"]
    build.add_argument("--metadata-dir", required=True)
    build.add_argument("--source-snapshot", required=True, help="Immutable official metadata snapshot ID or retrieval date.")
    build.add_argument("--output-root")
    download = commands.choices["download-images"]
    download.add_argument("--output-root")
    download.add_argument("--workers", type=int, default=8)
    download.add_argument("--progress-every", type=int, default=500)
    download.add_argument("--overwrite", action="store_true")
    validate_command = commands.choices["validate"]
    validate_command.add_argument("--output-root")
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
