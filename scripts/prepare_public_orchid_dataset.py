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
import ssl
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

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


_CERTIFI_CONTEXT: ssl.SSLContext | None = None
_CERTIFI_CONTEXT_LOCK = threading.Lock()
_CERTIFI_FALLBACK_NOTIFIED = False


def _certifi_context() -> ssl.SSLContext:
    global _CERTIFI_CONTEXT
    with _CERTIFI_CONTEXT_LOCK:
        if _CERTIFI_CONTEXT is None:
            import certifi

            _CERTIFI_CONTEXT = ssl.create_default_context(cafile=certifi.where())
        return _CERTIFI_CONTEXT


def _urlopen(url: str, *, timeout: int | None = None):
    """Open an HTTPS URL, reusing certifi after an initial verification failure."""
    global _CERTIFI_FALLBACK_NOTIFIED
    if _CERTIFI_CONTEXT is not None:
        return urllib.request.urlopen(url, timeout=timeout, context=_CERTIFI_CONTEXT)
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        context = _certifi_context()
        with _CERTIFI_CONTEXT_LOCK:
            if not _CERTIFI_FALLBACK_NOTIFIED:
                print("Default TLS verification failed; using certifi CA bundle.", file=sys.stderr)
                _CERTIFI_FALLBACK_NOTIFIED = True
        return urllib.request.urlopen(url, timeout=timeout, context=context)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _download_url(url: str, target: Path) -> None:
    """Download one file atomically while showing byte-level progress."""
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with _urlopen(url) as response, partial.open("wb") as stream:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            free_bytes = shutil.disk_usage(target.parent).free
            print(
                f"Download size: {_format_bytes(total_bytes)}; "
                f"free space at {target.parent}: {_format_bytes(free_bytes)}.",
                flush=True,
            )
            with tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {target.name}",
                dynamic_ncols=True,
            ) as progress:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    progress.update(len(chunk))
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def fetch_metadata(args: argparse.Namespace) -> None:
    destination = Path(args.metadata_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if args.snapshot:
        archive = destination / f"inaturalist-open-data-{args.snapshot}.tar.gz"
        if archive.exists() and not args.overwrite:
            raise FileExistsError(f"{archive} already exists; pass --overwrite to replace it.")
        url = _metadata_url("", args.snapshot)
        try:
            _download_url(url, archive)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(
                    f"Official metadata snapshot {args.snapshot!r} was not found: {url}. "
                    "Choose a dated snapshot currently published by iNaturalist."
                ) from error
            raise
        print(f"Extracting {archive.name}...", flush=True)
        shutil.unpack_archive(str(archive), str(destination))
        print(f"Fetched and extracted {archive}", flush=True)
        return
    for name in REQUIRED_METADATA:
        target = destination / name
        if target.exists() and not args.overwrite:
            print(f"Keeping existing {target}")
            continue
        _download_url(_metadata_url(name, None), target)
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


def _image_url_variants(url: str) -> tuple[str, ...]:
    """Return case variants for source extensions, which are case-sensitive on S3."""
    prefix, separator, extension = url.rpartition(".")
    if not separator:
        return (url,)
    return tuple(dict.fromkeys((url, f"{prefix}.{extension.upper()}")))


def _download_one(row: dict[str, str], root: Path, overwrite: bool) -> tuple[str, str, str]:
    target = root / row["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return row["relative_path"], "existing", sha256_file(target)
    partial = target.with_suffix(target.suffix + ".part")
    last_error: OSError | urllib.error.URLError | urllib.error.HTTPError | None = None
    for source_url in _image_url_variants(row["source_url"]):
        try:
            with _urlopen(source_url, timeout=60) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            partial.replace(target)
            return row["relative_path"], "downloaded", sha256_file(target)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
    return row["relative_path"], f"failed:{last_error}", ""


def download_images(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    root = Path(args.output_root or config["output_root"])
    manifest = root / config["outputs"]["manifest_directory"] / config["outputs"]["images"]
    if not manifest.is_file():
        raise FileNotFoundError(f"Build the manifest first: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    results: list[tuple[str, str, str]] = []
    print(
        f"Preparing {len(rows):,} image downloads; free space at {root}: "
        f"{_format_bytes(shutil.disk_usage(root).free)}.",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download_one, row, root, args.overwrite) for row in rows]
        with tqdm(total=len(rows), desc="Downloading images", unit=" image", dynamic_ncols=True) as progress:
            for future in as_completed(futures):
                results.append(future.result())
                progress.update()
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
