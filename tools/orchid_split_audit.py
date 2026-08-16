#!/usr/bin/env python3
"""Create a reproducible orchid split manifest and audit it for likely leakage.

This utility is deliberately conservative. It can prove exact duplicate files and
flag visually similar images, but it cannot prove that a split is specimen-disjoint.
Any candidate it emits must be reviewed by a person before the manifest is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = (
    "image_path",
    "genus_id",
    "species_name",
    "species_id",
    "split",
    "split_note",
)


def image_paths(dataset_root: Path) -> Iterable[tuple[Path, str, str]]:
    """Yield images in the required ``Genus/Species/image`` hierarchy."""
    for genus_dir in sorted(dataset_root.iterdir()):
        if not genus_dir.is_dir() or genus_dir.name.upper() == "UNLABELED":
            continue
        for species_dir in sorted(genus_dir.iterdir()):
            if not species_dir.is_dir() or species_dir.name.upper() == "UNLABELED":
                continue
            for path in sorted(species_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    yield path, genus_dir.name, species_dir.name


def allocate_splits(count: int, ratios: tuple[float, float, float]) -> list[str]:
    """Allocate a class while never pretending tiny classes appear in every split."""
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "val"]

    raw = [count * ratio for ratio in ratios]
    allocation = [int(value) for value in raw]
    allocation[0] = max(allocation[0], 1)
    while sum(allocation) < count:
        index = max(range(3), key=lambda item: raw[item] - allocation[item])
        allocation[index] += 1
    while sum(allocation) > count:
        index = max(range(3), key=lambda item: allocation[item])
        if allocation[index] > 1:
            allocation[index] -= 1
        else:
            break
    return [split for split, number in zip(SPLITS, allocation) for _ in range(number)]


def create_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}. Pass --overwrite to replace it.")
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    if not np.isclose(sum(args.ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    per_species: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for path, genus, species in image_paths(dataset_root):
        per_species[f"{genus}::{species}"].append((path, genus, species))

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    split_counts: dict[str, int] = defaultdict(int)
    tiny_classes: list[str] = []
    for species_id in sorted(per_species):
        samples = per_species[species_id]
        rng.shuffle(samples)
        assignments = allocate_splits(len(samples), tuple(args.ratios))
        if len(samples) < 3:
            tiny_classes.append(species_id)
        for (path, genus, species), split in zip(samples, assignments):
            split_counts[split] += 1
            rows.append(
                {
                    "image_path": path.relative_to(dataset_root).as_posix(),
                    "genus_id": genus,
                    "species_name": species,
                    "species_id": species_id,
                    "split": split,
                    "split_note": "tiny_class_not_all_splits" if len(samples) < 3 else "",
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "ratios": dict(zip(SPLITS, args.ratios)),
        "images": len(rows),
        "genera": len({row["genus_id"] for row in rows}),
        "species": len(per_species),
        "split_counts": dict(split_counts),
        "species_with_fewer_than_three_images": tiny_classes,
        "warning": "This is image-stratified, not specimen-disjoint. Run the audit and review candidates.",
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output} and {report_path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    """Return a 64-bit difference hash. It is a candidate generator, not proof."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def read_manifest(manifest: Path, dataset_root: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    missing = set(MANIFEST_COLUMNS) - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    for row in rows:
        row["absolute_path"] = str(dataset_root / row["image_path"])
    return rows


def fingerprint_row(row: dict[str, str]) -> dict[str, str]:
    row["dhash"] = dhash(Path(row["absolute_path"]))
    return row


def audit_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}. Pass --overwrite to replace it.")
    rows = read_manifest(manifest, dataset_root)

    failures: list[dict[str, str]] = []
    usable: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_rows = {executor.submit(fingerprint_row, row): row for row in rows}
        for index, future in enumerate(as_completed(future_rows), start=1):
            row = future_rows[future]
            try:
                usable.append(future.result())
            except (OSError, ValueError) as error:
                failures.append({"image_path": row["image_path"], "error": str(error)})
            if index % 500 == 0:
                print(f"Hashed visual fingerprints for {index}/{len(rows)} images...", flush=True)

    candidates: dict[tuple[str, str], dict[str, object]] = {}
    buckets: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in usable:
        label = row["species_id"] if args.same_species_only else "ALL_LABELS"
        value = int(row["dhash"])
        for band in range(8):
            buckets[(label, band, (value >> (band * 8)) & 0xFF)].append(row)

    def record(left: dict[str, str], right: dict[str, str], evidence: str, distance: int) -> None:
        if left["split"] == right["split"]:
            return
        first, second = sorted((left["image_path"], right["image_path"]))
        key = (first, second)
        current = candidates.get(key)
        if current is None or distance < int(current["dhash_distance"]):
            candidates[key] = {
                "image_path_a": first,
                "split_a": left["split"] if left["image_path"] == first else right["split"],
                "image_path_b": second,
                "split_b": right["split"] if right["image_path"] == second else left["split"],
                "species_id_a": left["species_id"] if left["image_path"] == first else right["species_id"],
                "species_id_b": right["species_id"] if right["image_path"] == second else left["species_id"],
                "evidence": evidence,
                "dhash_distance": distance,
                "review_status": "pending_human_review",
            }

    skipped_buckets = 0
    for group in buckets.values():
        if len(group) > args.max_bucket_size:
            skipped_buckets += 1
            continue
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                distance = hamming(int(left["dhash"]), int(right["dhash"]))
                if distance <= args.dhash_threshold:
                    record(left, right, "near_duplicate_dhash", distance)

    # Full-file hashes are intentionally delayed until a visual candidate has
    # been found. Hashing every image makes a 50k-image audit unnecessarily
    # expensive, while identical files always share the same dHash buckets.
    sha_cache: dict[str, str] = {}
    for candidate in candidates.values():
        first = str(candidate["image_path_a"])
        second = str(candidate["image_path_b"])
        if first not in sha_cache:
            sha_cache[first] = sha256(dataset_root / first)
        if second not in sha_cache:
            sha_cache[second] = sha256(dataset_root / second)
        if sha_cache[first] == sha_cache[second]:
            candidate["evidence"] = "exact_sha256_match"
            candidate["dhash_distance"] = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "image_path_a", "split_a", "image_path_b", "split_b", "species_id_a", "species_id_b",
        "evidence", "dhash_distance", "review_status",
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(candidates.values(), key=lambda row: (str(row["evidence"]), int(row["dhash_distance"]))))

    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "manifest": str(manifest),
        "dataset_root": str(dataset_root),
        "images_in_manifest": len(rows),
        "images_audited": len(usable),
        "images_unreadable": len(failures),
        "cross_split_candidates": len(candidates),
        "same_species_only": args.same_species_only,
        "dhash_threshold": args.dhash_threshold,
        "workers": args.workers,
        "skipped_large_buckets": skipped_buckets,
        "limitation": "dHash flags likely duplicates but misses many different-angle captures. Human review is required.",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        failure_path = output.with_name(output.stem + ".unreadable.csv")
        with failure_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_path", "error"))
            writer.writeheader()
            writer.writerows(failures)
    print(f"Wrote {output} with {len(candidates)} cross-split candidates")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-manifest", help="Create a deterministic image-stratified split manifest.")
    create.add_argument("--dataset-root", required=True, help="Directory organized as Genus/Species/image.")
    create.add_argument("--output", required=True, help="CSV manifest path.")
    create.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    create.add_argument("--seed", type=int, default=20260815)
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(handler=create_manifest)

    audit = commands.add_parser("audit", help="Flag cross-split exact and near-duplicate candidates.")
    audit.add_argument("--dataset-root", required=True)
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output", required=True, help="CSV path for human-review candidates.")
    audit.add_argument("--dhash-threshold", type=int, default=6, choices=range(0, 65))
    audit.add_argument("--max-bucket-size", type=int, default=200)
    audit.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Image-decoding worker threads.")
    audit.add_argument("--same-species-only", action=argparse.BooleanOptionalAction, default=True)
    audit.add_argument("--overwrite", action="store_true")
    audit.set_defaults(handler=audit_manifest)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
