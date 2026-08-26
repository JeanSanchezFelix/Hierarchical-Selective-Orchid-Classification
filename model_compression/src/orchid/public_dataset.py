"""Deterministic preparation of a bounded public iNaturalist orchid dataset."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from tqdm import tqdm


AWS_BUCKET_URL = "https://inaturalist-open-data.s3.amazonaws.com"
REQUIRED_METADATA = ("observations.csv.gz", "photos.csv.gz", "taxa.csv.gz")


@dataclass(frozen=True)
class PublicImageRecord:
    observation_id: str
    photo_id: str
    extension: str
    photo_license: str
    observer_id: str
    taxon_id: str
    genus_id: str
    species_name: str
    source_taxon_name: str
    relative_path: str
    source_url: str

    @property
    def species_id(self) -> str:
        return f"{self.genus_id}::{self.species_name}"


def read_config(path: str | Path) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if config.get("source", {}).get("provider") != "inaturalist-licensed-observation-images":
        raise ValueError("This preparer requires the iNaturalist Licensed Observation Images source.")
    if int(config.get("selection", {}).get("target_images", 0)) <= 0:
        raise ValueError("selection.target_images must be positive.")
    if int(config.get("selection", {}).get("target_species", 0)) <= 0:
        raise ValueError("selection.target_species must be positive.")
    ratios = config.get("splits", {})
    total = sum(float(ratios.get(name, 0.0)) for name in ("train", "validation", "calibration", "test"))
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train, validation, calibration, and test proportions must sum to 1.0.")
    return config


def _first(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_metadata_files(directory: Path) -> dict[str, Path]:
    """Find required metadata in either raw-download or extracted-archive layouts."""
    locations = [directory, *(path for path in directory.iterdir() if path.is_dir())]
    for location in locations:
        files: dict[str, Path] = {}
        for required in REQUIRED_METADATA:
            uncompressed = required.removesuffix(".gz")
            candidate = next(
                (path for path in (location / required, location / uncompressed) if path.is_file()),
                None,
            )
            if candidate is None:
                break
            files[required] = candidate
        if len(files) == len(REQUIRED_METADATA):
            return files
    raise FileNotFoundError(f"Missing official metadata file(s): {', '.join(REQUIRED_METADATA)}")


def _read_rows(path: Path) -> Iterator[dict[str, str]]:
    """Stream CSV rows while reporting the current phase and row throughput."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        dialect = csv.excel_tab if "\t" in sample else csv.excel
        rows = csv.DictReader(stream, dialect=dialect)
        with tqdm(desc=f"Reading {path.name}", unit=" rows", dynamic_ncols=True) as progress:
            for row in rows:
                progress.update()
                yield row


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _normalize_license(value: str) -> str:
    return value.strip().casefold().replace("_", "-")


def _is_descendant(ancestry: str, family_taxon_id: str) -> bool:
    tokens = {token for token in re.split(r"[,/| ]+", ancestry) if token}
    return family_taxon_id in tokens


def _safe_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Unsafe taxonomy component: {value!r}")
    return cleaned


def _taxon_parts(name: str) -> tuple[str, str]:
    parts = name.split()
    if len(parts) < 2:
        raise ValueError(f"Species-level source taxon must be binomial: {name!r}")
    return _safe_component(parts[0]), _safe_component(" ".join(parts[1:]))


def _allocate_splits(records: list[PublicImageRecord], ratios: Mapping[str, float], seed: int) -> dict[str, str]:
    split_names = ("train", "validation", "calibration", "test")
    by_species: dict[str, list[PublicImageRecord]] = defaultdict(list)
    for record in records:
        by_species[record.species_id].append(record)
    assignments: dict[str, str] = {}
    for species_id, group in by_species.items():
        # The selected corpus contains one image per observation. Keeping this
        # key makes the invariant explicit and survives a future policy change.
        observation_groups: dict[str, list[PublicImageRecord]] = defaultdict(list)
        for record in group:
            observation_groups[record.observation_id].append(record)
        ordered = sorted(
            observation_groups.items(), key=lambda item: _stable_key(seed, f"{species_id}:{item[0]}")
        )
        total = len(ordered)
        raw = [total * float(ratios[name]) for name in split_names]
        counts = [int(value) for value in raw]
        while sum(counts) < total:
            index = max(range(len(counts)), key=lambda item: raw[item] - counts[item])
            counts[index] += 1
        cursor = 0
        for split, count in zip(split_names, counts):
            for _, observation_records in ordered[cursor : cursor + count]:
                for record in observation_records:
                    assignments[record.photo_id] = split
            cursor += count
    return assignments


def select_from_aws_metadata(metadata_dir: str | Path, config: Mapping[str, object]) -> tuple[list[PublicImageRecord], dict[str, int]]:
    """Select bounded records from the official iNaturalist metadata snapshot.

    The source files are intentionally read as streams. Only Orchidaceae
    species-level research-grade observations are retained in memory.
    """
    directory = Path(metadata_dir)
    metadata_files = _resolve_metadata_files(directory)

    source = config["source"]
    selection = config["selection"]
    family_name = str(source["family_name"]).casefold()
    accepted_licenses = {_normalize_license(value) for value in source["accepted_photo_licenses"]}
    taxa: dict[str, tuple[str, str, str]] = {}
    family_id = ""
    for row in _read_rows(metadata_files["taxa.csv.gz"]):
        taxon_id = _first(row, "taxon_id", "id")
        name = _first(row, "name")
        rank = _first(row, "rank").casefold()
        ancestry = _first(row, "ancestry")
        if taxon_id and name:
            taxa[taxon_id] = (name, rank, ancestry)
        if name.casefold() == family_name and rank == "family":
            family_id = taxon_id
    if not family_id:
        raise ValueError(f"Could not find family {source['family_name']!r} in taxa.csv.gz.")

    eligible_taxa: dict[str, tuple[str, str]] = {}
    skipped_nonbinomial_taxa = 0
    for taxon_id, (name, rank, ancestry) in taxa.items():
        if rank != "species" or not _is_descendant(ancestry, family_id):
            continue
        try:
            eligible_taxa[taxon_id] = _taxon_parts(name)
        except ValueError:
            skipped_nonbinomial_taxa += 1

    observations: dict[str, tuple[str, str, str, str]] = {}
    for row in _read_rows(metadata_files["observations.csv.gz"]):
        observation_id = _first(row, "observation_uuid", "observation_id", "id")
        taxon_id = _first(row, "taxon_id")
        quality_grade = _first(row, "quality_grade").casefold()
        if observation_id and taxon_id in eligible_taxa and quality_grade == str(source["quality_grade"]).casefold():
            observer_id = _first(row, "observer_id", "user_id")
            observations[observation_id] = (*eligible_taxa[taxon_id], taxon_id, observer_id)

    # Retain the primary compatible photo for each observation. The source may
    # contain several photos; one photo prevents within-observation leakage.
    chosen: dict[str, tuple[int, str, str, str, str]] = {}
    for row in _read_rows(metadata_files["photos.csv.gz"]):
        observation_id = _first(row, "observation_uuid", "observation_id")
        if observation_id not in observations:
            continue
        license_id = _normalize_license(_first(row, "license", "photo_license"))
        if license_id not in accepted_licenses:
            continue
        photo_id = _first(row, "photo_id", "id")
        extension = _first(row, "extension", "file_extension").lstrip(".").casefold()
        if not photo_id or not extension:
            continue
        try:
            position = int(_first(row, "position") or "0")
        except ValueError:
            position = 0
        photo_observer = _first(row, "observer_id", "user_id")
        current = chosen.get(observation_id)
        candidate = (position, photo_id, extension, license_id, photo_observer)
        if current is None or candidate[:2] < current[:2]:
            chosen[observation_id] = candidate

    candidates: dict[str, list[PublicImageRecord]] = defaultdict(list)
    for observation_id, (position, photo_id, extension, license_id, photo_observer) in chosen.items():
        genus, species, taxon_id, observation_observer = observations[observation_id]
        observer_id = photo_observer or observation_observer or "unknown"
        filename = f"{observation_id}_{photo_id}.{extension}"
        relative_path = Path(genus) / species / filename
        record = PublicImageRecord(
            observation_id=observation_id,
            photo_id=photo_id,
            extension=extension,
            photo_license=license_id,
            observer_id=observer_id,
            taxon_id=taxon_id,
            genus_id=genus,
            species_name=species,
            source_taxon_name=f"{genus} {species}",
            relative_path=relative_path.as_posix(),
            source_url=f"{AWS_BUCKET_URL}/photos/{photo_id}/{selection['image_size']}.{extension}",
        )
        candidates[record.species_id].append(record)

    minimum = int(selection["minimum_images_per_species"])
    maximum = int(selection["maximum_images_per_species"])
    observer_cap = int(selection["maximum_images_per_observer_per_species"])
    seed = int(selection["deterministic_selection_seed"])
    capped_candidates: dict[str, list[PublicImageRecord]] = {}
    for species_id, rows in candidates.items():
        per_observer: Counter[str] = Counter()
        kept: list[PublicImageRecord] = []
        for record in sorted(rows, key=lambda value: _stable_key(seed, value.photo_id)):
            if per_observer[record.observer_id] >= observer_cap:
                continue
            kept.append(record)
            per_observer[record.observer_id] += 1
            if len(kept) == maximum:
                break
        capped_candidates[species_id] = kept

    eligible = [species_id for species_id, rows in capped_candidates.items() if len(rows) >= minimum]
    selected_species = sorted(
        eligible,
        key=lambda value: (-len(capped_candidates[value]), _stable_key(seed, f"species:{value}")),
    )[: int(selection["target_species"])]
    target_species = int(selection["target_species"])
    if len(selected_species) < target_species:
        raise ValueError(
            f"Only {len(selected_species)} species satisfy the minimum of {minimum} after applying the observer cap."
        )

    selected = [record for species_id in selected_species for record in capped_candidates[species_id]]
    target = int(selection["target_images"])
    print(
        f"Selection capacity after observer cap: {len(selected):,} images across "
        f"{target_species:,} species (target: {target:,}).",
        flush=True,
    )
    if len(selected) < target:
        raise ValueError(
            f"The {target_species} highest-capacity species provide only {len(selected)} images, "
            f"below target_images={target}."
        )
    if len(selected) > target:
        # Preserve every selected species while trimming only the global excess
        # deterministically. The minimum per species remains guaranteed.
        minimum_total = minimum * len(selected_species)
        if target < minimum_total:
            raise ValueError("target_images is smaller than the required species minimum total.")
        guaranteed: list[PublicImageRecord] = []
        extras: list[PublicImageRecord] = []
        for species_id in selected_species:
            ordered = sorted(capped_candidates[species_id], key=lambda value: _stable_key(seed, value.photo_id))
            guaranteed.extend(ordered[:minimum])
            extras.extend(ordered[minimum:])
        selected = guaranteed + sorted(extras, key=lambda value: _stable_key(seed, f"image:{value.photo_id}"))[
            : target - len(guaranteed)
        ]
        counts = Counter(record.species_id for record in selected)
        if any(count < minimum for count in counts.values()) or len(counts) != len(selected_species):
            raise ValueError("Global trimming violated the per-species minimum; increase target_images.")

    return sorted(selected, key=lambda value: value.relative_path), {
        "family_taxon_id": family_id,
        "skipped_nonbinomial_taxa": skipped_nonbinomial_taxa,
        "candidate_species": len(eligible),
        "selected_species": len(selected_species),
        "selected_images": len(selected),
    }


def write_manifests(output_root: str | Path, records: Iterable[PublicImageRecord], config: Mapping[str, object], source_summary: Mapping[str, object]) -> Path:
    root = Path(output_root)
    outputs = config["outputs"]
    manifest_dir = root / str(outputs["manifest_directory"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    assignments = _allocate_splits(records, config["splits"], int(config["splits"]["seed"]))

    image_fields = list(PublicImageRecord.__dataclass_fields__) + ["species_id", "split"]
    image_rows = [{**asdict(record), "species_id": record.species_id, "split": assignments[record.photo_id]} for record in records]
    with (manifest_dir / str(outputs["images"])).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(image_rows)

    observation_fields = ("observation_id", "genus_id", "species_name", "species_id", "taxon_id", "observer_id")
    with (manifest_dir / str(outputs["observations"])).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=observation_fields)
        writer.writeheader()
        writer.writerows({
            "observation_id": record.observation_id,
            "genus_id": record.genus_id,
            "species_name": record.species_name,
            "species_id": record.species_id,
            "taxon_id": record.taxon_id,
            "observer_id": record.observer_id,
        } for record in records)

    split_fields = ("image_path", "observation_id", "genus_id", "species_name", "species_id", "split", "split_note")
    with (manifest_dir / str(outputs["split"])).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=split_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_path": record.relative_path,
                "observation_id": record.observation_id,
                "genus_id": record.genus_id,
                "species_name": record.species_name,
                "species_id": record.species_id,
                "split": assignments[record.photo_id],
                "split_note": "observation_disjoint",
            })

    taxonomy = defaultdict(int)
    for record in records:
        taxonomy[(record.genus_id, record.species_name, record.species_id)] += 1
    (manifest_dir / str(outputs["taxonomy"])).write_text(json.dumps({
        "schema_version": 1,
        "records": [
            {"genus_id": genus, "species_name": species, "species_id": species_id, "image_count": count}
            for (genus, species, species_id), count in sorted(taxonomy.items())
        ],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    licenses = Counter(record.photo_license for record in records)
    with (manifest_dir / str(outputs["licenses"])).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("photo_license", "image_count"))
        writer.writeheader()
        writer.writerows({"photo_license": license_id, "image_count": count} for license_id, count in sorted(licenses.items()))

    # This version rejects records during source filtering rather than retaining
    # their potentially massive raw metadata. The empty file is an explicit
    # schema promise for a later rejected-record reason report.
    with (manifest_dir / str(outputs["rejected_records"])).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_id", "reason"))
        writer.writeheader()

    checksums = {path.name: sha256_file(path) for path in manifest_dir.iterdir() if path.is_file()}
    card = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "source": config["source"],
        "selection": config["selection"],
        "splits": config["splits"],
        "source_summary": dict(source_summary),
        "actual_images": len(records),
        "actual_species": len(taxonomy),
        "actual_genera": len({record.genus_id for record in records}),
        "manifest_sha256": checksums,
        "download_status": "manifest_only",
    }
    (manifest_dir / str(outputs["dataset_card"])).write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_dir


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifests(manifest_dir: str | Path, config: Mapping[str, object]) -> dict[str, object]:
    directory = Path(manifest_dir)
    images_path = directory / str(config["outputs"]["images"])
    split_path = directory / str(config["outputs"]["split"])
    with images_path.open(newline="", encoding="utf-8") as stream:
        images = list(csv.DictReader(stream))
    with split_path.open(newline="", encoding="utf-8") as stream:
        splits = list(csv.DictReader(stream))
    if len(images) != len(splits):
        raise ValueError("images and split manifests have different row counts.")
    expected_licenses = {_normalize_license(value) for value in config["source"]["accepted_photo_licenses"]}
    if any(_normalize_license(row["photo_license"]) not in expected_licenses for row in images):
        raise ValueError("Manifest includes an unapproved photo license.")
    split_by_path = {row["image_path"]: row for row in splits}
    if len(split_by_path) != len(splits):
        raise ValueError("Split manifest has duplicate image paths.")
    observation_splits: dict[str, set[str]] = defaultdict(set)
    for row in splits:
        observation_splits[row["observation_id"]].add(row["split"])
    leaked = [observation for observation, values in observation_splits.items() if len(values) != 1]
    if leaked:
        raise ValueError(f"Observation leakage across splits: {leaked[:3]}")
    species = Counter(row["species_id"] for row in images)
    minimum = int(config["selection"]["minimum_images_per_species"])
    if len(species) != int(config["selection"]["target_species"]):
        raise ValueError("Selected species count does not match the configured target.")
    if min(species.values()) < minimum:
        raise ValueError("At least one species violates minimum_images_per_species.")
    return {"images": len(images), "species": len(species), "genera": len({row['genus_id'] for row in images})}
