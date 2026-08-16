"""Canonical taxonomy contracts for a ``Genus/Species/image`` orchid dataset.

Folder names remain the source of truth for private images.  The qualified species
identifier avoids collisions between identical epithets that occur in different
genera and is stable across router, expert, evaluation, and export artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .constants import TASK_FLAT_SPECIES, TASK_GENUS, TASK_GENUS_SPECIES, TASK_TARGET_GENUS


def canonical_species_id(genus_id: str, species_name: str) -> str:
    """Return the stable identifier shared by manifests and model artifacts."""
    genus = genus_id.strip()
    species = species_name.strip()
    if not genus or not species:
        raise ValueError("Both genus_id and species_name must be non-empty.")
    return f"{genus}::{species}"


@dataclass(frozen=True)
class TaxonRecord:
    genus_id: str
    species_name: str
    species_id: str
    image_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OrchidTaxonomy:
    """Validated taxonomy inventory for the local image hierarchy."""

    records: tuple[TaxonRecord, ...]

    @property
    def genera(self) -> tuple[str, ...]:
        return tuple(sorted({record.genus_id for record in self.records}))

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(record.species_id for record in self.records)

    def species_for_genus(self, genus_id: str) -> tuple[TaxonRecord, ...]:
        return tuple(record for record in self.records if record.genus_id == genus_id)

    def require_genus(self, genus_id: str) -> tuple[TaxonRecord, ...]:
        matches = self.species_for_genus(genus_id)
        if not matches:
            raise ValueError(f"Unknown target genus '{genus_id}'. Available genera: {', '.join(self.genera)}")
        return matches

    def labels_for_task(self, task: str, target_genus: str | None = None) -> tuple[str, ...]:
        """Return deterministic class labels for one supported training task."""
        if task == TASK_GENUS:
            return self.genera
        if task == TASK_FLAT_SPECIES:
            return self.species_ids
        if task in {TASK_GENUS_SPECIES, TASK_TARGET_GENUS}:
            if not target_genus:
                raise ValueError(f"task='{task}' requires target_genus.")
            return tuple(record.species_id for record in self.require_genus(target_genus))
        raise ValueError(f"Unsupported orchid task '{task}'.")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "genera": list(self.genera),
            "records": [record.as_dict() for record in self.records],
        }


def _image_count(species_dir: Path, image_suffixes: Iterable[str]) -> int:
    suffixes = {suffix.lower() for suffix in image_suffixes}
    return sum(1 for path in species_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def scan_orchid_taxonomy(
    root_dir: str | Path,
    image_suffixes: Iterable[str] = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"),
) -> OrchidTaxonomy:
    """Scan a local orchid hierarchy while excluding explicitly unlabeled folders."""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    records: list[TaxonRecord] = []
    for genus_dir in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not genus_dir.is_dir() or genus_dir.name.upper() == "UNLABELED":
            continue
        for species_dir in sorted(genus_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not species_dir.is_dir() or species_dir.name.upper() == "UNLABELED":
                continue
            genus_id = genus_dir.name
            species_name = species_dir.name
            records.append(
                TaxonRecord(
                    genus_id=genus_id,
                    species_name=species_name,
                    species_id=canonical_species_id(genus_id, species_name),
                    image_count=_image_count(species_dir, image_suffixes),
                )
            )
    if not records:
        raise ValueError(f"No labeled Genus/Species folders found under {root}.")
    return OrchidTaxonomy(tuple(records))
