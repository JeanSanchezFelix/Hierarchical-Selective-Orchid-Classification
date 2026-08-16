"""Stable names shared by orchid training, evaluation, and deployment code."""

from __future__ import annotations

TASK_GENUS = "genus"
TASK_FLAT_SPECIES = "flat_species"
TASK_GENUS_SPECIES = "genus_species"
TASK_TARGET_GENUS = "target_genus"

ORCHID_TASKS = frozenset(
    {TASK_GENUS, TASK_FLAT_SPECIES, TASK_GENUS_SPECIES, TASK_TARGET_GENUS}
)

MANIFEST_REQUIRED_COLUMNS = frozenset({"image_path", "genus_id", "species_name", "species_id", "split"})

