"""Shared, configuration-driven launcher for orchid model experiments.

This module deliberately has no hard-coded dataset path. Configuration captures
the private-root location and reviewed split manifest while artifacts retain only
taxonomy and run provenance.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset
from model_compression.src.orchid.artifacts import OrchidArtifactLayout
from model_compression.src.orchid.constants import TASK_FLAT_SPECIES, TASK_GENUS, TASK_GENUS_SPECIES, TASK_TARGET_GENUS
from model_compression.src.train import transfer_learning
from model_compression.src.utils.preprocessing import load_data


REQUIRED_TOP_LEVEL = {"experiment_id", "dataset", "training"}
SUPPORTED_TASKS = {TASK_GENUS, TASK_FLAT_SPECIES, TASK_GENUS_SPECIES, TASK_TARGET_GENUS}


def load_orchid_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    missing = REQUIRED_TOP_LEVEL - set(config)
    if missing:
        raise ValueError(f"{path} is missing required section(s): {sorted(missing)}")
    if not isinstance(config["dataset"], dict) or not isinstance(config["training"], dict):
        raise ValueError("dataset and training must be YAML mappings.")
    dataset = config["dataset"]
    task = dataset.get("task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"dataset.task must be one of {sorted(SUPPORTED_TASKS)}.")
    if task in {TASK_GENUS_SPECIES, TASK_TARGET_GENUS} and not dataset.get("target_genus"):
        raise ValueError(f"dataset.target_genus is required for task '{task}'.")
    for key in ("root_dir", "split_manifest"):
        if not dataset.get(key):
            raise ValueError(f"dataset.{key} is required.")
    return config


def with_target_genus(config: dict[str, Any], genus: str) -> dict[str, Any]:
    """Return an expert configuration for a single requested genus."""
    configured = copy.deepcopy(config)
    configured["dataset"]["task"] = TASK_GENUS_SPECIES
    configured["dataset"]["target_genus"] = genus
    configured["experiment_id"] = f"{configured['experiment_id'].rstrip('/')}/{genus}"
    return configured


def run_training(config: dict[str, Any], artifact_root: str | Path = "artifacts/orchid") -> Path:
    """Train one configured task and save its provenance beside its checkpoints."""
    dataset_config = config["dataset"]
    training = config["training"]
    layout = OrchidArtifactLayout.create(artifact_root, config["experiment_id"])
    root_dir = Path(dataset_config["root_dir"]).expanduser().resolve()
    manifest = Path(dataset_config["split_manifest"]).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Reviewed split manifest not found: {manifest}")

    dataset_kwargs = {
        "root_dir": str(root_dir),
        "task": dataset_config["task"],
        "target_genus": dataset_config.get("target_genus"),
        "use_class_balances": bool(training.get("class_weights", False)),
        "use_minority_augmentation": False,
    }
    inventory = TaxonomicOrchidDataset(**dataset_kwargs)
    layout.write_json(layout.taxonomy_path, inventory.dataset.taxonomy.as_dict())
    layout.write_json(
        layout.run_metadata_path,
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "dataset_root": str(root_dir),
            "split_manifest": str(manifest),
            "task": dataset_config["task"],
            "target_genus": dataset_config.get("target_genus"),
            "class_labels": inventory.classes,
            "training": training,
        },
    )
    loaders = load_data(
        "TaxonomicOrchid",
        batch_size=int(training.get("batch_size", 32)),
        img_size=int(training.get("img_size", 224)),
        use_augmentation=bool(training.get("data_augmentation", True)),
        use_sampler=bool(training.get("sampler", False)),
        split_manifest=str(manifest),
        dataset_kwargs=dataset_kwargs,
    )
    transfer_learning(
        model_name=training.get("model_name", "mobilenet_v2"),
        data_loaders=loaders,
        save_dir=str(layout.checkpoints_dir),
        learning_rate=float(training.get("learning_rate", 0.001)),
        num_epochs=int(training.get("epochs", 10)),
        criterion_name=training.get("criterion", "cross_entropy"),
        optimizer_name=training.get("optimizer", "adam"),
        callbacks=None,
        pretrained_weights_path=training.get("pretrained_weights"),
        use_class_weights=bool(training.get("class_weights", False)),
    )
    return layout.experiment_dir


def write_evaluation_request(config_path: str | Path, output_root: str | Path = "artifacts/orchid") -> Path:
    """Record a validated cascade-evaluation request until Phase 7 adds execution."""
    config = load_orchid_config(config_path)
    layout = OrchidArtifactLayout.create(output_root, config["experiment_id"])
    output = layout.reports_dir / "evaluation_request.json"
    payload = {
        "schema_version": 1,
        "status": "registered_pending_phase_7_evaluator",
        "config": config,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
