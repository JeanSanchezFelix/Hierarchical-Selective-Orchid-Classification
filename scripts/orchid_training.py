"""Shared, configuration-driven launcher for orchid model experiments.

This module deliberately has no hard-coded dataset path. Configuration captures
the private-root location and reviewed split manifest while artifacts retain only
taxonomy and run provenance.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset
from model_compression.src.orchid.artifacts import OrchidArtifactLayout
from model_compression.src.orchid.constants import TASK_FLAT_SPECIES, TASK_GENUS, TASK_GENUS_SPECIES, TASK_TARGET_GENUS
from model_compression.src.train import transfer_learning
from model_compression.src.utils.callbacks import process_callbacks
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
    seed = training.get("seed")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    layout = OrchidArtifactLayout.create(artifact_root, config["experiment_id"])
    root_dir = Path(dataset_config["root_dir"]).expanduser().resolve()
    manifest = Path(dataset_config["split_manifest"]).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Reviewed split manifest not found: {manifest}")

    dataset_kwargs = {
        "root_dir": str(root_dir),
        "task": dataset_config["task"],
        "target_genus": dataset_config.get("target_genus"),
        # Loss-level class weights are calculated by ``transfer_learning``.
        # Returning per-sample weights here would apply imbalance correction twice.
        "use_class_balances": False,
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
            "seed": int(seed) if seed is not None else None,
            "model_name": training.get("model_name", "mobilenet_v2"),
            "img_size": int(training.get("img_size", 224)),
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
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
    callback_config = dict(training)
    callback_config.update(
        {
            "save_dir": str(layout.checkpoints_dir),
            "save_name": "callback",
        }
    )
    callbacks = list(process_callbacks(callback_config).values())
    transfer_learning(
        model_name=training.get("model_name", "mobilenet_v2"),
        data_loaders=loaders,
        save_dir=str(layout.checkpoints_dir),
        learning_rate=float(training.get("learning_rate", 0.001)),
        num_epochs=int(training.get("epochs", 10)),
        criterion_name=training.get("criterion", "cross_entropy"),
        optimizer_name=training.get("optimizer", "adam"),
        callbacks=callbacks,
        pretrained_weights_path=training.get("pretrained_weights"),
        use_class_weights=bool(training.get("class_weights", False)),
        orchid_checkpoint_path=str(layout.checkpoints_dir / "best_orchid_model.pt"),
        orchid_checkpoint_metadata={
            "task": dataset_config["task"],
            "target_genus": dataset_config.get("target_genus"),
            "class_labels": inventory.classes,
            "model_name": training.get("model_name", "mobilenet_v2"),
            "img_size": int(training.get("img_size", 224)),
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "taxonomy_path": str(layout.taxonomy_path),
            "split_manifest": str(manifest),
        },
    )
    return layout.experiment_dir
