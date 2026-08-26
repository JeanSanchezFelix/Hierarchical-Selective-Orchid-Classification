"""Single-model training, calibration, and evaluation for the orchid paper."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset
from model_compression.src.utils.metrics import (
    calculate_metrics,
    export_readable_metrics_report,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_metric_bar,
    plot_radar_chart,
    plot_roc_auc_curve,
)
from model_compression.src.utils.preprocessing import load_data

from .artifacts import OrchidArtifactLayout
from .calibration import HierarchicalSelectivePolicy, fit_hierarchical_selective_policy, forced_species_decisions, hierarchical_decisions
from .checkpoints import load_orchid_checkpoint, save_orchid_checkpoint
from .paper_results import hierarchical_aurc, risk_coverage_rows
from .models import (
    HSC_METHODS,
    BalancedSoftmaxLoss,
    CASCADE_METHODS,
    DUAL_HEAD_METHODS,
    METHOD_FLAT_CE,
    OrchidMultiTaskLoss,
    OrchidTaxonomyIndex,
    build_orchid_model,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_paper_config(path: str | Path, dataset_root: str | None = None, split_manifest: str | None = None, seed: int | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    for key in ("experiment_id", "dataset", "method", "training", "evaluation"):
        if key not in config:
            raise ValueError(f"{config_path} is missing required section {key!r}.")
    if dataset_root:
        config["dataset"]["root_dir"] = dataset_root
    if split_manifest:
        config["dataset"]["split_manifest"] = split_manifest
    if seed is not None:
        config["training"]["seed"] = int(seed)
    method = str(config["method"].get("id", ""))
    if method in CASCADE_METHODS:
        raise ValueError(f"{method} is evaluated through the cascade adapter, not this single-model runner.")
    if not config["dataset"].get("root_dir") or not config["dataset"].get("split_manifest"):
        raise ValueError("dataset.root_dir and dataset.split_manifest are required.")
    return config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _dataset_and_loaders(config: Mapping[str, Any]) -> tuple[TaxonomicOrchidDataset, dict[str, Any], OrchidTaxonomyIndex]:
    dataset_config = config["dataset"]
    inventory = TaxonomicOrchidDataset(
        root_dir=dataset_config["root_dir"], task="flat_species", use_class_balances=False, use_minority_augmentation=False
    )
    taxonomy = OrchidTaxonomyIndex.from_species_ids(inventory.classes)
    training = config["training"]
    loaders = load_data(
        "TaxonomicOrchid",
        batch_size=int(training["batch_size"]),
        img_size=int(training["image_size"]),
        use_augmentation=True,
        use_sampler=False,
        split_manifest=dataset_config["split_manifest"],
        dataset_kwargs={"root_dir": dataset_config["root_dir"], "task": "flat_species", "use_class_balances": False, "use_minority_augmentation": False},
    )
    if "calibration" not in loaders:
        raise ValueError("Paper runner requires a calibration split in the manifest.")
    return inventory, loaders, taxonomy


def _criterion(config: Mapping[str, Any], taxonomy: OrchidTaxonomyIndex, train_targets: list[int]) -> OrchidMultiTaskLoss:
    method = config["method"]
    counts = np.bincount(train_targets, minlength=len(taxonomy.species_ids)).tolist()
    species_loss = BalancedSoftmaxLoss(counts) if method.get("species_loss") == "balanced_softmax" else torch.nn.CrossEntropyLoss()
    return OrchidMultiTaskLoss(
        taxonomy,
        species_loss,
        genus_weight=float(method.get("genus_weight", 0.0)),
        consistency_weight=float(method.get("consistency_weight", 0.0)),
    )


def _loss_for_batch(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, criterion: OrchidMultiTaskLoss) -> torch.Tensor:
    outputs = model(images)
    if isinstance(outputs, tuple):
        return criterion(outputs[0], targets, outputs[1])[0]
    return criterion(outputs, targets)[0]


def _mean_loss(model: torch.nn.Module, loader: Any, criterion: OrchidMultiTaskLoss, device: torch.device) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images, targets = batch[:2]
            values.append(float(_loss_for_batch(model, images.to(device), targets.to(device), criterion).item()))
    return float(np.mean(values)) if values else float("inf")


def _metadata(config: Mapping[str, Any], taxonomy: OrchidTaxonomyIndex) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "task": "paper_single_model",
        "method": config["method"]["id"],
        "class_labels": list(taxonomy.species_ids),
        "genus_labels": list(taxonomy.genus_ids),
        "model_name": config["method"].get("backbone", "mobilenet_v2"),
        "img_size": int(config["training"]["image_size"]),
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "dataset_id": dataset.get("dataset_id"),
        "dataset_manifest_sha256": _sha256(dataset["split_manifest"]),
        "seed": int(config["training"]["seed"]),
        "config": config,
        "code_revision": _git_revision(),
        "torch_version": torch.__version__,
    }


def train(config: Mapping[str, Any], artifact_root: str | Path = "artifacts/orchid") -> Path:
    _seed_everything(int(config["training"]["seed"]))
    inventory, loaders, taxonomy = _dataset_and_loaders(config)
    method = config["method"]
    model = build_orchid_model(method["id"], taxonomy, bool(method.get("use_imagenet_weights", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = _criterion(config, taxonomy, list(loaders["train"].dataset.targets)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    layout = OrchidArtifactLayout.create(artifact_root, f"{config['experiment_id']}/seed-{config['training']['seed']}")
    layout.write_json(layout.taxonomy_path, {"schema_version": 1, "records": [record.as_dict() for record in inventory.dataset.taxonomy.records]})
    metadata = _metadata(config, taxonomy)
    layout.write_json(layout.run_metadata_path, metadata)
    history = {"train": [], "val": []}
    best_loss = float("inf")
    epochs_without_improvement = 0
    training = config["training"]
    max_epochs = int(training["epochs"])
    early_stopping_patience = int(training.get("early_stopping_patience", 5))
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 1e-4))
    if early_stopping_patience < 1:
        raise ValueError("training.early_stopping_patience must be at least 1.")
    if early_stopping_min_delta < 0:
        raise ValueError("training.early_stopping_min_delta must be non-negative.")
    checkpoint = layout.checkpoints_dir / "best_orchid_model.pt"
    for epoch in range(max_epochs):
        model.train()
        losses = []
        with tqdm(
            loaders["train"],
            desc=f"Train Epoch {epoch + 1}/{max_epochs}",
            unit=" batch",
            dynamic_ncols=True,
        ) as progress:
            for batch in progress:
                images, targets = batch[:2]
                optimizer.zero_grad(set_to_none=True)
                loss = _loss_for_batch(model, images.to(device), targets.to(device), criterion)
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.item())
                losses.append(batch_loss)
                progress.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{np.mean(losses):.4f}")
        train_loss = float(np.mean(losses))
        val_loss = _mean_loss(model, loaders["val"], criterion, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_loss - early_stopping_min_delta:
            best_loss = val_loss
            epochs_without_improvement = 0
            save_orchid_checkpoint(checkpoint, model, metadata, epoch=epoch, monitored_metric=best_loss, optimizer=optimizer, history=history)
        else:
            epochs_without_improvement += 1
        logging.info(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_loss,
                    "early_stopping_wait": epochs_without_improvement,
                    "early_stopping_patience": early_stopping_patience,
                }
            )
        )
        if epochs_without_improvement >= early_stopping_patience:
            logging.info(
                "Early stopping at epoch %s: validation loss did not improve by at least %s for %s consecutive epochs.",
                epoch + 1,
                early_stopping_min_delta,
                early_stopping_patience,
            )
            break
    return checkpoint


def _load_model(checkpoint: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], OrchidTaxonomyIndex]:
    bundle = load_orchid_checkpoint(checkpoint, map_location=device)
    metadata = bundle["metadata"]
    taxonomy = OrchidTaxonomyIndex.from_species_ids(metadata["class_labels"])
    model = build_orchid_model(metadata["method"], taxonomy, use_imagenet_weights=False).to(device).eval()
    model.load_state_dict(bundle["model_state_dict"])
    return model, metadata, taxonomy


def _collect_logits(model: torch.nn.Module, loader: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    species_logits: list[np.ndarray] = []
    genus_logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    image_paths: list[str] = []
    source_indices = list(loader.dataset.subset.indices)
    root_dataset = loader.dataset.subset.dataset
    image_folder = getattr(root_dataset, "dataset", root_dataset)
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit=" batch", dynamic_ncols=True):
            images, labels = batch[:2]
            output = model(images.to(device))
            if isinstance(output, tuple):
                species, genus = output
                genus_logits.append(genus.cpu().numpy())
            else:
                species = output
            species_logits.append(species.cpu().numpy())
            targets.append(labels.numpy())
            for index in source_indices[cursor : cursor + len(labels)]:
                image_path = Path(image_folder.samples[index][0])
                try:
                    image_paths.append(image_path.relative_to(root_dataset.rootDir).as_posix())
                except ValueError:
                    # Keep an auditable identifier even for custom dataset adapters.
                    image_paths.append(str(image_path))
            cursor += len(labels)
    return np.concatenate(species_logits), np.concatenate(targets), (np.concatenate(genus_logits) if genus_logits else None), image_paths


def calibrate(config: Mapping[str, Any], checkpoint: str | Path, output: str | Path | None = None) -> Path:
    _, loaders, _ = _dataset_and_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata, taxonomy = _load_model(checkpoint, device)
    species_logits, targets, genus_logits, _ = _collect_logits(model, loaders["calibration"], device)
    policy = fit_hierarchical_selective_policy(species_logits, targets, taxonomy, genus_logits)
    destination = Path(output) if output else Path(checkpoint).parents[1] / "reports" / "hierarchical_policy.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(policy), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(1 / bins, 1, bins)):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            result += abs(confidence[mask].mean() - correct[mask].mean()) * mask.mean()
    return float(result)


def _shot_stratified_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    train_class_counts: np.ndarray,
    evaluation_config: Mapping[str, Any],
) -> dict[str, float | int | None]:
    """Report species metrics by frozen training-set frequency strata."""
    strata_config = evaluation_config.get("shot_strata", {})
    medium_minimum = int(strata_config.get("medium_min_train_images", 50))
    many_minimum = int(strata_config.get("many_min_train_images", 100))
    if medium_minimum < 1 or many_minimum <= medium_minimum:
        raise ValueError("shot strata require 1 <= medium_min_train_images < many_min_train_images.")
    result: dict[str, float | int | None] = {}
    counts_per_example = train_class_counts[targets]
    masks = {
        "few": counts_per_example < medium_minimum,
        "medium": (counts_per_example >= medium_minimum) & (counts_per_example < many_minimum),
        "many": counts_per_example >= many_minimum,
    }
    for name, mask in masks.items():
        count = int(mask.sum())
        result[f"species_{name}_shot_n_test_images"] = count
        if not count:
            result[f"species_{name}_shot_top1_accuracy"] = None
            result[f"species_{name}_shot_macro_f1"] = None
            continue
        stratum_targets = targets[mask]
        stratum_predictions = predictions[mask]
        labels = np.unique(stratum_targets)
        result[f"species_{name}_shot_top1_accuracy"] = float(np.mean(stratum_predictions == stratum_targets))
        result[f"species_{name}_shot_macro_f1"] = float(
            f1_score(stratum_targets, stratum_predictions, labels=labels, average="macro", zero_division=0)
        )
    return result


def evaluate(config: Mapping[str, Any], checkpoint: str | Path, policy_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    _, loaders, _ = _dataset_and_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata, taxonomy = _load_model(checkpoint, device)
    species_logits, targets, genus_logits, image_paths = _collect_logits(model, loaders["test"], device)
    policy = HierarchicalSelectivePolicy(**json.loads(Path(policy_path).read_text(encoding="utf-8")))
    hsc_enabled = config["method"]["id"] in HSC_METHODS
    decisions = (
        hierarchical_decisions(species_logits, taxonomy, policy, genus_logits)
        if hsc_enabled
        else forced_species_decisions(species_logits, taxonomy, policy)
    )
    probabilities = np.asarray([torch.softmax(torch.as_tensor(row), dim=0).numpy() for row in species_logits])
    calibrated_probabilities = np.asarray([
        torch.softmax(torch.as_tensor(row / policy.species_temperature), dim=0).numpy() for row in species_logits
    ])
    forced = probabilities.argmax(axis=1)
    species_to_genus = np.asarray(taxonomy.species_to_genus, dtype=int)
    genus_targets = species_to_genus[targets]
    forced_genera = species_to_genus[forced]
    implied_genus_probabilities = np.zeros((len(probabilities), len(taxonomy.genus_ids)), dtype=float)
    for species_index, genus_index in enumerate(species_to_genus):
        implied_genus_probabilities[:, genus_index] += probabilities[:, species_index]
    if genus_logits is None:
        genus_probabilities = implied_genus_probabilities
        taxonomy_consistency_violation_rate = 0.0
    else:
        genus_probabilities = torch.softmax(torch.as_tensor(genus_logits), dim=1).numpy()
        taxonomy_consistency_violation_rate = float(np.mean(genus_probabilities.argmax(axis=1) != forced_genera))
    ranked_genera = np.argsort(-genus_probabilities, axis=1, kind="mergesort")
    genus_top1 = ranked_genera[:, 0]
    genus_top2 = ranked_genera[:, : min(2, len(taxonomy.genus_ids))]
    train_class_counts = np.bincount(
        np.asarray(loaders["train"].dataset.targets, dtype=int), minlength=len(taxonomy.species_ids)
    )
    shot_metrics = _shot_stratified_metrics(targets, forced, train_class_counts, config["evaluation"])
    accepted = [index for index, decision in enumerate(decisions) if decision["decision_level"] != "unknown"]
    species_coverage = float(np.mean([decision["decision_level"] == "species" for decision in decisions]))
    genus_coverage = float(np.mean([decision["decision_level"] == "genus" for decision in decisions]))
    unknown_coverage = float(np.mean([decision["decision_level"] == "unknown" for decision in decisions]))
    correct = []
    rows = []
    for index, decision in enumerate(decisions):
        true_species = taxonomy.species_ids[int(targets[index])]
        true_genus = taxonomy.genus_ids[taxonomy.species_to_genus[int(targets[index])]]
        predicted_species = taxonomy.species_ids[int(decision["species_index"])] if decision["species_index"] is not None else ""
        predicted_genus = taxonomy.genus_ids[int(decision["genus_index"])] if decision["genus_index"] is not None else ""
        is_correct = predicted_species == true_species or (decision["decision_level"] == "genus" and predicted_genus == true_genus)
        correct.append(is_correct)
        rows.append({"image_file": image_paths[index], "true_species_id": true_species, "true_genus_id": true_genus, "forced_species_id": taxonomy.species_ids[int(forced[index])], "forced_genus_id": taxonomy.genus_ids[int(forced_genera[index])], "genus_top1_id": taxonomy.genus_ids[int(genus_top1[index])], "decision_level": decision["decision_level"], "predicted_species_id": predicted_species, "predicted_genus_id": predicted_genus, "confidence": decision["confidence"], "margin": decision["margin"], "hierarchically_correct": is_correct})
    metrics: dict[str, Any] = {
        "n_test_images": int(len(targets)),
        "hsc_enabled": hsc_enabled,
        "species_top1_accuracy": float(np.mean(forced == targets)),
        "species_macro_f1": float(f1_score(targets, forced, average="macro", zero_division=0)),
        "species_balanced_recall": float(recall_score(targets, forced, average="macro", zero_division=0)),
        "species_top5_accuracy": float(np.mean([target in np.argsort(-row)[: min(5, len(row))] for row, target in zip(probabilities, targets)])),
        "ece_uncalibrated": _ece(probabilities, targets),
        "ece_calibrated": _ece(calibrated_probabilities, targets),
        "nll_uncalibrated": float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()),
        "nll_calibrated": float(-np.log(np.clip(calibrated_probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()),
        "coverage": len(accepted) / len(targets),
        "species_coverage": species_coverage,
        "genus_coverage": genus_coverage,
        "unknown_coverage": unknown_coverage,
        "unknown_rate": unknown_coverage,
        "hierarchical_selective_accuracy": float(np.mean(correct)),
        "selective_hierarchical_accuracy": float(np.mean([correct[index] for index in accepted])) if accepted else 0.0,
        "genus_top1_accuracy": float(np.mean(genus_top1 == genus_targets)),
        "genus_top2_accuracy": float(np.mean([target in ranked for target, ranked in zip(genus_targets, genus_top2)])),
        "taxonomy_consistency_violation_rate": taxonomy_consistency_violation_rate,
        **shot_metrics,
    }
    destination = Path(output_dir) if output_dir else Path(checkpoint).parents[1] / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    metrics["hAURC"] = hierarchical_aurc(rows)
    with (destination / "risk_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["coverage", "hierarchical_risk"])
        writer.writeheader()
        writer.writerows(risk_coverage_rows(rows))
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Reuse the standard model_compression metric exports for directly comparable
    # figures and readable CSV/Markdown reports.
    metric_dir = destination / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pipeline_metrics = calculate_metrics(targets, forced, probabilities)
    report_metrics = {**pipeline_metrics, **metrics}
    observed_class_ids = np.unique(targets)
    observed_class_names = [taxonomy.species_ids[int(index)] for index in observed_class_ids]
    confusion_class_ids = np.unique(np.concatenate((targets, forced)))
    confusion_class_names = [taxonomy.species_ids[int(index)] for index in confusion_class_ids]
    plot_metric_bar(pipeline_metrics, title="Test Metrics", save_path=str(metric_dir / "metrics.png"))
    plot_confusion_matrix(
        targets,
        forced,
        labels=confusion_class_names,
        title="Confusion Matrix",
        save_path=str(metric_dir / "confusion_matrix.png"),
    )
    plot_radar_chart(pipeline_metrics, save_path=str(metric_dir / "radar_chart.png"))
    if len(observed_class_ids) >= 2:
        observed_probabilities = probabilities[:, observed_class_ids]
        plot_roc_auc_curve(
            targets,
            observed_probabilities,
            observed_class_names,
            title="ROC-AUC Curve",
            save_path=str(metric_dir / "roc_auc.png"),
        )
        plot_calibration_curve(
            targets,
            observed_probabilities,
            observed_class_names,
            title="Calibration Curve",
            save_path=str(metric_dir / "calibration.png"),
        )
    export_readable_metrics_report(
        report_metrics,
        targets,
        forced,
        probabilities,
        str(metric_dir),
        list(taxonomy.species_ids),
        image_paths,
    )
    return metrics
