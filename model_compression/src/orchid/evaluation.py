"""Held-out cascade evaluation and paper-ready report generation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from model_compression.src.utils.model_setup import setup_model
from model_compression.src.utils.preprocessing import load_data

from .calibration import OpenSetDecision, UnknownPolicy, apply_unknown_policy
from .checkpoints import load_orchid_checkpoint
from .phylogeny import mean_phylogenetic_error
from .routing import HierarchicalCascadeRouter


@dataclass(frozen=True)
class EvaluationRecord:
    true_species_id: str
    true_genus_id: str
    predicted_species_id: str | None
    predicted_genus_id: str | None
    top1_genus_id: str
    routed_genus_ids: tuple[str, ...]
    joint_probability: float | None
    is_unknown: bool
    unknown_reason: str | None
    image_file: str = ""
    decision_level: str = "species"
    confidence: float | None = None


def load_bundle_model(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Reconstruct one saved model without downloading ImageNet weights."""
    bundle = load_orchid_checkpoint(path, map_location=device)
    metadata = bundle["metadata"]
    model = setup_model(
        metadata["model_name"],
        pretrained_weights_path=None,
        num_classes=len(metadata["class_labels"]),
        use_imagenet_weights=False,
    )
    model.load_state_dict(bundle["model_state_dict"])
    return model.to(device).eval(), metadata


def _taxonomy_cost(true_species: str, predicted_species: str | None) -> float:
    if predicted_species is None:
        return 2.0
    if true_species == predicted_species:
        return 0.0
    return 1.0 if true_species.split("::", 1)[0] == predicted_species.split("::", 1)[0] else 2.0


def summarize_records(
    records: Sequence[EvaluationRecord],
    phylogeny_labels: list[str] | None = None,
    phylogeny_matrix: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute hierarchical, selective-prediction, and optional phylogenetic metrics."""
    if not records:
        raise ValueError("Cannot summarize an empty evaluation.")
    total = len(records)
    accepted = [record for record in records if not record.is_unknown and record.predicted_species_id is not None]
    summary: dict[str, float | int] = {
        "n_test_images": total,
        "router_top1_genus_accuracy": float(np.mean([record.true_genus_id == record.top1_genus_id for record in records])),
        "router_top2_genus_accuracy": float(np.mean([record.true_genus_id in record.routed_genus_ids for record in records])),
        "cascade_species_accuracy": float(np.mean([record.true_species_id == record.predicted_species_id for record in records])),
        "unknown_rate": float(np.mean([record.is_unknown for record in records])),
        "known_coverage": len(accepted) / total,
        "selective_species_accuracy": float(np.mean([record.true_species_id == record.predicted_species_id for record in accepted])) if accepted else 0.0,
        "mean_taxonomic_error_cost": float(np.mean([_taxonomy_cost(record.true_species_id, record.predicted_species_id) for record in records])),
    }
    if phylogeny_labels is not None and phylogeny_matrix is not None:
        mapped = [record for record in accepted if record.true_species_id in phylogeny_labels and record.predicted_species_id in phylogeny_labels]
        summary["phylogenetic_metric_coverage"] = len(mapped) / total
        if mapped:
            summary["mean_phylogenetic_error"] = mean_phylogenetic_error(
                [record.true_species_id for record in mapped], [record.predicted_species_id for record in mapped], phylogeny_labels, phylogeny_matrix
            )
    return summary


def write_evaluation_report(records: Sequence[EvaluationRecord], summary: Mapping[str, object], output_dir: str | Path) -> Path:
    """Write machine-readable metrics plus an auditable image-level result table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return output


def evaluate_cascade(
    router_checkpoint: str | Path,
    expert_checkpoints: Mapping[str, str | Path],
    dataset_root: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    *,
    top_k: int = 2,
    unknown_policy: UnknownPolicy | None = None,
    batch_size: int = 32,
    device: torch.device | None = None,
    phylogeny_labels: list[str] | None = None,
    phylogeny_matrix: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Evaluate a saved cascade on the manifest's held-out test split."""
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router_model, router_metadata = load_bundle_model(router_checkpoint, runtime_device)
    experts = {}
    expert_metadata = {}
    for genus, checkpoint in expert_checkpoints.items():
        experts[genus], expert_metadata[genus] = load_bundle_model(checkpoint, runtime_device)
    cascade = HierarchicalCascadeRouter.from_metadata(router_metadata, expert_metadata)
    loaders = load_data(
        "TaxonomicOrchid",
        batch_size=batch_size,
        img_size=int(router_metadata["img_size"]),
        split_manifest=str(split_manifest),
        dataset_kwargs={"root_dir": str(dataset_root), "task": "flat_species", "use_class_balances": False, "use_minority_augmentation": False},
    )
    test_loader = loaders["test"]
    records: list[EvaluationRecord] = []
    source_indices = list(test_loader.dataset.subset.indices)
    root_dataset = test_loader.dataset.subset.dataset
    image_folder = getattr(root_dataset, "dataset", root_dataset)
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, target_indices = batch[:2]
            inputs = inputs.to(runtime_device)
            router_batch_logits = router_model(inputs).detach().cpu().numpy()
            for image_index, router_logits in enumerate(router_batch_logits):
                selected = cascade.select_genera(router_logits, top_k)
                expert_logits = {}
                image = inputs[image_index : image_index + 1]
                for candidate in selected:
                    model = experts.get(candidate.genus_id)
                    if model is not None:
                        expert_logits[candidate.genus_id] = model(image).detach().cpu().numpy().reshape(-1)
                routed = cascade.route(router_logits, expert_logits, top_k)
                decision: OpenSetDecision
                if unknown_policy:
                    decision = apply_unknown_policy(routed, unknown_policy)
                else:
                    candidate = routed.selected_species
                    decision = OpenSetDecision(False, None, candidate.species_id if candidate else None, candidate.joint_probability if candidate else None)
                true_species = test_loader.dataset.classes[int(target_indices[image_index])]
                predicted = decision.best_candidate_species_id
                records.append(EvaluationRecord(
                    true_species_id=true_species,
                    true_genus_id=true_species.split("::", 1)[0],
                    predicted_species_id=predicted,
                    predicted_genus_id=predicted.split("::", 1)[0] if predicted else None,
                    top1_genus_id=routed.routed_genera[0].genus_id,
                    routed_genus_ids=tuple(candidate.genus_id for candidate in routed.routed_genera),
                    joint_probability=decision.best_candidate_score,
                    is_unknown=decision.is_unknown,
                    unknown_reason=decision.reason,
                    image_file=Path(image_folder.samples[source_indices[cursor + image_index]][0]).relative_to(root_dataset.rootDir).as_posix(),
                    decision_level="unknown" if decision.is_unknown else "species",
                    confidence=decision.best_candidate_score,
                ))
            cursor += len(target_indices)
    summary = summarize_records(records, phylogeny_labels, phylogeny_matrix)
    write_evaluation_report(records, summary, output_dir)
    return summary
