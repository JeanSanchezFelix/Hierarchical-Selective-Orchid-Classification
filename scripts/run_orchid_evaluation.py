#!/usr/bin/env python3
"""Evaluate trained router and genus-expert bundles on the held-out manifest split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from orchid_training import load_orchid_config
from model_compression.src.orchid.calibration import UnknownPolicy
from model_compression.src.orchid.evaluation import evaluate_cascade
from model_compression.src.orchid.phylogeny import build_posterior_distance_summary
from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset


def parse_expert_checkpoints(values: list[str]) -> dict[str, str]:
    experts = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --expert-checkpoint must be Genus=path/to/best_orchid_model.pt")
        genus, path = value.split("=", 1)
        if not genus or not path or genus in experts:
            raise ValueError(f"Invalid or duplicate expert checkpoint: {value}")
        experts[genus] = path
    if not experts:
        raise ValueError("At least one --expert-checkpoint is required.")
    return experts

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/orchid/cascade_evaluation.yaml")
parser.add_argument("--router-checkpoint", required=True)
parser.add_argument("--expert-checkpoint", action="append", default=[], metavar="GENUS=PATH")
parser.add_argument("--unknown-policy", help="Optional JSON file created from validation-set calibration.")
parser.add_argument("--phylogeny-tree-directory", help="Optional directory of reviewed posterior .tre trees.")
parser.add_argument("--phylogeny-mapping", help="Reviewed species_mapping.csv required with --phylogeny-tree-directory.")
parser.add_argument("--phylogeny-minimum-coverage", type=float, default=0.90)
parser.add_argument("--output-dir", default=None)
args = parser.parse_args()

config = load_orchid_config(args.config)
evaluation = config.get("evaluation", {})
policy = None
if args.unknown_policy:
    values = json.loads(Path(args.unknown_policy).read_text(encoding="utf-8"))
    policy = UnknownPolicy(**values)
output_dir = args.output_dir or evaluation.get("output_dir", "artifacts/orchid/cascade_evaluation/top2/reports")
phylogeny_labels = phylogeny_matrix = None
if args.phylogeny_tree_directory or args.phylogeny_mapping:
    if not args.phylogeny_tree_directory or not args.phylogeny_mapping:
        raise ValueError("Both --phylogeny-tree-directory and --phylogeny-mapping are required together.")
    inventory = TaxonomicOrchidDataset(
        root_dir=config["dataset"]["root_dir"], task="flat_species", use_class_balances=False, use_minority_augmentation=False
    )
    phylogeny_labels, phylogeny_matrix, _, _ = build_posterior_distance_summary(
        args.phylogeny_tree_directory, inventory.classes, args.phylogeny_mapping, args.phylogeny_minimum_coverage
    )
summary = evaluate_cascade(
    args.router_checkpoint,
    parse_expert_checkpoints(args.expert_checkpoint),
    config["dataset"]["root_dir"],
    config["dataset"]["split_manifest"],
    output_dir,
    top_k=int(evaluation.get("top_k", 2)),
    unknown_policy=policy,
    batch_size=int(config["training"].get("batch_size", 32)),
    phylogeny_labels=phylogeny_labels,
    phylogeny_matrix=phylogeny_matrix,
)
print(json.dumps(summary, indent=2, sort_keys=True))
