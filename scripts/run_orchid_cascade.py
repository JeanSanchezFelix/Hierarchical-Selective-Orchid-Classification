#!/usr/bin/env python3
"""Train the reproducible router-plus-expert controls for one paper seed."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset
from model_compression.src.orchid.evaluation import evaluate_cascade
from orchid_training import run_training, with_target_genus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orchid/paper_cascade_template.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--artifact-root", default="artifacts/orchid")
    parser.add_argument("--skip-train", action="store_true", help="Evaluate existing seed checkpoints only.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    seed = config["training"]["seed"]
    base_id = f"{config['experiment_id'].rstrip('/')}/cascade_models/seed-{seed}"
    router = copy.deepcopy(config)
    router["experiment_id"] = f"{base_id}/router"
    router["dataset"]["task"] = "genus"
    if not args.skip_train:
        run_training(router, args.artifact_root)
    inventory = TaxonomicOrchidDataset(root_dir=config["dataset"]["root_dir"], task="genus", use_class_balances=False, use_minority_augmentation=False)
    for genus in inventory.classes:
        expert = copy.deepcopy(config)
        expert["experiment_id"] = f"{base_id}/experts"
        expert["dataset"]["task"] = "genus_species"
        if not args.skip_train:
            run_training(with_target_genus(expert, genus), args.artifact_root)
    artifact_root = Path(args.artifact_root)
    router_checkpoint = artifact_root / base_id / "router" / "checkpoints" / "best_orchid_model.pt"
    expert_checkpoints = {
        genus: artifact_root / base_id / "experts" / genus / "checkpoints" / "best_orchid_model.pt"
        for genus in inventory.classes
    }
    for method, top_k in (("cascade_top1", 1), ("cascade_top2", 2)):
        output = artifact_root / config["experiment_id"] / method / f"seed-{seed}" / "reports"
        evaluate_cascade(router_checkpoint, expert_checkpoints, config["dataset"]["root_dir"], config["dataset"]["split_manifest"], output, top_k=top_k, batch_size=config["training"]["batch_size"])


if __name__ == "__main__":
    main()
