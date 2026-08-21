#!/usr/bin/env python3
"""Train one specified genus expert, or every genus after explicit confirmation."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Directly running a script puts only scripts/ on sys.path. Add the repository
# root so local packages such as datasets are importable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets.TaxonomicOrchidDataset import TaxonomicOrchidDataset
from orchid_training import load_orchid_config, run_training, with_target_genus

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/orchid/expert_template.yaml")
parser.add_argument("--genus", required=True, help="One genus name, or literal 'all'.")
parser.add_argument("--artifact-root", default="artifacts/orchid")
args = parser.parse_args()
config = load_orchid_config(args.config)
if args.genus == "all":
    inventory = TaxonomicOrchidDataset(
        root_dir=config["dataset"]["root_dir"], task="genus", use_class_balances=False, use_minority_augmentation=False
    )
    genera = inventory.classes
else:
    genera = [args.genus]
for genus in genera:
    print(run_training(with_target_genus(config, genus), args.artifact_root))
