#!/usr/bin/env python3
"""Train the first-stage genus router."""
from __future__ import annotations
import argparse
from orchid_training import load_orchid_config, run_training

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/orchid/genus_router.yaml")
parser.add_argument("--artifact-root", default="artifacts/orchid")
args = parser.parse_args()
print(run_training(load_orchid_config(args.config), args.artifact_root))
