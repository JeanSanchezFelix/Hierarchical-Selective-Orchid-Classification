#!/usr/bin/env python3
"""Run one seed of a paper single-model orchid condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_compression.src.orchid.experiment import calibrate, evaluate, load_paper_config, train
from model_compression.src.orchid.models import TRAINABLE_METHODS


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("train", "calibrate", "evaluate", "all"))
    result.add_argument("--config", default="configs/orchid/paper_experiment_template.yaml")
    result.add_argument("--dataset-root")
    result.add_argument("--split-manifest")
    result.add_argument("--seed", type=int)
    result.add_argument("--method", choices=sorted(TRAINABLE_METHODS))
    result.add_argument("--experiment-id", help="Matrix root; the method and seed are appended automatically.")
    result.add_argument("--artifact-root", default="artifacts/orchid")
    result.add_argument("--checkpoint")
    result.add_argument("--policy")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_paper_config(args.config, args.dataset_root, args.split_manifest, args.seed)
    if args.method:
        config["method"]["id"] = args.method
        if args.method == "flat_ce":
            config["method"].update({"species_loss": "cross_entropy", "genus_weight": 0.0, "consistency_weight": 0.0})
        elif args.method in {"flat_balanced_softmax", "flat_hsc"}:
            config["method"].update({"species_loss": "balanced_softmax", "genus_weight": 0.0, "consistency_weight": 0.0})
        elif args.method == "dual_head":
            config["method"].update({"species_loss": "balanced_softmax", "genus_weight": 1.0, "consistency_weight": 0.0})
        else:
            config["method"].update({"species_loss": "balanced_softmax", "genus_weight": 1.0, "consistency_weight": 0.25})
    if args.experiment_id:
        config["experiment_id"] = f"{args.experiment_id.rstrip('/')}/{config['method']['id']}"
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.command in {"train", "all"}:
        checkpoint = train(config, args.artifact_root)
        print(f"checkpoint={checkpoint}")
    if args.command == "train":
        return
    if checkpoint is None:
        raise ValueError("--checkpoint is required for calibrate and evaluate.")
    policy = Path(args.policy) if args.policy else None
    if args.command in {"calibrate", "all"}:
        policy = calibrate(config, checkpoint)
        print(f"policy={policy}")
    if args.command == "calibrate":
        return
    if policy is None:
        raise ValueError("--policy is required for evaluate.")
    print(json.dumps(evaluate(config, checkpoint, policy), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
