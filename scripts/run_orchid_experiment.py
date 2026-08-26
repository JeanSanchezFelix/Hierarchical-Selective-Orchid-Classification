#!/usr/bin/env python3
"""Run one seed of a paper single-model orchid condition."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_compression.src.orchid.experiment import calibrate, evaluate, load_paper_config, train
from model_compression.src.orchid.models import METHOD_FLAT_BALANCED_SOFTMAX, METHOD_FLAT_HSC, TRAINABLE_METHODS
from model_compression.src.utils.logging_setup import configure_logging


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
    log_dir = Path(args.artifact_root) / config["experiment_id"] / f"seed-{config['training']['seed']}" / "reports"
    configure_logging(enable_console=True, log_dir=str(log_dir))
    logging.info("Starting paper experiment command=%s, method=%s, seed=%s", args.command, config["method"]["id"], config["training"]["seed"])
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    run_dir = Path(args.artifact_root) / config["experiment_id"] / f"seed-{config['training']['seed']}"
    if config["method"]["id"] == METHOD_FLAT_HSC and checkpoint is None:
        matrix_root = Path(config["experiment_id"]).parent
        checkpoint = (
            Path(args.artifact_root)
            / matrix_root
            / METHOD_FLAT_BALANCED_SOFTMAX
            / f"seed-{config['training']['seed']}"
            / "checkpoints"
            / "best_orchid_model.pt"
        )
    try:
        if args.command in {"train", "all"}:
            if config["method"]["id"] == METHOD_FLAT_HSC:
                if args.command == "train":
                    raise ValueError("flat_hsc is post-hoc; use 'all' to reuse a flat_balanced_softmax checkpoint.")
                logging.info("Reusing Balanced Softmax checkpoint for post-hoc HSC: %s", checkpoint)
            elif args.checkpoint:
                logging.info("Reusing supplied checkpoint without training: %s", checkpoint)
            else:
                checkpoint = train(config, args.artifact_root)
                logging.info("checkpoint=%s", checkpoint)
        if args.command == "train":
            return
        if checkpoint is None:
            raise ValueError("--checkpoint is required for calibrate and evaluate.")
        policy = Path(args.policy) if args.policy else None
        if args.command in {"calibrate", "all"}:
            policy = calibrate(config, checkpoint, output=run_dir / "reports" / "hierarchical_policy.json")
            logging.info("policy=%s", policy)
        if args.command == "calibrate":
            return
        if policy is None:
            raise ValueError("--policy is required for evaluate.")
        metrics = evaluate(config, checkpoint, policy, output_dir=run_dir / "reports")
        logging.info("evaluation_metrics=%s", json.dumps(metrics, sort_keys=True))
    except Exception:
        logging.exception("Paper experiment failed")
        raise


if __name__ == "__main__":
    main()
