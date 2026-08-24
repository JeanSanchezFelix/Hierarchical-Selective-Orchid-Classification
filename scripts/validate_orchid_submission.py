#!/usr/bin/env python3
"""Fail closed when required orchid paper artifacts are incomplete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--matrix", default="configs/orchid/paper_matrix.yaml")
parser.add_argument("--artifact-root", default="artifacts/orchid")
parser.add_argument("--dataset-manifest", required=True)
args = parser.parse_args()

matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
required = [Path(args.dataset_manifest)]
for method in matrix["methods"]:
    for seed in matrix["seeds"]:
        report = Path(args.artifact_root) / matrix["experiment_id"] / method / f"seed-{seed}" / "reports"
        required.extend([report / "metrics.json", report / "predictions.csv"])
summary = Path(args.artifact_root) / "paper_summary"
required.extend([summary / "seed_metrics.csv", summary / "paired_bootstrap.json", summary / "risk_coverage.csv"])
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("SUBMISSION BLOCKED: missing required artifacts:\n" + "\n".join(missing))

with (summary / "paired_bootstrap.json").open(encoding="utf-8") as stream:
    paired = json.load(stream)
if not paired:
    raise SystemExit("SUBMISSION BLOCKED: paired bootstrap output is empty.")
print("SUBMISSION GATE PASSED")
