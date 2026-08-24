#!/usr/bin/env python3
"""Summarize all completed orchid paper seeds without retraining them."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_compression.src.orchid.paper_results import summarize_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--matrix", default="configs/orchid/paper_matrix.yaml")
parser.add_argument("--artifact-root", default="artifacts/orchid")
parser.add_argument("--output-dir", default="artifacts/orchid/paper_summary")
args = parser.parse_args()
matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
runs = {
    method: {seed: Path(args.artifact_root) / matrix["experiment_id"] / method / f"seed-{seed}" / "reports" / "predictions.csv"
             for seed in matrix["seeds"]}
    for method in matrix["methods"]
}
missing = [str(path) for per_seed in runs.values() for path in per_seed.values() if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing completed prediction files:\n" + "\n".join(missing))
print(summarize_matrix(runs, matrix["reference_method"], args.output_dir))
