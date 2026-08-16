#!/usr/bin/env python3
"""Validate and register a cascade-evaluation configuration.

Execution is intentionally added in Phase 7 after routing, calibration, and the
metric contract exist. This command never reports model results.
"""
from __future__ import annotations
import argparse
from orchid_training import write_evaluation_request

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/orchid/cascade_evaluation.yaml")
parser.add_argument("--artifact-root", default="artifacts/orchid")
args = parser.parse_args()
print(write_evaluation_request(args.config, args.artifact_root))
