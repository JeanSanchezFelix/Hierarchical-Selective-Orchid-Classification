#!/usr/bin/env python3
"""Write a host-only edge-readiness report for one frozen paper condition."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model_compression.src.orchid.edge_audit import checkpoint_audit

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", action="append", required=True, help="Repeat once per packaged neural model.")
parser.add_argument("--output", required=True)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--trials", type=int, default=50)
args = parser.parse_args()
print(checkpoint_audit(args.checkpoint, args.output, warmup=args.warmup, trials=args.trials))
