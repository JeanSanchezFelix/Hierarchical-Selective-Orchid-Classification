#!/usr/bin/env python3
"""Record parity from identically ordered PyTorch and LiteRT logits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model_compression.src.orchid.edge_audit import litert_parity

parser = argparse.ArgumentParser()
parser.add_argument("--torch-logits", required=True, help="NxC .npy from the frozen PyTorch checkpoint.")
parser.add_argument("--litert-logits", required=True, help="Matching NxC .npy from the exported LiteRT model.")
parser.add_argument("--output", required=True)
args = parser.parse_args()
report = litert_parity(np.load(args.torch_logits), np.load(args.litert_logits))
Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(args.output)
