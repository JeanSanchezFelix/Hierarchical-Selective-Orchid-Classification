#!/usr/bin/env python3
"""Export a frozen flat or dual-head paper checkpoint to LiteRT."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model_compression.src.converters.to_tflite import convert_pytorch_model_to_tflite
from model_compression.src.orchid.edge_audit import load_auditable_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
model, metadata = load_auditable_model(args.checkpoint)
example = torch.zeros((1, 3, int(metadata["img_size"]), int(metadata["img_size"])))
convert_pytorch_model_to_tflite(model, (example,), args.output, torch.device("cpu"))
if not Path(args.output).is_file() or Path(args.output).stat().st_size == 0:
    raise RuntimeError("LiteRT export did not produce a non-empty file.")
print(Path(args.output))
