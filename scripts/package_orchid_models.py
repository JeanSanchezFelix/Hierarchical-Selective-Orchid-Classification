#!/usr/bin/env python3
"""Create a compressed router-plus-experts model pack from deployment JSON."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model_compression.src.orchid.model_packs import create_model_pack

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--model-directory", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
print(create_model_pack(manifest, args.model_directory, args.output))
