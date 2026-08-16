#!/usr/bin/env python3
"""Export one checkpoint bundle to LiteRT and emit its deployment entry JSON."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model_compression.src.orchid.deployment_manifest import write_deployment_manifest
from model_compression.src.orchid.export import export_checkpoint_to_litert

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--role", choices=["router", "expert"], required=True)
parser.add_argument("--genus")
parser.add_argument("--entry-output", required=True)
args = parser.parse_args()
entry = export_checkpoint_to_litert(args.checkpoint, args.output, role=args.role, genus_id=args.genus)
write_deployment_manifest(args.entry_output, {"schema_version": 1, "models": [entry]})
print(json.dumps(entry, indent=2))
