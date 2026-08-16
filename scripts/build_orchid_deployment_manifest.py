#!/usr/bin/env python3
"""Merge router/expert entry manifests into one deployable cascade manifest."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model_compression.src.orchid.deployment_manifest import make_deployment_manifest, write_deployment_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--entry-manifest", action="append", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
entries = []
for path in args.entry_manifest:
    entries.extend(json.loads(Path(path).read_text(encoding="utf-8"))["models"])
manifest = make_deployment_manifest(entries, {"assembly": "build_orchid_deployment_manifest.py"})
print(write_deployment_manifest(args.output, manifest))
