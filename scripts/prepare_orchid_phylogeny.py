#!/usr/bin/env python3
"""Generate a reviewed mapping template and verify the cited phylogeny archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_compression.src.orchid.phylogeny import PEREZ_ESCOBAR_2024
from model_compression.src.orchid.taxonomy import scan_orchid_taxonomy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", required=True)
parser.add_argument("--mapping-output", default="artifacts/orchid/phylogeny/species_mapping.csv")
parser.add_argument("--archive", default="data/phylogeny/perez_escobar_2024/10.PPtrees.AAR.rar")
args = parser.parse_args()

archive = Path(args.archive)
if archive.is_file():
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest.lower() != PEREZ_ESCOBAR_2024["tree_archive_md5"]:
        raise ValueError(f"Archive checksum mismatch: {digest}")
    print(f"Verified {archive} against Pérez-Escobar et al. (2024) Figshare data DOI {PEREZ_ESCOBAR_2024['data_doi']}")
else:
    print(f"Source archive not found: {archive}")

taxonomy = scan_orchid_taxonomy(args.dataset_root)
output = Path(args.mapping_output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["species_id", "genus_id", "species_name", "source_tip", "mapping_status", "mapping_note"])
    writer.writeheader()
    for record in taxonomy.records:
        writer.writerow({"species_id": record.species_id, "genus_id": record.genus_id, "species_name": record.species_name, "source_tip": "", "mapping_status": "unreviewed", "mapping_note": ""})
print(f"Wrote review-required mapping template: {output}")
