# Dataset split and near-duplicate audit

`orchid_split_audit.py` creates an image-stratified split manifest for an
orchid dataset laid out as `Genus/Species/image`, then identifies cross-split
duplicate candidates for human review.

It detects exact file duplicates and screens for visually similar images using
a 64-bit difference hash. It does **not** prove a specimen-disjoint split and
will miss many distinct-angle photographs. Do not claim specimen-disjoint
evaluation unless you have independent specimen or capture-session IDs.

## Create a manifest

```powershell
python tools/orchid_split_audit.py create-manifest `
  --dataset-root data/taxonomic-orchid/TaxonomicOrchidDataset `
  --output artifacts/leakage_audit/orchid_split.csv
```

The script creates the CSV plus an adjacent JSON summary. Every image path is
relative to `--dataset-root`, so the manifest can be shared internally without
embedding an absolute local path.

Classes with one or two images cannot appear in all three splits. The script
marks those rows with `tiny_class_not_all_splits`; do not use them to claim
per-class test performance.

## Audit for cross-split candidates

```powershell
python tools/orchid_split_audit.py audit `
  --dataset-root data/taxonomic-orchid/TaxonomicOrchidDataset `
  --manifest artifacts/leakage_audit/orchid_split.csv `
  --output artifacts/leakage_audit/candidates.csv
```

Review `candidates.csv` before altering the split. A confirmed duplicate
cluster must be assigned to one split as a group, followed by a repeat audit.
The default audit only compares images carrying the same `genus::species`
identifier; use `--no-same-species-only` to additionally find possible
cross-label errors.

The `artifacts/leakage_audit/` directory should remain private: it can expose
dataset paths and the split assignment for an unreleased dataset.

## Train with the reviewed manifest

`main.py` accepts the manifest directly. Supplying it disables automatic
splitting, including `random_split`, and uses the exact reviewed assignment:

```powershell
python main.py --dataset TaxonomicOrchid --split_manifest artifacts/leakage_audit/orchid_split.csv
```

The manifest must cover every image exposed by the configured dataset exactly
once. This check deliberately fails when the trainer's dataset configuration
filters classes or exposes a different dataset root; regenerate the manifest
for the exact training population rather than silently dropping rows.
