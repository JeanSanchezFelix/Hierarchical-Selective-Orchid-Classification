# Public iNaturalist Orchidaceae Dataset

This Phase 2 tool prepares the paper dataset locally from the official
iNaturalist Licensed Observation Images metadata snapshot. It does not use the
private `data/taxonomic-orchid/` collection and does not download all
Orchidaceae images.

## Contract

The checked-in selection contract is
`configs/orchid/public_orchid_dataset.yaml`:

- target: approximately 50,000 images from 200 species;
- accepted photos: CC0 and CC-BY only;
- one compatible primary photo per research-grade observation;
- 40 to 500 images per species, with at most 25 images per observer per species;
- `Genus/Species/image` local layout;
- train/validation/calibration/test split grouped by observation ID.

The exact counts may differ only when the tool fails with a clear error. Do not
reduce target counts or selection limits after seeing test metrics.

## 1. Fetch A Pinned Metadata Snapshot

Choose a dated official snapshot. Do not use `latest` for paper data because it
can change. The official metadata consists of `observations.csv.gz`,
`photos.csv.gz`, and `taxa.csv.gz`.

```bash
python scripts/prepare_public_orchid_dataset.py fetch-metadata \
  --metadata-dir data/inaturalist-metadata-YYYYMMDD \
  --snapshot YYYYMMDD
```

The snapshot archive can be large. It remains under ignored `data/` and is not
committed. Verify the extracted directory contains all three required files.

## 2. Build And Validate The Manifest

This step reads metadata and writes only the selected dataset manifests. It
does not download image files.

```bash
python scripts/prepare_public_orchid_dataset.py build-manifest \
  --metadata-dir data/inaturalist-metadata-YYYYMMDD \
  --source-snapshot YYYYMMDD

python scripts/prepare_public_orchid_dataset.py validate
```

The output root is `data/orchidaceae-inat-v1/` by default. Its `manifests/`
directory contains taxonomy, licenses, checksums, selected observations, image
URLs, and the frozen four-way split.

## 3. Download The Selected Images

Run this only after reviewing the generated dataset card and manifests. It is
resumable: existing files are retained and failed files are reported in
`manifests/checksums.csv`.

```bash
python scripts/prepare_public_orchid_dataset.py download-images \
  --workers 8
```

Do not use `--overwrite` unless deliberately replacing a local acquisition.
The source image URLs are medium-sized versions in the official public S3
bucket. Attribution and licenses remain in `manifests/images.csv`.

## 4. Train One Paper Condition

After Phase 4, the shared single-model runner uses the public root by default
and records dataset/manifests/configuration provenance beside every checkpoint:

```bash
python scripts/run_orchid_experiment.py all \
  --config configs/orchid/paper_experiment_template.yaml \
  --seed 17
```

Use `--dataset-root` and `--split-manifest` only to switch to a separately
frozen dataset version. Never use different datasets for methods in the same
comparison table.

## Reproducibility Boundary

The paper cites the source snapshot and publishes this configuration, code, and
generated manifests. The images remain local and ignored by Git. A later release
decision can provide the manifest and acquisition procedure without redistributing
the photo bytes.
