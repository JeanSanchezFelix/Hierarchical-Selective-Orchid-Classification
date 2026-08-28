# Orchid Paper: Start-to-Finish Runbook

This is the sole execution guide for the public Orchidaceae paper. Run every
command from a Bash shell at the repository root. It does not use the private
dataset, Android application, or physical-device claims.

## 0. One-Time Environment Setup

```bash
cd /path/to/Hierarchical-Selective-Orchid-Classification
conda create -n orchid_edge python=3.12 -y
conda activate orchid_edge
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_orchid_*.py" -v
```

If `conda activate` is unavailable, use `conda run --no-capture-output -n
orchid_edge python` in place of `python` below.

## 1. Freeze the Public Dataset

Choose a dated official iNaturalist metadata snapshot. Replace `YYYYMMDD` with
the exact snapshot identifier you selected; never use a moving `latest` source.
Run each stage only after the preceding command succeeds; a failed download means the manifests do not yet exist.

```bash
snapshot="YYYYMMDD"
metadata="data/inaturalist-metadata-${snapshot}"
dataset="data/orchidaceae-inat-v1"

python scripts/prepare_public_orchid_dataset.py fetch-metadata \
  --metadata-dir "$metadata" \
  --snapshot "$snapshot"

python scripts/prepare_public_orchid_dataset.py build-manifest \
  --metadata-dir "$metadata" \
  --source-snapshot "$snapshot" \
  --output-root "$dataset"

python scripts/prepare_public_orchid_dataset.py validate \
  --output-root "$dataset"
```

Review the manifests under `${dataset}/manifests` before downloading. Downloading
the selected roughly 50,000 image files is required for training and is resumable.

```bash
python scripts/prepare_public_orchid_dataset.py download-images \
  --output-root "$dataset" \
  --workers 8

python scripts/prepare_public_orchid_dataset.py validate \
  --output-root "$dataset"
```

Do not start training until `download-images` completes successfully and the final
`validate` command succeeds. Until then, the manifest has no local Genus/Species
image folders. Do not change the dataset configuration or manifest after inspecting
model results. All methods must use the same frozen root and split file.

## 2. Run All Single-Model Conditions

The five trainable conditions are `flat_ce`, `flat_balanced_softmax`,
`flat_hsc`, `dual_head`, and `dual_head_taxonomy_hsc` (Ours). The command
trains, calibrates, and evaluates one condition for one seed.

```bash
set -euo pipefail

root="data/orchidaceae-inat-v1"
manifest="${root}/manifests/split.csv"
experiment="public-50k/orchid-hsc-paper"
seeds=(17 42 123)
methods=(flat_ce flat_balanced_softmax flat_hsc dual_head dual_head_taxonomy_hsc)

for seed in "${seeds[@]}"; do
  for method in "${methods[@]}"; do
    python scripts/run_orchid_experiment.py all \
      --config configs/orchid/paper_experiment_template.yaml \
      --dataset-root "$root" \
      --split-manifest "$manifest" \
      --experiment-id "$experiment" \
      --method "$method" \
      --seed "$seed"
  done
done
```

Each run writes its checkpoint, calibration policy, metrics, image-level
predictions, and `reports/training.log` to
`artifacts/orchid/$experiment/<method>/seed-<seed>/`. The log includes any
Python traceback if that run fails.

### Post-hoc HSC comparison

`flat_hsc` is a post-hoc condition and must reuse the trained
`flat_balanced_softmax` checkpoint. To regenerate the forced B1 reports and the
HSC B2 reports without training, run:

```bash
for seed in 17 42 123; do
  checkpoint="artifacts/orchid/${experiment}/flat_balanced_softmax/seed-${seed}/checkpoints/best_orchid_model.pt"

  for method in flat_balanced_softmax flat_hsc; do
    python scripts/run_orchid_experiment.py all \
      --config configs/orchid/paper_experiment_template.yaml \
      --dataset-root "$root" \
      --split-manifest "$manifest" \
      --experiment-id "$experiment" \
      --method "$method" \
      --seed "$seed" \
      --checkpoint "$checkpoint"
  done
done
```

B1 reports contain `"hsc_enabled": false`; B2 reports contain
`"hsc_enabled": true`. Rerun the paper-summary command after all six reports
finish.

## 3. Run Both Cascade Controls

This trains one router and every genus expert, then evaluates C1 top-1 and C2
top-2 routing for each seed. The cascade config must point at the same frozen
root and split manifest. The following command writes a local runtime copy
without changing the checked-in template.

```bash
cascade_config="artifacts/orchid/paper_cascade_runtime.yaml"
sed \
  -e "s|^  root_dir: .*|  root_dir: ${root}|" \
  -e "s|^  split_manifest: .*|  split_manifest: ${manifest}|" \
  configs/orchid/paper_cascade_template.yaml > "$cascade_config"

for seed in "${seeds[@]}"; do
  python scripts/run_orchid_cascade.py --config "$cascade_config" --seed "$seed"
done
```

Expected reports:

```text
artifacts/orchid/public-50k/orchid-hsc-paper/cascade_top1/seed-<seed>/reports/
artifacts/orchid/public-50k/orchid-hsc-paper/cascade_top2/seed-<seed>/reports/
```

The router and each expert retain their own `reports/training.log`. The final
cascade evaluation is written to
`artifacts/orchid/$experiment/cascade_reports/seed-<seed>/training.log`,
including a traceback if evaluation fails.

## 4. Aggregate Paper Results

Run only after all 21 method-seed evaluations are complete.

```bash
python scripts/summarize_orchid_paper_results.py \
  --matrix configs/orchid/paper_matrix.yaml \
  --artifact-root artifacts/orchid \
  --output-dir artifacts/orchid/paper_summary
```

This creates `seed_metrics.csv`, `paired_bootstrap.json`,
`risk_coverage.csv`, and `paper_table_hAURC.md`.

## 5. Record Edge-Ready Evidence

Use host CPU only; do not call this Android performance. Audit every frozen
condition. For a single model, pass one checkpoint. For cascades, pass the
router plus every expert checkpoint; the report records the correct model count
and inference-call count.

Generate the complete 7-condition by 3-seed audit matrix (21 reports):

```bash
python scripts/audit_orchid_edge_matrix.py \
  --experiment-root artifacts/orchid/public-50k/orchid-hsc-paper \
  --output-dir artifacts/orchid/edge_audits/all_seeds \
  --seeds 17 42 123 \
  --warmup 10 \
  --trials 50
```

The combined table is written to
`artifacts/orchid/edge_audits/all_seeds/README.md`, with machine-readable rows
in `edge_metrics.csv` and one JSON report per method and seed. Cascade package
footprint includes the router and all 68 experts; latency uses a documented
conservative proxy comprising the router and the largest one or two expert
heads for top-1 or top-2 routing, respectively.

To audit a single checkpoint instead, run:

```bash
python scripts/audit_orchid_edge.py \
  --checkpoint artifacts/orchid/public-50k/orchid-hsc-paper/dual_head_taxonomy_hsc/seed-17/checkpoints/best_orchid_model.pt \
  --output artifacts/orchid/edge_audits/ours_seed17.json
```

Export only a frozen single-model checkpoint to FP32 LiteRT when conversion is
available. Record parity from identically ordered logits; do not report INT8
results until an INT8 conversion and matching parity evaluation have completed.

```bash
python scripts/export_orchid_paper_litert.py \
  --checkpoint artifacts/orchid/public-50k/orchid-hsc-paper/dual_head_taxonomy_hsc/seed-17/checkpoints/best_orchid_model.pt \
  --output artifacts/orchid/exports/ours_seed17_fp32.tflite

python scripts/generate_orchid_litert_logits.py \
  --checkpoint artifacts/orchid/public-50k/orchid-hsc-paper/dual_head_taxonomy_hsc/seed-17/checkpoints/best_orchid_model.pt \
  --litert-model artifacts/orchid/exports/ours_seed17_fp32.tflite \
  --torch-logits artifacts/orchid/parity/ours_seed17_torch_logits.npy \
  --litert-logits artifacts/orchid/parity/ours_seed17_litert_logits.npy

python scripts/compare_orchid_litert_logits.py \
  --torch-logits artifacts/orchid/parity/ours_seed17_torch_logits.npy \
  --litert-logits artifacts/orchid/parity/ours_seed17_litert_logits.npy \
  --output artifacts/orchid/parity/ours_seed17_fp32.json
```

## 6. Populate and Build the WACV Manuscript

Run these commands from the separate manuscript repository.

```bash
cd /path/to/WACV

python tools/export_orchid_results_to_wacv.py \
  --summary-dir /path/to/Hierarchical-Selective-Orchid-Classification/artifacts/orchid/paper_summary \
  --tex-output sec/generated_results.tex \
  --figure-output figures/hierarchical_risk_coverage.pdf

pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Replace approximate dataset counts and pre-results language only with values
from the frozen manifests and generated artifacts. Keep host CPU and Android
claims separate.

## 7. Run the Submission Gate

```bash
cd /path/to/Hierarchical-Selective-Orchid-Classification
python scripts/validate_orchid_submission.py \
  --matrix configs/orchid/paper_matrix.yaml \
  --artifact-root artifacts/orchid \
  --dataset-manifest data/orchidaceae-inat-v1/manifests/split.csv
```

`SUBMISSION GATE PASSED` confirms the required dataset, predictions, summary,
and paired-bootstrap files exist. It does not validate scientific quality or
grant permission to make unsupported claims.

## Non-Negotiable Boundaries

- Never tune on the test split.
- Never compare methods trained on different manifests.
- `Unknown` is known-class uncertainty abstention, not non-orchid detection.
- Host CPU measurements are not Android or real-time measurements.
- Do not claim INT8 performance without a separate INT8 conversion and parity
  report.
