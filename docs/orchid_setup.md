# Orchid classifier: beginner run guide

This guide runs the complete research workflow from a private image folder to
LiteRT model-pack artifacts. Run commands from PowerShell. The dataset remains
private; only derived artifacts are written under `artifacts/orchid/`.

## 1. Open the repository

```powershell
Set-Location "C:\Users\jampi\VS_Codes\UPRM_Code\ML_Projects\model-compression"
```

## 2. Create or use the Python environment

The pinned dependencies target Python 3.12. If `orchid_edge` already exists,
skip the creation command.

```powershell
conda create -n orchid_edge python=3.12 -y
conda activate orchid_edge
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

If `conda activate` is unavailable in the terminal, prefix every Python command
in this guide with `conda run -n orchid_edge`, for example:

```powershell
conda run -n orchid_edge python scripts\train_orchid_router.py
```

The dependency file requests CUDA 12.4 PyTorch wheels. A CUDA-capable NVIDIA
GPU is optional; the workflow can run on CPU, but training will take longer.

## 3. Place and verify the private dataset

The expected hierarchy is exactly:

```text
taxonomic-orchid/
  Bletia/
    Bti. patula/
      image_001.jpg
  Cattleya/
    C. trianae/
      image_002.jpg
```

By default, the configs expect this directory:

```text
datasets\taxonomic-orchid
```

If your dataset is elsewhere, edit `dataset.root_dir` in all three files before
continuing:

```text
configs\orchid\baseline_flat.yaml
configs\orchid\genus_router.yaml
configs\orchid\expert_template.yaml
```

Use forward slashes in YAML paths, for example:

```yaml
root_dir: C:/OrchidData/taxonomic-orchid
```

Do not put unlabeled non-orchid images inside a genus or species directory.
Put them in a top-level `UNLABELED/` directory if you keep them beside the
dataset; the taxonomy scanner ignores that directory.

## 4. Create and audit the frozen train/validation/test split

Set your dataset path once for the current PowerShell session:

```powershell
$datasetRoot = "C:\Users\jampi\VS_Codes\UPRM_Code\ML_Projects\model-compression\datasets\taxonomic-orchid"
```

Create the split manifest:

```powershell
python tools\orchid_split_audit.py create-manifest `
  --dataset-root $datasetRoot `
  --output artifacts\leakage_audit\orchid_split.csv
```

Audit exact and visually similar cross-split images:

```powershell
python tools\orchid_split_audit.py audit `
  --dataset-root $datasetRoot `
  --manifest artifacts\leakage_audit\orchid_split.csv `
  --output artifacts\leakage_audit\duplicate_candidates.csv
```

Open and review `artifacts\leakage_audit\duplicate_candidates.csv`. Repeated
captures of the same plant can leak into different splits. Correct the CSV by
hand if necessary, then keep it frozen. Do **not** rerun `create-manifest` over
an existing reviewed CSV unless you intentionally begin a new experiment.

## 5. Run a quick environment check

```powershell
python -m unittest discover -s tests -v
```

Resolve test failures before training. This command does not train a model.

## 6. Train the required models

Run these in order. Check the generated checkpoint after each command before
starting the next one.

### B0 — flat species baseline

This is the single 199-species MobileNetV2 comparison.

```powershell
python scripts\train_orchid_baseline.py
```

Expected checkpoint:

```text
artifacts\orchid\baseline_flat\mobilenet_v2\checkpoints\best_orchid_model.pt
```

### B1 — genus router

This model predicts the genus before expert routing.

```powershell
python scripts\train_orchid_router.py
```

Expected checkpoint:

```text
artifacts\orchid\genus_router\mobilenet_v2\checkpoints\best_orchid_model.pt
```

### H1/H2 — genus species experts

Train one expert first. Replace `Phalaenopsis` with a represented genus if you
prefer a different initial smoke test.

```powershell
python scripts\train_orchid_experts.py --genus Phalaenopsis
```

Check its `taxonomy.json`, `run_metadata.json`, and
`checkpoints\best_orchid_model.pt`. Then train every genus:

```powershell
python scripts\train_orchid_experts.py --genus all
```

A genus with exactly one species is deterministic after genus routing. It should
be represented as deterministic routing in the mobile deployment rather than
presented as evidence from a one-class learned model.

## 7. Evaluate the cascade on the held-out test split

Run evaluation only after choosing the final router and all required specialists
using validation data. Do not select thresholds or architectures from test
results.

```powershell
python scripts\run_orchid_evaluation.py `
  --router-checkpoint artifacts\orchid\genus_router\mobilenet_v2\checkpoints\best_orchid_model.pt `
  --expert-checkpoint "Phalaenopsis=artifacts\orchid\species_experts\mobilenet_v2\Phalaenopsis\checkpoints\best_orchid_model.pt" `
  --expert-checkpoint "Dendrobium=artifacts\orchid\species_experts\mobilenet_v2\Dendrobium\checkpoints\best_orchid_model.pt"
```

Add one `--expert-checkpoint "Genus=path"` line for every trained expert. The
reports are written to:

```text
artifacts\orchid\cascade_evaluation\top2\reports\
```

Keep both `metrics.json` and `predictions.csv`. The default command evaluates
top-2 routing. Add `--unknown-policy path\to\unknown_policy.json` only after
you have fitted and frozen that policy from validation predictions; do not fit it
on the test split.

## 8. Optional phylogenetic error analysis

First create the mapping for the private taxonomy:

```powershell
python scripts\prepare_orchid_phylogeny.py `
  --dataset-root $datasetRoot
```

After reviewing `artifacts\orchid\phylogeny\species_mapping.csv`, add both
arguments to the evaluation command:

```powershell
  --phylogeny-tree-directory data\phylogeny\perez_escobar_2024\extracted `
  --phylogeny-mapping artifacts\orchid\phylogeny\species_mapping.csv
```

Use this as an error-severity analysis only. It does not prove the visual model
learned phylogeny.

## 9. Export frozen winning models to LiteRT

Export the router:

```powershell
python scripts\export_orchid_litert.py `
  --checkpoint artifacts\orchid\genus_router\mobilenet_v2\checkpoints\best_orchid_model.pt `
  --output artifacts\orchid\exports\router-genus.tflite `
  --role router `
  --entry-output artifacts\orchid\exports\router-entry.json
```

Export one expert (repeat for every expert):

```powershell
python scripts\export_orchid_litert.py `
  --checkpoint artifacts\orchid\species_experts\mobilenet_v2\Phalaenopsis\checkpoints\best_orchid_model.pt `
  --output artifacts\orchid\exports\Phalaenopsis.tflite `
  --role expert `
  --genus Phalaenopsis `
  --entry-output artifacts\orchid\exports\Phalaenopsis-entry.json
```

## 10. Build the deployment manifest and compressed pack

Merge the router and every expert entry manifest:

```powershell
python scripts\build_orchid_deployment_manifest.py `
  --entry-manifest artifacts\orchid\exports\router-entry.json `
  --entry-manifest artifacts\orchid\exports\Phalaenopsis-entry.json `
  --output artifacts\orchid\exports\deployment_manifest.json
```

Add an `--entry-manifest` argument for every remaining genus. Then package the
models. `--model-directory` must contain the exported `.tflite` files referenced
by the deployment manifest.

```powershell
python scripts\package_orchid_models.py `
  --manifest artifacts\orchid\exports\deployment_manifest.json `
  --model-directory artifacts\orchid\exports `
  --output artifacts\orchid\exports\orchid_models.pack
```

Copy only the validated deployment manifest and generated pack(s) into the
Android project. Then rerun the physical-device benchmark with real orchid input
images; the placeholder-pack latency measurements are not paper results.

## Common problems

| Problem | What to check |
| --- | --- |
| `Split manifest not found` | `dataset.split_manifest` in the YAML must point to the reviewed CSV. |
| `FileNotFoundError` for data | Correct `dataset.root_dir` in all orchid YAML configs. |
| CUDA / PyTorch installation fails | Verify Python 3.12 and your NVIDIA driver; a CPU-compatible PyTorch install is acceptable for a slower run. |
| Expert has zero or one class | Check the folder hierarchy and the split CSV; one-species genera should be deterministic. |
| Android pack fails checksum | Re-export and rebuild the deployment manifest and pack together; never mix files from separate export runs. |
