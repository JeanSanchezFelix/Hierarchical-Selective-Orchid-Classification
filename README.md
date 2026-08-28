# Hierarchical Selective Orchid Classification

This repository contains the implementation for taxonomy-consistent
hierarchical selective classification of Orchidaceae images. It supports a
public iNaturalist-based orchid benchmark, flat and dual-head MobileNetV2
models, hierarchical selective classification (HSC), and genus-routed cascade
controls.

- **Public dataset preparation** from a pinned iNaturalist metadata snapshot
- **Training** for flat, dual-head, and genus-expert cascade models
- **Hierarchical selective prediction** with calibration and abstention
- **Evaluation** and paper-result reporting

## Project Structure

```text
repository/
├── configs/
│   └── orchid/                       # Dataset, experiment, and cascade configurations
├── data/                             # Local metadata, images, and frozen manifests (ignored)
├── datasets/                         # Orchid PyTorch Dataset implementations
│   ├── CustomHierarchicalDataset.py
│   ├── TaxonomicOrchidDataset.py
│   └── registry.py
├── model_compression/           
│   └── src/
│       ├── orchid/                   # Models, training, HSC, routing, and evaluation
│       │   ├── calibration.py
│       │   ├── evaluation.py
│       │   ├── experiment.py
│       │   ├── models.py
│       │   ├── routing.py
│       │   └── taxonomy.py
│       └── utils/                    # Shared preprocessing, metrics, and logging utilities
├── scripts/                          # Dataset, experiment, cascade, export, and audit CLIs
│   ├── prepare_public_orchid_dataset.py
│   ├── run_orchid_experiment.py
│   ├── run_orchid_cascade.py
│   ├── summarize_orchid_paper_results.py
│   └── audit_orchid_edge.py
├── tests/                            # Orchid workflow and metric tests
├── docs/                             # Setup, dataset, and paper protocol documentation
├── ORCHID_DELIVERABLE.md             # Supported research workflow
└── requirements.txt                  # Python dependencies
```

## Key Components

### 1. Data & Datasets

- **`scripts/prepare_public_orchid_dataset.py`**: Builds a frozen local
  Orchidaceae dataset from a dated iNaturalist metadata snapshot.
- **`data/`**: Stores local metadata, images, and manifests in
  `Genus/Species/image` layout; its contents are excluded from Git.
- **`datasets/`**: Provides taxonomy-aware PyTorch datasets for the orchid
  experiments.

### 2. Training

- **`scripts/run_orchid_experiment.py`**: Trains, calibrates, and evaluates a
  single-model condition for one seed.
- **Single-model conditions**: Flat cross-entropy, flat Balanced Softmax,
  post-hoc flat HSC, dual-head, and dual-head taxonomy HSC.
- **`scripts/run_orchid_cascade.py`**: Trains the genus router and genus
  experts used by the top-1 and top-2 cascade controls.

### 3. Hierarchical Selective Classification

- **Dual-head model**: Produces species and genus predictions in one forward
  pass.
- **Calibration**: Fits temperatures and decision thresholds on the dedicated
  calibration split.
- **Decision policy**: Returns a species when confident, otherwise its genus
  when supported, or `Unknown` when the model abstains.

### 4. Evaluation

- **`evaluation.py`**: Computes species, genus, calibration, and hierarchical
  selective metrics.
- **`scripts/summarize_orchid_paper_results.py`**: Aggregates paired,
  multi-seed results into paper-ready tables and reports.
- **`tools/orchid_split_audit.py`**: Audits frozen dataset splits and
  manifests.

### 5. Edge-Readiness & Export

- **`scripts/audit_orchid_edge.py`**: Records host-CPU edge-readiness evidence
  for frozen checkpoints.
- **LiteRT utilities**: Export a frozen single model and compare PyTorch and
  LiteRT logits for parity.
- Host-CPU measurements are deployment proxies; this project does not make
  Android or physical-device performance claims.

## How to Use

### Orchid hierarchical pipeline

The reproducible public-dataset workflow and complete paper experiment protocol
are documented in [docs/orchid_setup.md](docs/orchid_setup.md). Private images
and generated model artifacts remain excluded from version control.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Hierarchical-Selective-Orchid-Classification.git
cd Hierarchical-Selective-Orchid-Classification
```

### 2. Install Dependencies

```bash
conda create -n orchid_edge python=3.12 -y
conda activate orchid_edge
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Prepare the Public Dataset

Choose a dated iNaturalist metadata snapshot and build the dataset manifest.

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

Review the manifests before downloading images. The complete dataset procedure
is in [docs/orchid_public_dataset.md](docs/orchid_public_dataset.md).

### 4. Train and Evaluate a Model

Run one experiment condition with `run_orchid_experiment.py`.

```bash
python scripts/run_orchid_experiment.py all \
    --config configs/orchid/paper_experiment_template.yaml \
    --dataset-root data/orchidaceae-inat-v1 \
    --split-manifest data/orchidaceae-inat-v1/manifests/split.csv \
    --experiment-id public-50k/orchid-hsc-paper \
    --method dual_head_taxonomy_hsc \
    --seed 17
```

The run saves its checkpoint, calibration policy, metrics, predictions, and log
under `artifacts/orchid/`.

#### Example: Run a different condition

```bash
python scripts/run_orchid_experiment.py all \
    --config configs/orchid/paper_experiment_template.yaml \
    --method flat_balanced_softmax \
    --seed 42
```

### 5. Run the Full Study

Use [docs/orchid_setup.md](docs/orchid_setup.md) for the three-seed experiment
matrix, cascade controls, result aggregation, edge audit, and LiteRT parity
steps.

## Notes

- The public benchmark uses the same frozen dataset and split manifest for all
  compared methods.
- The standard paper seeds are `17`, `42`, and `123`.
- `model_compression/` is a legacy source-package name, not the scope of this
  repository.
- See [docs/orchid_paper_protocol.md](docs/orchid_paper_protocol.md) for the
  research scope, evaluation rules, and reproducibility contract.
