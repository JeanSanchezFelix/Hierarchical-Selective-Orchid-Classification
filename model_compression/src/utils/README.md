# Utils Module

The `utils` module provides foundational functionality supporting data loading, model setup, training orchestration, logging, metrics visualization, benchmarking, and callback management for the `model_compression` project.

---

## Folder Structure
```plaintext
utils/
├── __init__.py
├── callbacks/          # Callback classes and registry
├── data_imbalance.py   # Compute class weights and samplers
├── logging_setup.py    # Configure Python logging
├── metrics.py          # Compute and plot evaluation metrics
├── model_setup.py      # Instantiate models, loss, and optimizers
├── parsing.py          # CLI and config file argument parsing
├── preprocessing.py    # Dataset transforms and DataLoader utilities
└── benchmarking.py     # Measure performance, memory, power, and latency
```

---

## Modules Overview

### `model_setup.py`
- **Functions:**
  - `setup_model` — Initialize torchvision model with custom head and optional weights.
  - `setup_criterion` — Create loss function (with optional class weights).
  - `setup_optimizer` — Instantiate optimizer (Adam, SGD, etc.).
  - `tf_setup` — Full setup: model, criterion, optimizer.

### `preprocessing.py`
- **Functions:**
  - `load_data` — Load dataset by name, apply transforms, split/folder logic, and return `DataLoader`s.
  - `_log_dataset_statistics` / `_log_all_statistics` — Log sample counts and class distributions.

### `data_imbalance.py`
- **Functions:**
  - `calculate_model_weights` — Compute per-class weights for imbalanced datasets.
  - `get_weighted_sampler` — Create `WeightedRandomSampler` to oversample minority classes.

### `logging_setup.py`
- **Function:**
  - `configure_logging` — Set up file and/or console logging with standardized format.

### `parsing.py`
- **Functions:**
  - `parse` — Parse CLI args and optional config files (CSV/JSON/YAML), validate, and return configuration dict.
  - `_load_args_from_file` — Load and merge arguments from external file.
  - `_validate_args` — Ensure numeric and path constraints.

### `metrics.py`
- **Functions:**
  - `calculate_metrics` — Accuracy, precision, recall, F1, AUC.
  - Plotting utilities: `plot_metric_bar`, `plot_confusion_matrix`, `plot_train_val_curve`, `plot_roc_auc_curve`, `plot_calibration_curve`, `plot_log_loss`.

### `benchmarking.py`
- **Functions:**
  - `_seed_everything` — Reproducible seeds.
  - `_warm_up` — Untimed inference.
  - `measure_inference_performance` — Latency and throughput.
  - `calculate_speedup` — Compare two models.
  - Memory and power profiling: `measure_memory_usage`, `measure_idle_power_consumption`, `measure_power_consumption`.
  - `model_size_mb` — Disk footprint.
  - `measure_latency_percentiles` — P50/P95/P99.
  - `measure_throughput_per_watt` — Energy efficiency.
  - `benchmark` — Comprehensive comparison table.

### `callbacks/`
- **Subpackage** exporting:
  - `Callback`, `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, `LRScheduler`
  - `process_callbacks`, `CALLBACK_REGISTRY`

---

## Usage Examples

### Data Loading and Preprocessing
```python
from model_compression.src.utils import load_data
loaders = load_data(
    dataset_name='MonkeyPox',
    batch_size=32,
    use_augmentation=True,
    use_sampler=True
)
train_loader = loaders['train']
```

### Model Setup
```python
from model_compression.src.utils import tf_setup
model, loss_fn, optimizer = tf_setup(
    model_name='resnet18',
    pretrained_weights_path=None,
    dataloader=train_loader,
    criterion_name='cross_entropy',
    optimizer_name='adam',
    learning_rate=1e-3
)
```

### Logging
```python
from model_compression.src.utils import configure_logging
configure_logging(enable_console=True, log_dir='./logs')
```

### Argument Parsing
```python
from model_compression.src.utils import parse
config = parse()
# config['CALLBACKS'] contains instantiated callbacks
```

### Metrics Visualization
```python
from model_compression.src.utils.metrics import plot_calibration_curve, plot_log_loss
plot_calibration_curve(y_true, y_proba, save_path='calib.png')
plot_log_loss({'train': train_losses, 'val': val_losses}, save_path='logloss.png')
```

### Benchmarking
```python
from model_compression.src.utils.benchmarking import benchmark
benchmark(model_a, model_b, val_loader, device)
```

### Callbacks
```python
from model_compression.src.utils.callbacks import process_callbacks
callbacks = process_callbacks(config)
# Use callbacks in training loop
```

---

For detailed API and advanced options, refer to the source code in the `utils` folder.