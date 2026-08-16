# Model Compression Engine

This repository contains the implementation of a model compression engine for biomedical applications, such as anemia detection through conjunctiva pallor, monkeypox classification, and skin lesion detection. The project includes pre-trained models, custom models, and utilities for training, evaluation, and preprocessing.

- **Training** (transfer learning, knowledge distillation)
- **Quantization** (Post-Training Quantization, Quantization-Aware Training)
- **Model conversion** (ONNX, TFLite, ExecuTorch)
- **Evaluation** and **visualization**

---

## Project Structure

```
repository/
├── data/                         # Raw image datasets
│   ├── cp-anemia/
│   ├── monkeypox/
│   ├── skin-lesions/
│   └── taxonomic-orchid/
├── datasets/                     # PyTorch Dataset implementations
│   ├── CpAnemiaDataset.py
│   ├── MonkeypoxDataset.py
│   ├── SkinCancerDataset.py
│   ├── TaxonomicOrchidDataset.py
│   └── registry.py               # Map names → dataset classes
├── model_compression/            # Core package
│   └── src/
│       ├── converters/           # Model conversion
│       │   ├── __init__.py
│       │   ├── to_executorch.py
│       │   ├── to_onnx.py
│       │   └── to_tflite.py
│       ├── eval/                 # Evaluation modules for PyTorch, ONNX, TF, TFLite
│       │   ├── __init__.py
│       │   ├── onnx_eval.py
│       │   ├── predictions.py
│       │   ├── pytorch_eval.py
│       │   ├── tf_eval.py
│       │   └── tflite_eval.py
│       ├── quantization/         # Quantization submodules
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   ├── quantize.py
│       │   └── utils/
│       │       ├── __init__.py
│       │       ├── calibration.py
│       │       ├── inspect.py
│       │       ├── model_setup.py
│       │       ├── post_training_quantization.py
│       │       └── validate.py
│       ├── tensorrt/             
│       ├── train/                # Training scripts
│       │   ├── __init__.py
│       │   ├── knowledge_distillation.py
│       │   └── train.py
│       ├── utils/                # Training & evaluation utilities
│       │   ├── callbacks/
│       │   │   ├── callbacks.py
│       │   │   └── registry.py
│       │   ├── __init__.py
│       │   ├── benchmarking.py
│       │   ├── data_imbalance.py
│       │   ├── logging_setup.py
│       │   ├── metrics.py
│       │   ├── model_setup.py
│       │   ├── parsing.py
│       │   └── preprocessing.py
├── models/                       # Trained checkpoints and exports
├── notebooks/                    # Jupyter notebooks for experiments
├── main.py                       # Entry point: end-to-end training & evaluation
└── requirements.txt              # Python dependencies
```

## Key Components

### 1. Data & Datasets
- **`data/`**: Place raw image folders here, structured by split or unsplit.
- **`datasets/registry.py`**: Maps strings to custom `Dataset` classes for easy instantiation.

### 2. Training
- **Transfer learning**: `train.py` supports pre-trained backbones, freezing, class weights.
- **Knowledge Distillation**: `knowledge_distillation.py` combines soft-label loss, QAT, and callbacks.
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and LearningRateScheduler.

### 3. Utils
- **`model_setup.py`**: Generic model, loss, and optimizer configuration.
- **`preprocessing.py`**: Data loading, augmentation, weighted sampling, and logging of class distribution.
- **`parsing.py`**: Command-line and config file parsing with validation.
- **`metrics.py`**: Accuracy, precision, recall, F1, AUC, confusion matrix, ROC, radar charts.

### 4. Quantization
- **Post-Training**: FX and PT2E-based quantization in `quantization/core`.
- **QAT**: Eager, FX, and PT2E modes via `quantization/utils/mode_setup.py`.
- **Calibration**: `representative_data_gen` for TFLite.
- **Inspection**: TFLite op reports, checkpoint weight analysis, Numeric Suite SQNR.

### 5. Converters
- **ONNX**: Export PyTorch → ONNX → TensorFlow SavedModel.
- **TFLite**: Direct export via `ai_edge_torch` or TF LiteConverter (full & static quant).
- **ExecuTorch**: Compile to `.pte` for edge deployments.

### 6. Evaluation
- **PyTorch**: `pytorch_eval.py` for model inference, metrics, and plots.
- **ONNX**: `onnx_eval.py` using ONNX Runtime.
- **TensorFlow**: `tf_eval.py`, TFLite eval, and SavedModel evaluation.

---

## How to Use

### Orchid hierarchical pipeline

The orchid-specific workflow, reproducible configurations, deployment pack format,
and paper-ready experiment protocol are documented in
[`docs/orchid_experiments.md`](docs/orchid_experiments.md). The private images and
generated model artifacts remain excluded from version control.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/model-compression.git
cd model-compression
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train a Model
Run the `main.py` script to train and evaluate a model. Use command-line arguments or configuration files for customization.

#### Example Usage with Command-line Arguments
```bash
python main.py \
    --model_name mobilenet_v2 \
    --epochs 10 \
    --dataset SkinCancer \
    --batch_size 32 \
    --learning_rate 0.001 \
    --train_split 0.8 \
    --test_split 0.1 \
    --img_size 224 \
    --data_augmentation 1 \
    --callbacks ModelCheckpoint EarlyStopping \
    --ModelCheckpoint_monitor val_loss \
    --ModelCheckpoint_save_best_only \
    --EarlyStopping_patience 5 \
    --ReduceLROnPlateau_factor 0.1 \
    --save_dir ./saved_models
```

#### Example Usage with Configuration File
```bash
python main.py --config_file config.yaml
```

#### Example Usge Combining Configuration File and Command-line Arguments
```bash
python main.py --config_file config.yaml \
    --epochs 20 \
    --learning_rate 0.0005
```
- In this example, values from `config.yaml` will be overridden by the `epochs` and `learning_rate` arguments provided via the command line.

---

### 4. Explore Jupyter Notebooks
Use the notebooks in the `notebooks/` directory for further experimentation and analysis.

---

## Notes

- Ensure that datasets are correctly structured under the `data/` directory.
- The `main.py` script integrates various utilities, including callbacks, argument parsing, and model training, for an end-to-end training pipeline.
- Refer to the individual module documentation for detailed information on datasets, utilities, and training workflows.
