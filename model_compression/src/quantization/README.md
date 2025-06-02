# Quantization Module

The `quantization` module provides end-to-end support for both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) workflows, along with utilities for Knowledge Distillation (KD) integration.

---

## Folder Structure
```plaintext
quantization/
├── core/                    # Core PTQ and export quantization routines
│   ├── __init__.py
│   └── quantize.py          # FX & PT2E quantization and ai_edge_torch exports
│
└── utils/                   # Helper functions for QAT preparation, calibration, and inspection
    ├── __init__.py
    ├── calibration.py       # Representative dataset generator for TFLite calibration
    ├── inspect.py           # Inspect models and TFLite operator usage
    ├── validate.py          # Validate PyTorch vs. TFLite inference outputs
    ├── mode_setup.py        # Prepare models for QAT and KD setup
    └── post_training_quantization.py  # Numeric Suite debugging and SQNR plotting
```

---

## Submodules

### Core (`quantization/core`)

Contains functions to perform:
- **FX Graph Mode Quantization**: `quantize_pytorch_model(model, 'fx', save_path)`
- **PT2E Export Mode Quantization**: `quantize_pytorch_model(model, 'export', save_path)`
- **Post-Training Quantization Export**: various `post_training_quantization_*` functions for ai_edge_torch TFLite export.

Refer to [quantize.py](core/quantize.py) for detailed APIs and examples.

### Utils (`quantization/utils`)

Provides supporting utilities:
- **Calibration**: Representative data generator for calibration: `representative_data_gen`
- **Inspection**: Check quantized layers and report TFLite operators: `check_quantized_modules`, `report_tflite_ops`.
- **Validation**: Numeric comparison of PyTorch vs. TFLite outputs: `check_pytorch_to_tflite`.
- **QAT Preparation**: Prepare models for QAT in `eager`, `fx`, or `export` modes and KD setup: `quantization_mode`, `kd_setup`.
- **Post-Training Debug**: Compare weights and plot SQNR for quantized models: `debug_quantized_model`, `plot_sqnr`.

Refer to each file under `utils/` for full documentation.

---

## Getting Started

### 1. Post-Training Quantization (PTQ)
```python
from model_compression.src.quantization.core import quantize_pytorch_model

# FX Graph Mode
qt_model = quantize_pytorch_model(model, 'fx', save_path='fx_model.pth')
# PT2E Export Mode
qt_model = quantize_pytorch_model(model, 'export', save_path='pt2e_model.pth')
```

### 2. Quantization-Aware Training (QAT)
```python
from model_compression.src.quantization.utils import quantization_mode

# Eager QAT
qat_model = quantization_mode(model, 'eager', config='qnnpack')
```

### 3. Quantization + KD Setup
```python
from model_compression.src.quantization.utils import kd_setup

teacher, student, criterion, optimizer = kd_setup(
    teacher_name='resnet50',
    student_name='mobilenet_v2',
    learning_rate=1e-3,
    criterion_name='cross_entropy',
    optimizer_name='adam',
    dataloader=train_loader,
    quant_mode='fx',
    config='qnnpack'
)
```

### 4. TFLite Calibration & Export
```python
from model_compression.src.quantization.utils import representative_data_gen
from model_compression.src.converters.to_tflite import convert_to_static_quant_tflite

rep_ds = representative_data_gen(train_loader, num_samples=100)
convert_to_static_quant_tflite(
    'saved_model_dir', 'quant_model.tflite', rep_ds
)
```

---

For more details, explore the docstrings in each module and example notebooks under `notebooks/`.

