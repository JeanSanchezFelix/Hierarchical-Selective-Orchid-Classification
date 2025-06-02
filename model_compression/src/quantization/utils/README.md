# Quantization Utils Module

The `quantization/utils` subpackage provides helper functions for preparing, calibrating, validating, and inspecting PyTorch models during quantization workflows.

---

## Folder Structure
```plaintext
quantization/utils/
├── __init__.py
├── calibration.py                # Representative dataset generator for TFLite calibration
├── inspect.py                    # Inspect and report on quantized models and TFLite ops
├── validate.py                   # Validate consistency between PyTorch and TFLite outputs
├── mode_setup.py                 # Prepare models for QAT (eager, fx, export) and KD setup
└── post_training_quantization.py # Debug and plot SQNR for post-training quantized models
```

---

## Contents

- [Overview](#modules-overview)
- [How to Use](#how-to-use)

--

## Modules Overview

### Calibration (`calibration.py`)
- **`representative_data_gen(dataloader, num_samples=100)`**
  - Generates TensorFlow tensors from PyTorch `DataLoader` batches for TFLite calibration.

### Inspection (`inspect.py`)
- **`check_quantized_modules(model)`**
  - Logs module names/types to verify quantized layers.
- **`is_quantized_model(model)`**
  - Returns `True` if the model is a quantized `GraphModule`.
- **`report_tflite_ops(tflite_model_path)`**
  - Reports counts of built-in vs. fallback (SELECT_TF_OPS) operators in a `.tflite` file.
- **`inspect_pth_weights(pth_file)`**
  - Prints stats on quantized (INT8) vs. non-quantized (FP32) tensors in a PyTorch `.pth`.

### Validation (`validate.py`)
- **`check_pytorch_to_tflite(torch_output, edge_output)`**
  - Compares PyTorch and TFLite inference outputs for numeric closeness.

### QAT & KD Setup (`mode_setup.py`)
- **`quantization_mode(model, mode, example_inputs=None, config=None)`**
  - Prepare model for QAT in `eager`, `fx`, or `export` mode.
- **`kd_setup(teacher_name, student_name, learning_rate, criterion_name, optimizer_name, dataloader, ...)`**
  - Set up teacher and student models, criterion, and optimizer for Knowledge Distillation with optional QAT.

### Post-Training Quantization (`post_training_quantization.py`)
- **`debug_quantized_model(original_model, quantized_model)`**
  - Uses PyTorch Numeric Suite to compare weights and compute SQNR.
- **`plot_sqnr(xdata, ydata, xlabel, ylabel, title)`**
  - Plot SQNR values for visual analysis.

---

## How to Use

```python
from model_compression.src.quantization.utils import (
    representative_data_gen,
    report_tflite_ops,
    check_pytorch_to_tflite,
    quantization_mode,
    kd_setup,
    debug_quantized_model,
    plot_sqnr
)

# Generate calibration data for TFLite
rep_samples = representative_data_gen(train_loader, num_samples=200)

# Inspect if a model is quantized
print(is_quantized_model(q_model))

# Prepare model for QAT in FX mode
qat_model = quantization_mode(model, 'fx', example_inputs=(torch.rand(1,3,224,224),), config='qnnpack')

# KD setup
teacher, student, loss_fn, opt = kd_setup(...)

# Debug SQNR
sqnr_vals = debug_quantized_model(model_float, model_quant)
plot_sqnr(list(range(len(sqnr_vals))), sqnr_vals, 'Layer', 'SQNR', 'Quantization Quality')

# Report TFLite ops
report_tflite_ops('model.tflite')
```

For detailed API and advanced options, refer to the source code in the `utils` folder.

