# Converters

The converters module provides utilities to transform PyTorch models into various target formats for deployment, including ONNX, TensorFlow SavedModel, TFLite, and ExecuTorch PTE files.

---

## Folder Structure

```
converters/
├── __init__.py
├── to_executorch.py # ExecuTorch PTE conversion
├── to_onnx.py         # ONNX and ONNX→SavedModel conversion
└── to_tflite.py       # PyTorch/TensorFlow→TFLite conversion
```

---

## Contents

- [Overview of Converters Module](#overview-of-the-converters-module)
- [Available Functions](#available-functions)
  - [to_executorch](#executorch-pte)
  - [to_onnx](#onnx)
  - [to_tflite](#tflite)
- [How to Use](#how-to-use)

--- 

## Overview of the `converters` Module

The `converters` module facilitates model deployment across different platforms by providing end-to-end tools to:

- Convert standard or quantized PyTorch models to **ONNX** and **TensorFlow SavedModels**.
- Convert PyTorch models directly to **TFLite** using `ai_edge_torch`.
- Export optimized and quantized PyTorch models into **ExecuTorch PTE** format for edge deployment.

---

## Available Functions

### ExecuTorch PTE

- **`convert_to_executorch_program(model, example_inputs, save_path="model.pte", verbose=False)`**  
  Convert a PyTorch model into an ExecuTorch edge-optimized IR and save as `.pte`.

- **`convert_quantized_to_edge_pte(quantized_model, example_inputs, save_path="quantized_model.pte")`**  
  Compile a quantized PyTorch model into a PTE program for edge deployment.


### ONNX

- **`export_pytorch_to_onnx(model, example_input, onnx_file_path, dynamo=True, opset_version=18)`**  
  Export a PyTorch model to ONNX, with optional dynamic batch support.

- **`export_onnx_to_savedmodel(onnx_path, saved_model_dir)`**  
  Convert an ONNX model into a TensorFlow SavedModel via `onnx-tf` backend.


### TFLite 

- **`convert_pytorch_model_to_tflite(model, example_inputs, save_dir)`**  
  Use `ai_edge_torch` to export a PyTorch model directly to a TFLite FlatBuffer.

- **`convert_tensorflow_model_to_tflite(saved_model_dir, tflite_model_path)`**  
  Convert a TensorFlow SavedModel to TFLite using the official TFLiteConverter.

- **`convert_to_static_quant_tflite(saved_model_dir, output_tflite_path, representative_dataset, ...)`**  
  Generate a fully int8-quantized TFLite model with a representative dataset for calibration.

---

## How to Use

Below are common workflows for converting models:

### 1. PyTorch → ONNX → TensorFlow SavedModel → TFLite
```python
from model_compression.src.converters.to_onnx import (
    export_pytorch_to_onnx,
    export_onnx_to_savedmodel
)
from model_compression.src.converters.to_tflite import convert_tensorflow_model_to_tflite

# 1. Export PyTorch to ONNX
export_pytorch_to_onnx(model, example_input, "model.onnx")

# 2. Convert ONNX to SavedModel
export_onnx_to_savedmodel("model.onnx", "saved_model_dir")

# 3. Convert SavedModel to TFLite
convert_tensorflow_model_to_tflite("saved_model_dir", "model.tflite")
```

### 2. PyTorch → ExecuTorch `.pte`
```python
from model_compression.src.converters.to_executorch import convert_to_executorch_program

pte_path = convert_to_executorch_program(
    model, (input_tensor,), save_path="model.pte", verbose=True
)
```

### 3. Quantized PyTorch → ExecuTorch `.pte`
```python
from model_compression.src.converters.to_executorch import convert_quantized_to_edge_pte

pte_path = convert_quantized_to_edge_pte(
    quantized_model, (input_tensor,)
)
```

### 4. PyTorch → TFLite via `ai_edge_torch`
```python
from model_compression.src.converters.to_tflite import convert_pytorch_model_to_tflite

convert_pytorch_model_to_tflite(
    model, (example_tensor,), "output_model.tflite"
)
```

### 5. SavedModel → Static INT8 TFLite
```python
from model_compression.src.converters.to_tflite import convert_to_static_quant_tflite

# representative_dataset yields numpy arrays for calibration
convert_to_static_quant_tflite(
    "saved_model_dir",
    "quant_model.tflite",
    representative_dataset=lambda: (x for x, _ in calib_loader)
)
```

---

For detailed API and advanced options, refer to the source code in the `converters` folder.