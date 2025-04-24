# Converters

This folder contains utility scripts to convert PyTorch models into various target formats including ONNX, TFLite, and ExecuTorch PTE (Portable Torch Executable) files.

---

## Folder Structure

```
converters/
├── to_executorch.py
├── to_onnx.py
├── to_tflite.py
```

---

## Contents

- [Overview of Converters Module](#overview-of-the-converters-module)
- [Available Functions](#available-functions)
  - [to_executorch](#to_executorch)
  - [to_onnx](#to_onnx)
  - [to_tflite](#to_tflite)
- [How to Use](#how-to-use)

--- 

## Overview of the `converters` Module

The `converters` module facilitates model deployment across different platforms by providing end-to-end tools to:

- Convert standard or quantized PyTorch models to **ONNX** and **TensorFlow SavedModels**.
- Convert PyTorch models directly to **TFLite** using `ai_edge_torch`.
- Export optimized and quantized PyTorch models into **ExecuTorch PTE** format for edge deployment.

---

## Available Functions

### `to_executorch`

- `convert_to_executorch_program(model, example_inputs, save_path="model.pte", verbose=False)`
  - Converts a PyTorch model to ExecuTorch format using the Torch Export API and saves as `.pte`.

- `convert_quantized_to_edge_pte(quantized_model, example_inputs, save_path="quantized_model.pte")`
  - Converts a quantized PyTorch model to ExecuTorch `.pte` using export-to-edge APIs.

---

### `to_onnx`

- `export_pytorch_to_onnx(model, example_input, onnx_file_path, dynamo=True, opset_version=18)`
  - Exports a PyTorch model to ONNX format.

- `export_onnx_to_savedmodel(onnx_path, saved_model_dir)`
  - Converts an ONNX model to a TensorFlow SavedModel.

- `export_savedmodel_to_tflite(saved_model_dir, tflite_model_path, calibration_dataloader, num_calibration_steps=100)`
  - Converts a TensorFlow SavedModel to a fully quantized TFLite model using a representative dataset.

---

### `to_tflite`

- `convert_pytorch_model_to_tflite(model, example_inputs, save_dir)`
  - Converts a PyTorch model directly to a TFLite model using `ai_edge_torch`.

- `convert_tensorflow_model_to_tflite(saved_model_dir, tflite_model_path)`
  - Converts a TensorFlow SavedModel to a TFLite flatbuffer.

---

## How to Use

All scripts are designed to be imported as modules and used programmatically in your training or deployment pipeline.

### 1. Convert PyTorch → ONNX → TensorFlow → INT8 TFLite

```python
from converters.to_onnx import (
    export_pytorch_to_onnx,
    export_onnx_to_savedmodel,
    export_savedmodel_to_tflite
)

# Step 1: Export to ONNX
export_pytorch_to_onnx(model, example_input, "model.onnx")

# Step 2: ONNX → TF
export_onnx_to_savedmodel("model.onnx", "saved_model_dir")

# Step 3: TF → INT8 TFLite
export_savedmodel_to_tflite("saved_model_dir", "model.tflite", dataloader)
```

---

### 2. Convert PyTorch → ExecuTorch `.pte`

```python
from converters.to_executorch import convert_to_executorch_program

filename = convert_to_executorch_program(model, (input_tensor,), save_path="model.pte")
```

---

### 3. Convert Quantized PyTorch → ExecuTorch `.pte`

```python
from converters.to_executorch import convert_quantized_to_edge_pte

filename = convert_quantized_to_edge_pte(quant_model, (input_tensor,))
```

---

### 4. Convert PyTorch → TFLite via `ai_edge_torch`

```python
from converters.to_tflite import convert_pytorch_model_to_tflite

convert_pytorch_model_to_tflite(model, (example_tensor,), "output_dir")
```

---

### 5. Convert TF SavedModel → TFLite

```python
from converters.to_tflite import convert_tensorflow_model_to_tflite

convert_tensorflow_model_to_tflite("saved_model_dir", "model.tflite")
```