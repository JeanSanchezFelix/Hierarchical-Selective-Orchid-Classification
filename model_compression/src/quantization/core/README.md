# Quantization Core Module

The `quantization/core` subpackage contains utilities to quantize PyTorch models using FX Graph Mode or PT2E Export Mode and export them for deployment via ai_edge_torch.

---

## Folder Structure
```plaintext
quantization/core/
├── __init__.py
└── quantize.py         # Main quantization functions
```

## Contents

- [Overview](#modules-overview)
- [How to Use](#how-to-use)

---

## Modules Overview

All functions are located in `quantize.py` and can be imported directly from the `quantization.core` package.

### `quantize_pytorch_model`
```python
quantize_pytorch_model(
    model: nn.Module,
    quant_mode: str,
    save_path: Optional[str] = None
) -> nn.Module
```
- **Description:** Quantize a PyTorch model using either FX Graph Mode (`"fx"`) or PT2E Export Mode (`"export"`).
- **Args:**
  - `model`: Trained PyTorch model.
  - `quant_mode`: One of `"fx"` or `"export"`.
  - `save_path`: Optional path to save the quantized state dict.
- **Returns:** Quantized PyTorch model.

### `post_training_quantization_pytorch_model`
```python
post_training_quantization_pytorch_model(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    save_path: Optional[str] = None
) -> Any
```
- **Description:** Perform PT2E post-training quantization, calibrate with example inputs, convert to a quantized model, and export via `ai_edge_torch`.
- **Args:**
  - `model`: Float PyTorch model.
  - `example_inputs`: Example inputs for calibration.
  - `save_path`: Path to export the quantized TFLite model.
- **Returns:** Exported `ai_edge_torch` model instance.

### `post_training_quantization_pytorch_model_tflite`
```python
post_training_quantization_pytorch_model_tflite(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    recipe: str = 'full_int8_dynamic',
    save_path: Optional[str] = None
) -> Any
```
- **Description:** Export a post-training quantized TFLite model using a quantization recipe.
- **Args:**
  - `recipe`: One of `'full_int8_dynamic'`, `'full_int8_weight_only'`, `'full_fp16'`.

### `post_training_quantization_pytorch_model_tflite_legacy`
```python
post_training_quantization_pytorch_model_tflite_legacy(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    data_loader: Any,
    save_path: Optional[str] = None
) -> Any
```
- **Description:** Legacy flow using a representative dataset for quantization via TFLite converter flags.

---

## How to use

```python
from model_compression.src.quantization.core import (
    quantize_pytorch_model,
    post_training_quantization_pytorch_model,
    post_training_quantization_pytorch_model_tflite
)

# Simple FX quantization
quant_model = quantize_pytorch_model(model, quant_mode='fx', save_path='model_fx.pth')

# PT2E quantization and ai_edge_torch export
tq_model = post_training_quantization_pytorch_model(
    model, example_inputs=(input_tensor,), save_path='model_pt2e.tflite'
)

# Recipe-based TFLite export via ai_edge_torch
tfl_model = post_training_quantization_pytorch_model_tflite(
    model, (input_tensor,), recipe='full_int8_dynamic', save_path='model_recipe.tflite'
)
```

---

Refer to each function's docstring for full parameter details and advanced options.

