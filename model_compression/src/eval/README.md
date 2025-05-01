# Eval Module

The `eval` module provides utilities for running inference and evaluating trained models across multiple frameworks: PyTorch, ONNX Runtime, TFLite, and TensorFlow SavedModel. It also includes helper functions for computing losses and predictions.

---

## Folder Structure
```plaintext
eval/
├── __init__.py
├── predictions.py        # Compute loss, probabilities, and predictions
├── pytorch_eval.py       # Evaluate PyTorch models and SavedModel-style evaluation
├── onnx_eval.py          # Evaluate ONNX models with ONNX Runtime
├── tflite_eval.py        # Evaluate TFLite models with TensorFlow Lite Interpreter
└── tf_eval.py            # Evaluate TensorFlow SavedModel directories
```

---
## Contents

- [Overview](#modules-overview)
- [How to Use](#how-to-use)

---

## Modules Overview

### `predictions.py`
- **Functions:**
  - `_compute_loss_and_predictions(outputs, labels, criterion)`
    - Calculates loss, probabilities, and predicted labels for various loss functions (BCE, CrossEntropy, NLL).
  - `_compute_predictions(outputs)`
    - Derives probabilities and predictions from raw logits for binary or multiclass.

### `pytorch_eval.py`
- **Functions:**
  - `test_inference(model, data_loader, device, criterion=None, save_dir=None)`
    - Runs inference on a PyTorch model, computes metrics, and optionally saves plots (bar chart, confusion matrix, ROC-AUC, calibration curve).
  - `evaluate(model_path, metadata_path, img_size, dataset_name, save_dir=None)`
    - Loads saved state and metadata, prepares dataset and model, then calls `test_inference`.

### `onnx_eval.py`
- **Function:**
  - `test_inference_onnx(onnx_model_path, data_loader, save_dir=None)`
    - Uses ONNX Runtime to run inference on an ONNX model, computes metrics, and optionally saves plots.

### `tflite_eval.py`
- **Function:**
  - `test_inference_tflite(tflite_model_path, test_dataset, input_type=None, save_dir=None)`
    - Runs inference using TensorFlow Lite Interpreter on a `tf.data.Dataset`, computes metrics, and saves plots.

### `tf_eval.py`
- **Function:**
  - `test_inference_savedmodel(saved_model_path, test_dir, batch_size=32, image_size=(224,224), save_dir=None)`
    - Loads and runs a TensorFlow SavedModel on images in a directory structure (`test_dir/test`), computes metrics, and saves plots.

---

## How to Use

### PyTorch Model Evaluation
```python
from model_compression.src.eval import test_inference, evaluate

# Direct inference
metrics = test_inference(
    model=my_pytorch_model,
    data_loader=test_loader,
    device=torch.device('cuda'),
    criterion=loss_fn,
    save_dir='./eval_plots'
)

# Full evaluate from saved files
evaluate(
    model_path='./models/resnet_best.pth',
    metadata_path='./models/metadata.pth',
    img_size=224,
    dataset_name='MonkeyPox',
    save_dir='./eval_full'
)
```

### ONNX Model Evaluation
```python
from model_compression.src.eval import test_inference_onnx

metrics = test_inference_onnx(
    onnx_model_path='model.onnx',
    data_loader=test_loader,
    save_dir='./onnx_eval'
)
```

### TFLite Model Evaluation
```python
from model_compression.src.eval import evaluate_tflite_model

# test_dataset is a tf.data.Dataset yielding (image, label)
metrics = evaluate_tflite_model(
    tflite_model_path='model.tflite',
    test_dataset=test_dataset,
    save_dir='./tflite_eval'
)
```

### TensorFlow SavedModel Evaluation
```python
from model_compression.src.eval import test_inference_savedmodel

metrics = test_inference_savedmodel(
    saved_model_path='saved_model_dir',
    test_dir='./data/skin_lesions',
    batch_size=16,
    image_size=(224,224),
    save_dir='./tf_eval'
)
```

---

For detailed API and advanced options, refer to the source code in the `eval` folder.

