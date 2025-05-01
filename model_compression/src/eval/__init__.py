# model_compression/src/eval/__init__.py
"""
Evaluation module initialization.
Exports functions for evaluating models across frameworks (PyTorch, ONNX, TFLite, SavedModel).
"""
from .pytorch_eval import test_inference, evaluate
from .onnx_eval import test_inference_onnx
from .tflite_eval import test_inference_tflite
from .tf_eval import test_inference_savedmodel
from .predictions import _compute_loss_and_predictions, _compute_predictions

__all__ = [
    'test_inference',
    'evaluate',
    'test_inference_onnx',
    'test_inference_tflite',
    'test_inference_savedmodel',
    '_compute_loss_and_predictions',
    '_compute_predictions'
]
