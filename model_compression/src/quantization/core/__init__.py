# model_compression/src/quantization/core/__init__.py
"""
ExecuTorch quantization core subpackage initialization.
Exports high-level quantization functions for PyTorch models.
"""
from .quantize import (
    quantize_pytorch_model,
    post_training_quantization_pytorch_model,
    post_training_quantization_pytorch_model_tflite,
    post_training_quantization_pytorch_model_tflite_legacy
)

__all__ = [
    'quantize_pytorch_model',
    'post_training_quantization_pytorch_model',
    'post_training_quantization_pytorch_model_tflite',
    'post_training_quantization_pytorch_model_tflite_legacy'
]
