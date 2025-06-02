# model_compression/src/converters/__init__.py
"""
Converters package initialization.
Exports utilities for model format conversion (ExecuTorch, ONNX, TFLite).
"""
from .to_executorch import (
    save_pte_program,
    convert_to_executorch_program,
    convert_quantized_to_edge_pte
)
from .to_onnx import (
    export_pytorch_to_onnx,
    export_onnx_to_savedmodel
)
from .to_tflite import (
    convert_pytorch_model_to_tflite,
    convert_tensorflow_model_to_tflite,
    convert_to_static_quant_tflite
)

__all__ = [
    'save_pte_program',
    'convert_to_executorch_program',
    'convert_quantized_to_edge_pte',
    'export_pytorch_to_onnx',
    'export_onnx_to_savedmodel',
    'convert_pytorch_model_to_tflite',
    'convert_tensorflow_model_to_tflite',
    'convert_to_static_quant_tflite'
]
