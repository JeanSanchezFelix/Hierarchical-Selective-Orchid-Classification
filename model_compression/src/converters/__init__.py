# model_compression/src/converters/__init__.py
"""Converter utilities with optional format-specific dependencies."""

from .to_onnx import (
    export_pytorch_to_onnx,
    export_onnx_to_savedmodel,
)
from .to_tflite import (
    convert_pytorch_model_to_tflite,
    convert_tensorflow_model_to_tflite,
    convert_to_static_quant_tflite,
)

__all__ = [
    "export_pytorch_to_onnx",
    "export_onnx_to_savedmodel",
    "convert_pytorch_model_to_tflite",
    "convert_tensorflow_model_to_tflite",
    "convert_to_static_quant_tflite",
]

try:
    from .to_executorch import (
        convert_quantized_to_edge_pte,
        convert_to_executorch_program,
        save_pte_program,
    )
except ImportError:
    # ExecuTorch is optional; importing a TFLite or ONNX converter must not
    # require its separate runtime package.
    pass
else:
    __all__ += [
        "save_pte_program",
        "convert_to_executorch_program",
        "convert_quantized_to_edge_pte",
    ]
