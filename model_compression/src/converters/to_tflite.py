
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Generator, List

from model_compression.src.quantization.utils.inspect import is_quantized_model

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import ai_edge_torch
except ImportError:
    ai_edge_torch = None 

def convert_pytorch_model_to_tflite(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    save_dir: str,
    device: torch.device
) -> None:
    """
    Converts a PyTorch model to a TensorFlow Lite (TFLite) model and exports it to the specified directory.
    
    The model is first switched to evaluation mode for export, then converted using the ai_edge_torch 
    utility, and finally exported to the provided save directory.
    
    Args:
        model: PyTorch model to convert (should be in eval mode).
        example_inputs: Example inputs for tracing conversion.
        save_dir: Directory or file path to save the TFLite model.

    Raises:
        ImportError: If TensorFlow or ai_edge_torch is not installed.
        IOError: If export or file writing fails.
    """
    if tf is None or ai_edge_torch is None:
        raise ImportError("TensorFlow and ai_edge_torch are required for PyTorch->TFLite conversion.")

    logging.info("Converting PyTorch model to TensorFlow Lite format.")

    # Ensure the model is in evaluation mode for export.
    if is_quantized_model(model):
        # restores eval/train to call move_exported_model_* under the hood
        model = torch.ao.quantization.allow_exported_model_train_eval(model)

    model.to(device).eval()

    # Convert the model to tflite
    try:
        edge_model = ai_edge_torch.convert(model, example_inputs)
    except Exception as e:
        logging.error(f"ai_edge_torch conversion failed: {e}")
        raise RuntimeError("Conversion to edge model failed.") from e

    # Export the converted model to the specified directory.
    try:
        os.makedirs(os.path.dirname(save_dir) or '.', exist_ok=True)
        edge_model.export(save_dir)
        logging.info(f"TFLite model exported to {save_dir}")
    except Exception as e:
        logging.error(f"Failed to export TFLite model: {e}")
        raise IOError(f"Could not save TFLite model to {save_dir}") from e

def convert_tensorflow_model_to_tflite(
    saved_model_dir: str,
    tflite_model_path: str
)-> None:
    """
    Convert a TensorFlow SavedModel to TensorFlow Lite format.

    Args:
        saved_model_dir: Path to TensorFlow SavedModel directory.
        tflite_model_path: File path to save the TFLite FlatBuffer.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If the SavedModel directory does not exist.
        RuntimeError: If conversion fails.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for SavedModel->TFLite conversion.")
    if not os.path.isdir(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

    # Load the SavedModel and create a TFLiteConverter.
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Enable the new converter (if not already default) and allow select TensorFlow ops.
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,   # Use built-in TFLite ops.
        # tf.lite.OpsSet.SELECT_TF_OPS      # Fallback for ops not natively supported.
    ]
    
    # Optionally, you can enable optimizations (e.g., full integer quantization)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    try:
        tflite_model = converter.convert()
        os.makedirs(os.path.dirname(tflite_model_path) or '.', exist_ok=True)
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        logging.info(f"Model converted and saved to {tflite_model_path}")
    except Exception as e:
        logging.info(f"Error converting the model: {e}")
        raise RuntimeError("Error converting the model") from e

def convert_to_static_quant_tflite(
    saved_model_dir: str,
    output_tflite_path: str,
    representative_dataset: Generator[List[np.ndarray], None, None],
    supported_ops: Optional[List[tf.lite.OpsSet]] = None,
    inference_input_type: Optional[tf.dtypes.DType] = tf.int8,
    inference_output_type: Optional[tf.dtypes.DType] = tf.int8,
) -> None:
    """
    Convert a SavedModel to a fully static-quantized TFLite model.

    Args:
        saved_model_dir: Directory of the TensorFlow SavedModel.
        output_tflite_path: File path to save the quantized TFLite model.
        representative_dataset: Generator for calibration data, yielding numpy arrays inputs.
        supported_ops: List of tf.lite.OpsSet; defaults to int8 builtin ops.
        inference_input_type: TF dtype for model input (e.g., tf.uint8).
        inference_output_type: TF dtype for model output.

    Raises:
        ImportError: If TensorFlow is not installed.
        FileNotFoundError: If SavedModel directory missing.
        ValueError: If conversion or calibration fails.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for static quantization.")
    if not os.path.isdir(saved_model_dir):
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

    # Create the TFLiteConverter from the SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Enable default optimizations (for quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set supported operations, defaulting to full int8 if none provided
    converter.target_spec.supported_ops = (
        supported_ops or [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    )
    # Force input and output tensors to desired dtypes
    if inference_input_type:
        converter.inference_input_type = inference_input_type
    if inference_output_type:
        converter.inference_output_type = inference_output_type
    # Assign representative dataset for calibration
    converter.representative_dataset = representative_dataset

    try:
        # Perform the conversion to a TFLite FlatBuffer
        quantized_model = converter.convert()
        os.makedirs(os.path.dirname(output_tflite_path) or '.', exist_ok=True)
        with open(output_tflite_path, 'wb') as f:
                f.write(quantized_model)
        logging.info(f"Quantized TFLite model saved at {output_tflite_path}")
    except Exception as e:
        logging.error(f"Static quantization failed: {e}")
        raise ValueError(f"Failed to convert to static quant TFLite: {e}") from e