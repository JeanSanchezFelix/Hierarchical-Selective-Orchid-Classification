
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Generator, List

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import ai_edge_torch
except ImportError:
    ai_edge_torch = None 

def convert_pytorch_model_to_tflite(model: nn.Module, example_inputs: tuple[torch.Tensor, ...], save_dir: str,) -> None:
    """
    Converts a PyTorch model to a TensorFlow Lite (TFLite) model and exports it to the specified directory.
    
    The model is first switched to evaluation mode for export, then converted using the ai_edge_torch 
    utility, and finally exported to the provided save directory.
    
    Args:
        model (nn.Module): The PyTorch model to convert.
        save_dir (str): The file path where the TFLite model will be saved.
        example_inputs (Tuple[Tensor, ...]): A tuple of example input tensors used for tracing during conversion.
    
    Returns:
        None
    """
    logging.info("Converting PyTorch model to TensorFlow Lite format.")
    # Ensure the model is in evaluation mode for export.
    # model = torch.ao.quantization.move_exported_model_to_eval(model)
    # Convert the model using the ai_edge_torch conversion utility.
    edge_model = ai_edge_torch.convert(model, example_inputs)
    # Export the converted model to the specified directory.
    edge_model.export(save_dir)
    logging.info(f"TFLite model exported to {save_dir}")

def convert_tensorflow_model_to_tflite(saved_model_dir: str, tflite_model_path: str) -> None:
    """
    Convert a TensorFlow SavedModel to TensorFlow Lite format.

    Args:
        saved_model_dir (str): Path to the TensorFlow SavedModel directory.
        tflite_model_path (str): Path to save the converted TensorFlow Lite model.
    """
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
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        logging.info(f"Model converted and saved to {tflite_model_path}")
    except Exception as e:
        logging.info(f"Error converting the model: {e}")

def convert_to_static_quant_tflite(
    saved_model_dir: str,
    output_tflite_path: str,
    representative_dataset: Generator[List[np.ndarray], None, None],
    supported_ops: Optional[List[tf.lite.OpsSet]] = None,
    inference_input_type: tf.dtypes.DType = tf.uint8,
    inference_output_type: tf.dtypes.DType = tf.uint8,
) -> None:
    """
    Convert a SavedModel to a fully static-quantized TFLite model.

    Args:
        saved_model_dir (str): Path to the TensorFlow SavedModel directory.
        output_tflite_path (str): File path to write the quantized TFLite model.
        representative_dataset (Generator): Generator yielding representative samples.
        supported_ops (List[tf.lite.OpsSet], optional): Supported ops sets. Defaults to TFLITE_BUILTINS_INT8.
        inference_input_type (tf.dtypes.DType): Data type for input tensor. Defaults to tf.uint8.
        inference_output_type (tf.dtypes.DType): Data type for output tensor. Defaults to tf.uint8.

    Raises:
        ValueError: If conversion fails or parameters are invalid.
    """
    # Create the TFLiteConverter from the SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Enable default optimizations (for quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Set supported operations, defaulting to full int8 if none provided
    converter.target_spec.supported_ops = (
        supported_ops or [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    )
    # Force input and output tensors to desired dtypes
    converter.inference_input_type = inference_input_type
    converter.inference_output_type = inference_output_type
    # Assign representative dataset for calibration
    converter.representative_dataset = representative_dataset

    try:
        # Perform the conversion to a TFLite FlatBuffer
        quantized_model = converter.convert()
    except Exception as e:
        # Raise on failure to convert
        raise ValueError(f"TFLite conversion failed: {e}")

    # Write the quantized model to disk
    with open(output_tflite_path, 'wb') as f:
        f.write(quantized_model)