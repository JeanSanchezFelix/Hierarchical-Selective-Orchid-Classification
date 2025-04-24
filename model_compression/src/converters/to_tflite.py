
import logging
import numpy as np
import torch
import torch.nn as nn

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