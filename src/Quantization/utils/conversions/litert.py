import ai_edge_torch
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from torch.ao.quantization.quantize_pt2e import convert_pt2e

def quantize_pytorch_export_model(model: nn.Module, save_dir) -> nn.Module:
    """
    Convert a calibrated/trained model to a quantized model
    
    Parameters:
        model (nn.Module): The PyTorch model to convert.

    Returns:
        nn.Module: Quantized model.
    """
    print("Quantized PyTorch model.")
    quantized_model = convert_pt2e(model, fold_quantize=False)
    if save_dir:
        torch.save(quantized_model.state_dict(), save_dir)
    return quantized_model

def convert_pytorch_model_to_tflite(model: nn.Module, save_dir: str, example_inputs: tuple) -> None:
    """
    Convert a PyTorch model to a TensorFlow Lite model.
    
    Parameters:
        model (nn.Module): The PyTorch model to convert.

    Returns:
        nn.Module: The converted TensorFlow Lite model.
    """
    print("COnverted PyTorch model to tflite.")
    torch.ao.quantization.move_exported_model_to_eval(model)
    edge_model = ai_edge_torch.convert(model, example_inputs)
    edge_model.export(save_dir)

def check_pytorch_to_tflite(torch_output, edge_output):
    """
    Compare the output of a PyTorch model and a TensorFlow Lite model.
    """

    if (np.allclose(
        torch_output.detach().numpy(),
        edge_output,
        atol=1e-5,
        rtol=1e-5,
    )):
        print("Inference result with Pytorch and TfLite was within tolerance")
    else:
        print("Something wrong with Pytorch --> TfLite")

def convert_tensorflow_model_to_tflite(saved_model_dir: str, tflite_model_path: str) -> None:
    """
    Convert a TensorFlow SavedModel to TensorFlow Lite format.

    Args:
        saved_model_dir (str): Path to the TensorFlow SavedModel directory.
        tflite_model_path (str): Path to save the converted TensorFlow Lite model.
    """
    import tensorflow as tf
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
        print(f"Model converted and saved to {tflite_model_path}")
    except Exception as e:
        print(f"Error converting the model: {e}")

def check_quantized_modules(model: torch.nn.Module) -> None:
    """
    Prints out the module names and their types to help identify if quantized 
    versions of layers are present.
    
    Args:
        model (torch.nn.Module): The quantized model to inspect.
    """
    for name, module in model.named_modules():
        print(f"{name}: {type(module)}")

def is_quantized_model(model: torch.nn.Module) -> bool:
    """
    Checks if the model is quantized based on its type.
    
    The quantized model produced by export quantization is a GraphModule.
    This function returns True if the model's type name matches that.
    
    Args:
        model (torch.nn.Module): The model to check.
    
    Returns:
        bool: True if the model appears quantized, False otherwise.
    """
    return type(model).__name__ == "GraphModule"