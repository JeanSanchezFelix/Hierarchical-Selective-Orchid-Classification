
import logging
import numpy as np
import torch
import torch.nn as nn
import ai_edge_torch
from typing import Optional
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_fx import convert_fx

def quantize_pytorch_model(model: nn.Module, quant_mode: str, save_dir: Optional[str] = None) -> nn.Module:
    """
    Converts a calibrated/trained PyTorch model to a quantized model using the specified quantization mode.
    
    Parameters:
        model (nn.Module): The PyTorch model to convert.
        quant_mode (str): The quantization mode to use. Valid options are 'fx' or 'export'.
        save_dir (Optional[str]): Path to save the quantized model's state dictionary. If None, saving is skipped.
    
    Returns:
        nn.Module: The quantized model.
    
    Raises:
        ValueError: If an invalid quantization mode is specified.
    """
    # Convert the model based on the quantization mode.
    if quant_mode == "fx":
        logging.info("Quantizing model using FX Graph Mode.")
        quantized_model = convert_fx(model)
    elif quant_mode == "export":
        logging.info("Quantizing model using PT2E Export Mode.")
        quantized_model = convert_pt2e(model, fold_quantize=True)
    else:
        raise ValueError("Invalid mode. Choose either 'fx' or 'export'.")
    
    # Save the quantized model's state dictionary if a save directory is provided.
    if save_dir:
        torch.save(quantized_model.state_dict(), save_dir)
    return quantized_model


def convert_pytorch_model_to_tflite(model: nn.Module, save_dir: str, example_inputs: tuple[torch.Tensor, ...]) -> None:
    """
    Converts a PyTorch model to a TensorFlow Lite (TFLite) model and exports it to the specified directory.
    
    The model is first switched to evaluation mode for export, then converted using the ai_edge_torch 
    utility, and finally exported to the provided save directory.
    
    Parameters:
        model (nn.Module): The PyTorch model to convert.
        save_dir (str): The file path where the TFLite model will be saved.
        example_inputs (Tuple[Tensor, ...]): A tuple of example input tensors used for tracing during conversion.
    
    Returns:
        None
    """
    logging.info("Converting PyTorch model to TensorFlow Lite format.")
    # Ensure the model is in evaluation mode for export.
    torch.ao.quantization.move_exported_model_to_eval(model)
    # Convert the model using the ai_edge_torch conversion utility.
    edge_model = ai_edge_torch.convert(model, example_inputs)
    # Export the converted model to the specified directory.
    edge_model.export(save_dir)
    logging.info(f"TFLite model exported to {save_dir}")


def check_pytorch_to_tflite(torch_output: torch.Tensor, edge_output: np.ndarray) -> None:
    """
    Compares the outputs of a PyTorch model and its corresponding TensorFlow Lite model.
    
    Parameters:
        torch_output (Tensor): The output from the PyTorch model.
        edge_output (np.ndarray): The output from the TensorFlow Lite model.
    
    Returns:
        None: logging.infos a message indicating whether the outputs are within the specified tolerance.
    """
    # Compare the two outputs using np.allclose with specified tolerance.
    if np.allclose(
        torch_output.detach().cpu().numpy(),
        edge_output,
        atol=1e-5,
        rtol=1e-5,
    ):
        logging.info("Inference results from PyTorch and TFLite are within tolerance.")
    else:
        logging.info("Discrepancy detected between PyTorch and TFLite inference outputs.")

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
        logging.info(f"Model converted and saved to {tflite_model_path}")
    except Exception as e:
        logging.info(f"Error converting the model: {e}")

def check_quantized_modules(model: torch.nn.Module) -> None:
    """
    logging.infos out the module names and their types to help identify if quantized 
    versions of layers are present.
    
    Args:
        model (torch.nn.Module): The quantized model to inspect.
    """
    for name, module in model.named_modules():
        logging.info(f"{name}: {type(module)}")

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