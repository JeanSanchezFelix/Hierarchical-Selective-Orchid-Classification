import torch
import logging
import torch.nn as nn
from typing import Optional
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_fx import convert_fx

def quantize_pytorch_model(model: nn.Module, quant_mode: str, save_dir: Optional[str] = None) -> nn.Module:
    """
    Converts a calibrated/trained PyTorch model to a quantized model using the specified quantization mode.
    
    Args:
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