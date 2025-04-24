import logging
import torch

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