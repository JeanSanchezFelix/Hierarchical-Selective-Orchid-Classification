import numpy as np
import logging
import torch

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