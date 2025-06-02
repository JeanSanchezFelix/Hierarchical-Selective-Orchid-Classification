import logging
import numpy as np
import torch

def check_pytorch_to_tflite(torch_output: torch.Tensor, edge_output: np.ndarray) -> None:
    """
    Compares outputs from a PyTorch model and its TFLite counterpart for similarity.

    Args:
        torch_output (torch.Tensor): Output from the PyTorch model.
        edge_output (np.ndarray): Output from the corresponding TFLite model.

    Logs:
        Info message indicating whether the outputs are within the specified tolerance.
    """
    torch_np = torch_output.detach().cpu().numpy()
    if np.allclose(torch_np, edge_output, atol=1e-5, rtol=1e-5):
        logging.info("Inference results from PyTorch and TFLite are within tolerance.")
    else:
        logging.info("Discrepancy detected between PyTorch and TFLite inference outputs.")
