import torch
import torch.nn as nn
import torch.ao.ns._numeric_suite_fx as ns
from typing import List, Any

import matplotlib.pyplot as plt

def debug_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module
) -> List[float]:
    """
    Compare weights between float and quantized models using Numeric Suite and compute SQNR.

    Args:
        original_model: The original float PyTorch model.
        quantized_model: The quantized PyTorch model to inspect.

    Returns:
        List of SQNR (Signal-to-Quantization-Noise Ratio) values for each weight tensor.

    Raises:
        RuntimeError: If weight extraction or SQNR computation fails.
    """
    logging_prefix = "[Numeric Suite]"
    print(f"{logging_prefix} Debugging quantized model parameters...")

    try:
        # Extract weight pairs between the original (float) and quantized models.
        # Here we label the weights from the float model as 'float' and from the quantized model as 'quantized'.
        weight_comp = ns.extract_weights(
            'float', original_model,
            'quant', quantized_model
        )
         # Extend the comparison dictionary by computing the SQNR (Signal-to-Quantization-Noise Ratio)
        ns.extend_logger_results_with_comparison(
            weight_comp,
            'float', 'quant',
            ns.utils.compute_sqnr,
            'sqnr'
        )
    except Exception as e:
        raise RuntimeError(f"Failed numeric suite weight comparison: {e}") from e

    sqnr_values: List[float] = []
    print(f"{logging_prefix} Weight Comparison (SQNR):")
    for name, comp in weight_comp.items():
        try:
            sqnr_entry = comp['weight']['quant'][0]['sqnr']
            sqnr_val = float(sqnr_entry)
        except Exception:
            sqnr_val = float('nan')
        sqnr_values.append(sqnr_val)
        print(f"  {name}: SQNR = {sqnr_val}")

    return sqnr_values


def plot_sqnr(
    xdata: List[Any],
    ydata: List[float],
    xlabel: str,
    ylabel: str,
    title: str
) -> None:
    """
    Plot SQNR values against a given x-axis data list.

    Args:
        xdata: List of x-axis values (e.g., layer indices).
        ydata: Corresponding SQNR values.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Plot title.

    Raises:
        ValueError: If xdata and ydata lengths mismatch or are empty.
    """
    if not xdata or not ydata or len(xdata) != len(ydata):
        raise ValueError("xdata and ydata must be non-empty lists of equal length.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xdata, ydata, marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set axis limits with margins
    ax.set_xlim(min(xdata) - 1, max(xdata) + 1)
    ax.set_ylim(min(ydata) - 1, max(ydata) + 1)

    # Configure ticks
    ax.set_xticks(xdata)
    plt.tight_layout()
    plt.show()