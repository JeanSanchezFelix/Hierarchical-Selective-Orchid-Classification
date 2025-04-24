import torch
import torch.nn as nn
import torch.ao.ns._numeric_suite_fx as ns
import matplotlib as plt

def debug_quantized_model(original_model: nn.Module, quantized_model: nn.Module) -> list[int]:
    """
    Comparing Weights Using the Numeric Suite.
    
    Parameters:
        original_model (nn.Module): The original model to comapre to.
        quantized_model (nn.Module): The quantized model to inspect.
    """
    print("Debugging quantized model parameters:")

    # Extract weight pairs between the original (float) and quantized models.
    # Here we label the weights from the float model as 'float' and from the quantized model as 'quantized'.
    weight_comparison = ns.extract_weights('float', original_model, 'quantized', quantized_model)

    # Extend the comparison dictionary by computing the SQNR (Signal-to-Quantization-Noise Ratio)
    ns.extend_logger_results_with_comparison(
        weight_comparison, 'float', 'quantized', torch.ao.ns.fx.utils.compute_sqnr, 'sqnr'
    )

    sqnr_list = []

    # Now, print the SQNR values for each weight pair for inspection.
    print("Weight Comparison (SQNR):")
    for name, comp in weight_comparison.items():
        # Each 'comp' contains entries for both sides and the computed SQNR.
        sqnr = comp['weight']['quantized'][0].get('sqnr', 'N/A')
        sqnr_list.append(sqnr[0])
        print(f"{name}: SQNR = {sqnr}")

    return sqnr_list

def plot_sqnr(xdata, ydata, xlabel, ylabel, title):
    # Create the plot
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Use plt.gca() to get the current axes (which uses data coordinates)
    ax = plt.gca()

    # Plot xdata vs ydata
    ax.plot(xdata, ydata)

    print(min(ydata))

    # Set the x and y axis limits to their respective min and max values
    ax.set_xlim([min(xdata)-1, max(xdata)+1])
    ax.set_ylim([min(ydata)-1, max(ydata)+1])

    # Set the x and y ticks to increment by 1
    ax.set_xticks(range(min(xdata), max(xdata), 2))  # Set x ticks from min to max with step 2
    # ax.set_yticks(range(min(ydata), max(ydata) + 1, 5))  # Set y ticks from min to max with step 1

    plt.tight_layout()

    # Show the plot
    plt.show()
